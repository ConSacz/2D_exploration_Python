try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import numpy as np
import networkx as nx
from utils.genarea import genarea
from utils.graph_functions import Graph, Connectivity_graph
from utils.Cov_Func import Cov_Func
from utils.plot_functions import plot2Ddeployment
from utils.Workspace_functions import save_mat

# %%
cases = ["image0", "image1", "image2", "image3", "image4"]
for case in cases:
    
    for Trial in range(1,50):
        
        frames = []   # lưu toàn bộ hình để ghép GIF
        
        
        # Network parameter
        Obstacle_Area = genarea(case)
        Covered_Area = np.zeros_like(Obstacle_Area)
        obs_row, obs_col = np.where(Obstacle_Area == 0)
        
        # Node info
        MaxIt = 500
        a = 1
        N = 50
        rc = 16
        rs = 8 * np.ones(N)
        sink = [5, 5]
        trap_thresh = 10
        float_thresh = 100
        trap2float = float_thresh/10
        v = 5
        safe_d = 1
        
        # %% Init pop
        initpop = np.random.uniform(max(sink[0]-rc/2, 2), sink[1]+rc/2, (N, 2))
        initpop[0, :] = sink
        pop = initpop.copy()
        del initpop
        
        no_move_counts = np.zeros(N)
        no_profit_move_counts = np.zeros(N)
        trap_matrix = np.zeros(N)
        BestCostIt = np.zeros(MaxIt)
        popIt = np.zeros((MaxIt, 2*N))
        popIt[0, :] = pop.flatten()
        
        BestCostIt[0], Covered_Area = Cov_Func(pop, rs, Obstacle_Area, Covered_Area)
        # frames = plot2Ddeployment(pop, rs, rc, BestCostIt[0], 0, Obstacle_Area, Covered_Area, frames)
        
        # %% Main
        for it in range(1, MaxIt):
            G = Graph(pop, rc)
            al_trap_matrix = np.zeros(N)
            for l in range(1, N):
                K = list(G.neighbors(l))
                if K:
                    al_trap_matrix[l] = no_move_counts[l] + np.mean(trap_matrix[K])
            trap_matrix = al_trap_matrix
            
            # Decision order
            distance_dict = nx.single_source_shortest_path_length(G, 0)
            N_layers = max(distance_dict.values())
        
            nodesInLayers = [[[] for _ in range(N)] for _ in range(N_layers)]
            orderInLayers = [[] for _ in range(N_layers)]
        
            for d in range(1, N_layers + 1):
                nodesInLayer = [k for k, v in distance_dict.items() if v == d]
                sub_G = G.subgraph(nodesInLayer)
                comps = list(nx.connected_components(sub_G))
                for i, comp in enumerate(comps):
                    nodesInLayers[d-1][i] = list(comp)
        
            for i in range(N_layers):
                maxLength = max([len(nodesInLayers[i][j]) for j in range(len(nodesInLayers[i])) if nodesInLayers[i][j]] + [0])
                for k in range(maxLength):
                    for j in range(len(nodesInLayers[i])):
                        if len(nodesInLayers[i][j]) > k:
                            orderInLayers[i].append(nodesInLayers[i][j][k])
            del i, j, k, l, maxLength, comps, comp, sub_G, distance_dict, K, nodesInLayer
            
            # Decision making
            for Layers in range(len(orderInLayers)):
                for decision in range(len(orderInLayers[Layers])):
                    i = orderInLayers[Layers][decision]
                    al_pop = pop[i, :].copy()
                    G = Graph(pop, rc)
                    K = list(G.neighbors(i))
                    neighbor_pop = pop[K, :] if K else np.empty((0, 2))
        
                    while True:
                        phi = a * np.random.uniform(-1, 1, 2) 
                        # trap node
                        if no_move_counts[i] > trap_thresh:
                            neibor_cov, _ = Cov_Func(neighbor_pop, rs, Obstacle_Area, Covered_Area)
                            node_cov, _ = Cov_Func(np.vstack([neighbor_pop, pop[i, :]]), rs, Obstacle_Area, Covered_Area)
                            # if node's contribution is large enough   
                            if node_cov - neibor_cov > 0.008 * (1 + 0.5 * it/MaxIt):    # global trap node
                                fitness_ratio = 1
                                al_pop = pop[i, :] + phi * v * (1 - no_profit_move_counts[i]/MaxIt)**2
                                node_type = 1
                            else:                                                       # local trap node
                                fitness_ratio = 0
                                al_pop = pop[i, :] + phi * v
                                node_type = 2
                        # float node
                        elif no_profit_move_counts[i] > float_thresh:
                            if np.random.rand() >= 0.2: # to max trap node
                                k = K[np.argmax(trap_matrix[K])]
                                vt = np.abs(phi) * (pop[k, :] - pop[i, :])
                                if np.linalg.norm(vt) >= v:
                                    vt = vt * v/np.linalg.norm(vt)
                                al_pop = pop[i, :] +  vt
                                fitness_ratio = 0.993
                                node_type = 3
                            else:                       # random move
                                al_pop = pop[i, :] + phi * v
                                fitness_ratio = 1
                                node_type = 4
                        # default node
                        else:
                            fitness_ratio = 1
                            k = K[np.random.randint(len(K))] if K else i
                            vt = phi * (pop[i, :] - pop[k, :])
                            if np.linalg.norm(vt) >= v:
                                vt = vt * v/np.linalg.norm(vt)
                            al_pop = pop[i, :] + vt
                            node_type = 5
        
                        al_pop[0] = np.clip(al_pop[0], np.min(obs_row)+1, Obstacle_Area.shape[0])
                        al_pop[1] = np.clip(al_pop[1], np.min(obs_col)+1, Obstacle_Area.shape[1])
        
                        obs = np.vstack([np.column_stack([obs_row, obs_col]), pop[:i, :], pop[i+1:, :]])
                        obs_check1 = np.linalg.norm(obs - al_pop, axis=1)
                        if np.all(obs_check1 > safe_d):
                            break
        
                    if node_type == 1:
                        no_profit_move_counts[i] = trap2float
                    elif node_type == 2:
                        no_profit_move_counts[i] += trap2float
                        no_move_counts[i] = 0
        
                    neighbor_G = Graph(np.vstack([neighbor_pop, al_pop]), rc)
                    if Connectivity_graph(neighbor_G, None) == 1:
                        new_cov, Covered_Area = Cov_Func(np.vstack([neighbor_pop, al_pop]), rs, Obstacle_Area, Covered_Area)
                        old_cov, Covered_Area = Cov_Func(np.vstack([neighbor_pop, pop[i, :]]), rs, Obstacle_Area, Covered_Area)
                        if new_cov > old_cov:
                            no_move_counts[i] = 0
                            no_profit_move_counts[i] = max(no_profit_move_counts[i] - 1, 0)
                            pop[i, :] = al_pop
                        elif new_cov >= old_cov * fitness_ratio:
                            no_move_counts[i] = 0
                            no_profit_move_counts[i] += 1
                            pop[i, :] = al_pop
                        else:
                            no_move_counts[i] += 1
                    else:
                        no_move_counts[i] += 1
                # try:
            # del obs, obs_check1, k, phi, al_pop, new_cov, old_cov
            BestCostIt[it], Covered_Area = Cov_Func(pop, rs, Obstacle_Area, Covered_Area)
            popIt[it, :] = pop.flatten()
            
            print(f"{BestCostIt[it]*100:.2f}% at time step: {it}")
            # frames = plot2Ddeployment(pop, rs, rc, BestCostIt[it], it, Obstacle_Area, Covered_Area, frames)
            
        del al_pop, al_trap_matrix, d, decision, fitness_ratio, i, k, K, Layers, N_layers, neibor_cov, neighbor_G, neighbor_pop, new_cov, node_cov, node_type, nodesInLayers, obs, obs_check1
        del old_cov, orderInLayers, phi, vt, obs_col, obs_row, G
        
        folder_name = f'data/case_{case}/SOMEA'
        file_name = f'SOMEA_{Trial}.mat'
        save_mat(folder_name, file_name, popIt, BestCostIt, Obstacle_Area)
    

# %%--- xuất file GIF ---
# frames[0].save(
#     "simulation_case7.gif",
#     save_all=True,
#     append_images=frames[1:],
#     duration=80,    # ms giữa các frame (80 ms ~ 12.5 FPS)
#     loop=0
# )