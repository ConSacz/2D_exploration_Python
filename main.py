globals().clear()
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from genarea import genarea
from Graph import Graph
from Cov_Func import Cov_Func
from Connectivity_graph import Connectivity_graph
from PIL import Image

frames = []   # lưu toàn bộ hình để ghép GIF
fig = plt.figure()

# Network parameter
Obstacle_Area = genarea()
Covered_Area = np.zeros_like(Obstacle_Area)
obs_row, obs_col = np.where(Obstacle_Area == 0)

# Node info
MaxIt = 500
a = 1
N = 50
rc = 16
rs = 8 * np.ones(N)
sink = [10, 10]
trap_thresh = 10
float_thresh = 100
v = 5
safe_d = 1

# Init pop
initpop = np.random.uniform(max(sink[0]-rc/2, 2), sink[1]+rc/2, (N, 2))
initpop[0, :] = sink
pop = initpop.copy()

no_move_counts = np.zeros(N)
no_profit_move_counts = np.zeros(N)
trap_matrix = np.zeros(N)
BestCostIt = np.zeros(MaxIt)
popIt = np.zeros((MaxIt, 2*N))
popIt[0, :] = pop.flatten()

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
                phi = a * np.random.uniform(-1, 1, 2) * (1 - no_profit_move_counts[i]/MaxIt)**2
                if no_move_counts[i] > trap_thresh:
                    new_cov, _ = Cov_Func(neighbor_pop, rs, Obstacle_Area, Covered_Area)
                    old_cov, _ = Cov_Func(np.vstack([neighbor_pop, pop[i, :]]), rs, Obstacle_Area, Covered_Area)
                    if old_cov - new_cov > 0.05 * old_cov:
                        fitness_ratio = 1
                        al_pop = pop[i, :] + phi * v
                        node_type = 1
                    else:
                        fitness_ratio = 0
                        al_pop = pop[i, :] + phi * v
                        node_type = 2
                elif no_profit_move_counts[i] > float_thresh:
                    if np.random.rand() >= 0.2:
                        k = K[np.argmax(trap_matrix[K])]
                        al_pop = pop[i, :] + np.abs(phi) * (pop[k, :] - pop[i, :]) * (v/rc)
                        fitness_ratio = 0.993
                        node_type = 3
                    else:
                        al_pop = pop[i, :] + phi * v
                        fitness_ratio = 1
                        node_type = 4
                else:
                    fitness_ratio = 1
                    k = K[np.random.randint(len(K))] if K else i
                    al_pop = pop[i, :] + phi * (pop[i, :] - pop[k, :]) * (v/rc)
                    node_type = 5

                al_pop[0] = np.clip(al_pop[0], np.min(obs_row)+1, Obstacle_Area.shape[0])
                al_pop[1] = np.clip(al_pop[1], np.min(obs_col)+1, Obstacle_Area.shape[1])

                obs = np.vstack([np.column_stack([obs_row, obs_col]), pop[:i, :], pop[i+1:, :]])
                obs_check1 = np.linalg.norm(obs - al_pop, axis=1)
                if np.all(obs_check1 > safe_d):
                    break

            if node_type == 1:
                no_profit_move_counts[i] = float_thresh / 10
            elif node_type == 2:
                no_profit_move_counts[i] += float_thresh / 10
                no_move_counts[i] = 0

            neighbor_G = Graph(np.vstack([neighbor_pop, al_pop]), rc)
            if Connectivity_graph(neighbor_G, None) == 1:
                new_cov, Covered_Area = Cov_Func(np.vstack([neighbor_pop, al_pop]), rs, Obstacle_Area, Covered_Area)
                old_cov, Covered_Area = Cov_Func(np.vstack([neighbor_pop, pop[i, :]]), rs, Obstacle_Area, Covered_Area)
                if new_cov > old_cov:
                    no_move_counts[i] = 0
                    no_profit_move_counts[i] = max(no_profit_move_counts[i] - 1, 0)
                    pop[i, :] = al_pop
                elif new_cov > old_cov * fitness_ratio:
                    no_move_counts[i] = 0
                    no_profit_move_counts[i] += 1
                    pop[i, :] = al_pop
                else:
                    no_move_counts[i] += 1
            else:
                no_move_counts[i] += 1
        # try:
        #     del obs, obs_check1, k, phi, al_pop, new_cov, old_cov
    BestCostIt[it], Covered_Area = Cov_Func(pop, rs, Obstacle_Area, Covered_Area)
    popIt[it, :] = pop.flatten()

    # Plotting
    print(f"{BestCostIt[it]*100:.2f}% at time step: {it}")
    plt.clf()
    obs_row, obs_col = np.where(Obstacle_Area == 1)
    plt.plot(obs_col, obs_row, '.', markersize=0.1, color='blue')
    obs_row, obs_col = np.where(Obstacle_Area == 0)
    plt.plot(obs_col, obs_row, '.', markersize=5, color='black')
    discovered_obs_row, discovered_obs_col = np.where(Covered_Area == -1)
    plt.plot(discovered_obs_col, discovered_obs_row, '.', markersize=5, color='red')
    #discovered_row, discovered_col = np.where(Covered_Area == 1)
    #plt.plot(discovered_col, discovered_row, '.', markersize=5, color='green')

    theta = np.linspace(0, 2*np.pi, 500)
    for i in range(N):
        plt.plot(pop[i,1], pop[i,0], 'o', markersize=3, color='blue')
        plt.text(pop[i,1], pop[i,0], str(i+1), fontsize=10, color='red')
        x = pop[i,1] + rs[i] * np.cos(theta)
        y = pop[i,0] + rs[i] * np.sin(theta)
        plt.fill(x, y, color=(0.6, 1, 0.6), alpha=0.2, edgecolor='k')
        
    del x, y, theta
    plt.xlim([0, Obstacle_Area.shape[1]])
    plt.ylim([0, Obstacle_Area.shape[0]])
    plt.title(f"{BestCostIt[it]*100:.2f}% at time step: {it}")
    plt.gca().invert_yaxis()
    plt.grid(True)
    #plt.pause(0.001)
    
    # --- lưu frame ---
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    image = np.asarray(renderer.buffer_rgba())
    frames.append(Image.fromarray(image[:, :, :3]))
    
# %%--- xuất file GIF ---
frames[0].save(
    "simulation.gif",
    save_all=True,
    append_images=frames[1:],
    duration=80,    # ms giữa các frame (80 ms ~ 12.5 FPS)
    loop=0
)