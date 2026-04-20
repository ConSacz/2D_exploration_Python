try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import numpy as np
from utils.genarea import genarea
from utils.graph_functions import Graph, Connectivity_graph
from utils.Cov_Func import Cov_Func
from utils.plot_functions import plot2Ddeployment
from utils.Workspace_functions import save_mat

# %%
cases = ["image0", "image1", "image2", "image3", "image4"]
for case in cases:
    
    for Trial in range(0,50):
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
        v = 5
        safe_d = 1
        
        # %% Init pop
        initpop = np.random.uniform(max(sink[0]-rc/2, 2), sink[1]+rc/2, (N, 2))
        initpop[0, :] = sink
        pop = initpop.copy()
        
        BestCostIt = np.zeros(MaxIt)
        popIt = np.zeros((MaxIt, 2*N))
        popIt[0, :] = pop.flatten()
        C = np.zeros(N)
        
        # %% Main
        for it in range(1, MaxIt):
            for i in range(1,N):
                al_pop = pop.copy()
            
                while True:
            
                    K = np.concatenate((np.arange(0, i),np.arange(i+1, N)))
                    k = np.random.choice(K)
            
                    phi = a*np.random.uniform(-1, 1, 2)*(1-C[i]/MaxIt)**5
                    vt = phi*(pop[i, :] - pop[k, :])
            
                    if np.linalg.norm(vt) >= v:
                        vt = vt * v/np.linalg.norm(vt)
            
                    al_pop[i, :] = pop[i, :] + vt
            
                    obs_idx = np.argwhere(Obstacle_Area == 0)
                    obs_x, obs_y = obs_idx[:, 0], obs_idx[:, 1]
            
                    al_pop[i, 0] = np.clip(al_pop[i, 0],np.min(obs_x)+1,Obstacle_Area.shape[0])
                    al_pop[i, 1] = np.clip(al_pop[i, 1],np.min(obs_y)+1,Obstacle_Area.shape[1])
            
                    obs = np.vstack((
                        np.column_stack((obs_x, obs_y)),
                        pop[:i, :],
                        pop[i+1:, :]
                    ))
            
                    obs_check1 = np.sqrt((obs[:, 0]-al_pop[i, 0])**2 + (obs[:, 1]-al_pop[i, 1])**2)
            
                    if not np.any(obs_check1 < safe_d):
                        break
            
                if Connectivity_graph(Graph(al_pop, rc), []) == 1:
                    new_Cov, _ = Cov_Func(al_pop, rs, Obstacle_Area, Covered_Area)
                    old_Cov, _ = Cov_Func(pop, rs, Obstacle_Area, Covered_Area)
            
                    if new_Cov >= old_Cov:
                        pop = al_pop.copy()
                    else:
                        C[i] += 1
                else:
                    C[i] += 1
        
            BestCostIt[it], Covered_Area = Cov_Func(pop, rs, Obstacle_Area, Covered_Area)
            popIt[it, :] = pop.reshape(1, 2*N)
        
            print(f"{BestCostIt[it]*100:.2f}%  at iteration:  {it+1}")
            # frames = plot2Ddeployment(pop, rs, rc, BestCostIt[it], it, Obstacle_Area, Covered_Area, frames)
        folder_name = f'data/case_{case}/SOMEA'
        file_name = f'SOMEA_{Trial}.mat'
        save_mat(folder_name, file_name, popIt, BestCostIt, Obstacle_Area)
    
# %%--- xuất file GIF ---
# frames[0].save(
#     "simulation_case6.2.gif",
#     save_all=True,
#     append_images=frames[1:],
#     duration=80,    # ms giữa các frame (80 ms ~ 12.5 FPS)
#     loop=0
# )