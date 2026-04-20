import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot2Ddeployment(pop, rs, rc, Coverage, it, Obstacle_Area, Covered_Area, frames):
    N = np.shape(pop)[0]
    
    # Plotting
    fig = plt.figure(figsize=(6, 6))
    plt.clf()
    obs_row, obs_col = np.where(Obstacle_Area == 1)
    plt.plot(obs_row, obs_col, 's', markersize=0.1, color='blue')
    # obs_row, obs_col = np.where(Obstacle_Area == 0)
    # plt.plot(obs_row, obs_col, 's', markersize=3, color='brown')
    discovered_obs_row, discovered_obs_col = np.where(Covered_Area == -1)
    plt.plot(discovered_obs_row, discovered_obs_col, 's', markersize=3, color='red',linestyle='None')
    discovered_row, discovered_col = np.where(Covered_Area == 0)
    plt.plot(discovered_row, discovered_col, 's', markersize=0.5, color='gray')

    theta = np.linspace(0, 2*np.pi, 500)
    for i in range(N):
        plt.plot(pop[i,0], pop[i,1], 'o', markersize=3, color='blue')
        plt.text(pop[i,0], pop[i,1], str(i+1), fontsize=8, color='red')
        x = pop[i,0] + rs[i] * np.cos(theta)
        y = pop[i,1] + rs[i] * np.sin(theta)
        plt.fill(x, y, color=(0.6, 1, 0.6), alpha=0.2, edgecolor='k')
    del i, x, y, theta
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = pop[i,0] - pop[j,0]
            dy = pop[i,1] - pop[j,1]
            d = np.sqrt(dx*dx + dy*dy)
    
            if d <= rc:
                plt.plot([pop[i,0], pop[j,0]],
                         [pop[i,1], pop[j,1]],
                         '-', color='gray', linewidth=0.5)
    del i, j, dx, dy, d
    
    
    plt.xlim([0, Obstacle_Area.shape[0]])
    plt.ylim([0, Obstacle_Area.shape[1]])
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.title(f"{Coverage*100:.2f}% at time step: {it}")
    
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.pause(0.0001)
    
    # --- lưu frame ---
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    image = np.asarray(renderer.buffer_rgba())
    frames.append(Image.fromarray(image[:, :, :3]))

    return frames


# %% constraint preservation draw

    # alpop = [45, 67]
    # plt.plot(alpop[1], alpop[0], 'o', markersize=3, color='blue')
    # plt.text(alpop[1], alpop[0], '4*', fontsize=8, color='red')
    # plt.annotate(
    #     '',
    #     xy=(alpop[1], alpop[0]),        # đích
    #     xytext=(pop[3,1], pop[3,0]),    # nguồn
    #     arrowprops=dict(
    #         arrowstyle='->',
    #         color='red',
    #         linewidth=0.8,
    #         linestyle='--'
    #     )
    # )
    
    # for i in range(N):
    #     dx = pop[i,1] - alpop[1]
    #     dy = pop[i,0] - alpop[0]
    #     d = np.sqrt(dx*dx + dy*dy)

    #     if d <= rc and i!=3:
    #         plt.plot([pop[i,1], alpop[1]],
    #                  [pop[i,0], alpop[0]],
    #                  '-', color='gray', linewidth=0.8)
    # del i, dx, dy, d
    
    
    # plt.title("Local Connectivity constraint preserved")