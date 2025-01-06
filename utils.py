import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
import models
import os
from os.path import join
import gc


def plot_path_2d(path: np.ndarray, t: np.ndarray) -> None:
    x, y = path.T
    a, b = path[0], path[-1]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(t.min(), t.max()))
    lc.set_array(t)
    lc.set_linewidth(2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_collection(lc)
    ax.scatter(*a, color='red', marker='*', s=100, zorder=3, label="a")
    ax.scatter(*b, color='blue', marker='*', s=100, zorder=3, label="b")
    ax.autoscale()
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    xrange, yrange = xmax-xmin, ymax-ymin
    padding = 0.2
    ax.set_xlim(xmin-padding*xrange, xmax+padding*xrange)
    ax.set_ylim(ymin-padding*yrange, ymax+padding*yrange)
    plt.colorbar(lc, ax=ax, label='Time')
    plt.title("2D Path Colored by Time")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(None)
    plt.legend()
    plt.show()


def animation_sb_gaussian(Z: np.ndarray, T: float=10*1000) -> FuncAnimation:
    """
    Create an animation of a dynamic Schrödinger bridge given a set of paths Z.
    T is the animation duration. (default: 10 seconds)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Z[:,0,0], Z[:,0,1], c='red', label=r'$\mathbf{x}_0 \sim \nu_0$')
    ax.scatter(Z[:,-1,0], Z[:,-1,1], c='blue', label=r'$\mathbf{x}_N \sim \nu_1$')
    sc = ax.scatter(Z[:,0,0], Z[:,0,1], c='grey', label = r'$\mathbf{x}_k$')
    ax.set_title("Dynamic Schrödinger Bridge")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    def update(frame):
        sc.set_offsets(Z[:,frame,:])
        return sc,

    N = Z.shape[1]
    ani = FuncAnimation(fig, update, frames=N, blit=True)
    plt.close()
    return ani


def plot_bridge(X: torch.Tensor, reverse: bool=True, title: str=None, ax=None, **kwargs):
    """
    Plots the "bridge" X from p_prior to p_data if reverse=True, otherwise from p_data to p_prior.
    """
    X_init, X_end = (X[-1], X[0]) if reverse else (X[0], X[-1])
    X_init_mean = X_init.mean(dim=0)
    distance = (X_init-X_init_mean.view(1,-1)).pow(exponent=2).sum(dim=1)
    distance = 1 - distance / distance.max() # normalize to [0,1] and invert sign to have yellow at the center
    
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    ax.scatter(*X_init.T, c=distance, cmap="viridis", s=1, alpha=0)
    ax.scatter(*X_end.T, c=distance, cmap="viridis", s=1, alpha=1)
    
    if title is None:
        title = "Reverse Diffusion" if reverse else "Forward Diffusion"
    ax.set_title(title)        
    ax.axis(False)


def animation_bridge(X: torch.Tensor, reverse: bool=True, filename: str=None, interval: int=100, **kwargs):
    """
    Plots the "bridge" X from p_prior to p_data if reverse=True, otherwise from p_data to p_prior.
    """     
    X_init, X_end = (X[-1], X[0]) if reverse else (X[0], X[-1])
    X_init_mean = X_init.mean(dim=0)
    distance = (X_init-X_init_mean.view(1,-1)).pow(exponent=2).sum(dim=1)
    distance = 1 - distance / distance.max() # normalize to [0,1] and invert sign to have yellow at the center

    fig, ax = plt.subplots(**kwargs)
    ax.scatter(*X_end.T, s=0) # to make sure the ax is large enough to accomodate both X_init and X_end
    sc = ax.scatter(*X_init.T, c=distance, cmap='viridis', s=1)

    if reverse:
        def update(frame):
            sc.set_offsets(X[-(frame+1)])
            return sc,
    else:
        def update(frame):
            sc.set_offsets(X[frame])
            return sc,      

    ax.set_title('reverse diffusion' if reverse else 'forward diffusion')
    ax.axis(False)
    ani = FuncAnimation(fig, update, frames=X.shape[0], interval=interval, blit=True)
    filename = "bridge" if filename is None else filename
    ani.save(f'animations/{filename}.mp4')
    plt.close()


def animation_full(X: torch.Tensor, filename: str=None, interval: int=100, **kwargs):
    # N is the number of steps per bridge (so X is of size (L,N+1,M,d))
    L, N = X.shape[0], X.shape[1]-1
    n_frames = L*(N+1)
    X_init = X[0,0]
    X_end = X[-1,-1]
    X_init_mean = X_init.mean(dim=0)
    distance = (X_init-X_init_mean.view(1,-1)).pow(exponent=2).sum(dim=1)
    distance = 1 - distance / distance.max() # normalize to [0,1] and invert sign to have yellow at the center

    fig, ax = plt.subplots(**kwargs)
    sc = ax.scatter(*X_init.T, c=distance, cmap='viridis', s=1)

    # Add annotation for DSB iteration
    annotation = ax.text(
        0.05, 0.95, '', 
        transform=ax.transAxes, 
        ha='left', va='top', 
        fontsize=12, 
        color='red', 
        fontweight='bold'
    )

    def update(frame):
        l, n = divmod(frame, N+1)
        next_frame = X[l,n] if l%2==0 else X[l,N-n]
        sc.set_offsets(next_frame)
        annotation.set_text(f"DSB={l//2}")
        return sc,

    ax.axis(False)
    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
    filename = "full" if filename is None else filename
    ani.save(f'animations/{filename}.mp4')
    plt.close()


def plot_exp(parent_dir: str, exp: str, list_n: list[int], logger, M: int, reverse=True, ema=False, remove_last_noise=True):
    n_plots = len(list_n)
    config_file = join(parent_dir, exp, "config.yaml")
    fig, axs = plt.subplots(1, n_plots, figsize=(3*n_plots, 3))
    for i, n in enumerate(list_n):
        dsb = models.CachedDSB.from_config(config_file, logger)
        dsb.load_model('beta', n, ema=ema)
        X = dsb.generate_path('alpha', M=M, remove_last_noise=remove_last_noise).cpu()
        plot_bridge(X, reverse=reverse, ax=axs[i], title=f"n={n}")
    plt.suptitle(exp)
    plt.show()


def plot_sweep(sweep_dir: str, list_n: list[int], logger, M: int, reverse=True, ema=False, remove_last_noise=True):
    exps = [exp for exp in os.listdir(sweep_dir) if os.path.isdir(join(sweep_dir, exp))]
    n_plots = len(list_n)
    for exp in exps:
        plot_exp(sweep_dir, exp, list_n, logger, M, reverse=reverse, ema=ema, remove_last_noise=remove_last_noise)


def plot_image(path: torch.Tensor):
    n_plots = len(path)
    fig, axs = plt.subplots(1, n_plots, figsize=(3*n_plots, 3))
    for i in range(n_plots):
        axs[i].imshow(path[i][0], cmap="gray")
        axs[i].axis(False)
    plt.show()


def plot_both_directions(dsb: models.CachedDSB, n: int):
    # backward
    dsb.load_model('beta', n)
    dsb.init_cache()
    dsb.refresh_cache("alpha")
    XN = dsb.cache[-1].cpu()
    XN_mean = XN.mean(dim=0)
    distance = (XN-XN_mean.view(1,-1)).pow(exponent=2).sum(dim=1)
    distance = 1 - distance / distance.max() # normalize to [0,1] and invert sign to have yellow at the center
    # generate backward plots
    k_idx = [20,15,10,5,0]
    fig, axs = plt.subplots(nrows=2, ncols=len(k_idx), figsize=(20,10))
    for i,k in enumerate(k_idx):
        Xk = dsb.cache[k].cpu()
        ax = axs[0,i]
        ax.scatter(*Xk.T, c=distance, cmap="viridis", s=1)
        # ax.set_title(f"k={k}/20")
        ax.axis(False)

    # forward
    dsb.load_model('alpha', n)
    dsb.init_cache()
    dsb.refresh_cache("beta")
    X0 = dsb.cache[0].cpu()
    X0_mean = X0.mean(dim=0)
    distance = (X0-X0_mean.view(1,-1)).pow(exponent=2).sum(dim=1)
    distance = 1 - distance / distance.max() # normalize to [0,1] and invert sign to have yellow at the center
    # generate forward plots
    for i,k in enumerate(k_idx):
        Xk = dsb.cache[k].cpu()
        ax = axs[1,i]
        ax.scatter(*Xk.T, c=distance, cmap="viridis", s=1)
        # ax.set_title(f"k={k}/20")
        ax.axis(False)

    plt.show()