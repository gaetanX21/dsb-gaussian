import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
import models
import os
from os.path import join
import yaml
from scipy.linalg import sqrtm
from torch.linalg import matrix_norm
import subprocess


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

#################################################################

def error(estimator: torch.Tensor, reference: torch.Tensor, rel: bool) -> float:
    if rel:
        err = matrix_norm(estimator-reference) / matrix_norm(reference)
    else:
        err = matrix_norm(estimator-reference)
    return err

def assess_performance(dsb: models.CachedDSB, L: int, M: int, direction: str, rel: bool) -> tuple:
    Sigma, Sigma_prime = dsb.pdata.Sigma, dsb.pprior.Sigma
    sigma2 = 2 * dsb.N * dsb.gamma
    C = get_C(Sigma, Sigma_prime, sigma2)
    Sigma_error, Sigma_prime_error, C_error = torch.zeros(L), torch.zeros(L), torch.zeros(L)
    for n in range(0, L):
        Sigma_emp, Sigma_prime_emp, C_emp = empirical_cov(dsb, n, M, direction)
        Sigma_error[n] = error(Sigma_emp, Sigma, rel)
        Sigma_prime_error[n] = error(Sigma_prime_emp, Sigma_prime, rel)
        C_error[n] = error(C_emp, C, rel)
    return Sigma_error, Sigma_prime_error, C_error

def empirical_cov(dsb: models.CachedDSB, n: int, M: int, direction: str) -> tuple:
    if direction == "reverse":
        dsb.load_model('beta', n) # load beta_n to generate reverse paths
        X = dsb.generate_path('alpha', M=M).cpu() # sample M bridges
    elif direction == "forward":
        dsb.load_model('alpha', n) # load alpha_n to generate forward paths
        X = dsb.generate_path('beta', M=M).cpu() # sample M bridges     
    else:
        raise ValueError(f"Unknown direction: {direction}")   
    X0, XN = X[0], X[-1]
    Sigma_emp = (X0.T @ X0) / M # empirical covariance of X0
    Sigma_prime_emp = (XN.T @ XN) / M # empirical covariance of XN
    C_emp = (XN.T @ X0) / M # cross-correlation
    return Sigma_emp, Sigma_prime_emp, C_emp

def get_C(Sigma: torch.Tensor, Sigma_prime: torch.Tensor, sigma2: float) -> torch.Tensor:
    """
    Computes C according to the closed-form formula for Gaussian Schrödinger Bridge.
    """
    Sigma, Sigma_prime = Sigma.numpy(), Sigma_prime.numpy() # for compatibility with sqrtm()
    Sigma_sqrt = sqrtm(Sigma)
    Sigma_sqrt_inv = np.linalg.inv(Sigma_sqrt)
    I = np.eye(Sigma.shape[0])
    sigma4 = sigma2**2
    D = sqrtm(4 * Sigma_sqrt @ Sigma_prime @ Sigma_sqrt + sigma4 * I)
    C = 0.5 * (Sigma_sqrt @ D @ Sigma_sqrt_inv - sigma2 * I)
    return torch.tensor(C, dtype=torch.float32)

def plot_perf(parent_dir: str, exp: str, L: int=20, direction: str="reverse", M: int=25_000):
    cov_types = ["spherical", "diagonal", "general"]
    x = range(L)
    fig, axs = plt.subplots(ncols=2, figsize=(10,5))
    for cov_type in cov_types:
        name = exp+"_"+cov_type[:3]
        config_file = join(parent_dir, name, "config.yaml")
        dsb = models.CachedDSB.from_config(config_file, logger=None)
        Sigma_error, Sigma_prime_error, C_error = assess_performance(dsb, L, M=M, direction=direction)
        axs[0].plot(x, Sigma_error, label=cov_type)
        axs[1].plot(x, C_error, label=cov_type)

    symbols = ["\Sigma", "C"]
    for i, ax in enumerate(axs):
        s = symbols[i]
        ax.set_title(r"Relative error of $\hat{" + s + r"}$ over DSB iterations")
        ax.set_xlabel("DSB iteration $n$")
        ax.set_ylabel(r"$\frac{||\hat{" + s + r"}-" + s + r"||^2_F}{||" + s + r"||^2_F}$")
        ax.legend()

    plt.tight_layout()
    plt.show()

def create_sweep(sweep_dir: str, N: int, dim: int, cov_type: str):
    for i in range(N):
        name = f"{dim}d_{i}_{cov_type}"
        cmd = f"python main.py --parent_dir {sweep_dir} --dataset {dim}d --name {name} --cov_type {cov_type} --cov_seed {i} --config_only"
        os.system(cmd)
    print(f"Sweep {sweep_dir} created successfully.")

def assess_performance_sweep(sweep_dir: str, L: int, cov_type: str, M: int=250_000, direction: str="reverse", rel: bool=False) -> tuple:
    exps = [exp for exp in os.listdir(sweep_dir) if os.path.isdir(join(sweep_dir, exp))]
    print(f"{len(exps)} folders in {sweep_dir}")
    exps = [exp for exp in exps if exp.endswith(cov_type)]
    print(f"{len(exps)} folders after filtering on cov_type={cov_type}")
    exps = [exp for exp in exps if os.path.exists(join(sweep_dir, exp, "weights", f"beta_{L-1}.pt"))]
    n_exp = len(exps)
    print(f"{n_exp} exps after filtering on existence of beta_{L-1}.pt")
    Sigma_error, Sigma_prime_error, C_error = torch.zeros((n_exp,L)), torch.zeros((n_exp,L)), torch.zeros((n_exp,L))
    for i, exp in enumerate(exps):
        config_file = join(sweep_dir, exp, "config.yaml")
        dsb = models.CachedDSB.from_config(config_file, None)
        Sigma_error[i], Sigma_prime_error[i], C_error[i] = assess_performance(dsb, L, M, direction, rel)
    return Sigma_error, Sigma_prime_error, C_error


def plot_error(error: torch.Tensor, rank: bool=False, normalize: bool=False, ylabel: str=None):
    """
    Plots the error over DSB iterations.
    error of shape (n_exp,L)
    """
    error, mean, std = clean_error(error, rank, normalize)
    x = range(len(mean))

    plt.figure(dpi=150)
    plt.errorbar(x, mean, std, fmt='o-', capsize=6)
    plt.plot(error, alpha=0.25)

    plt.xlabel("DSB iteration $n$")
    plt.xticks(x[::5])
    if ylabel is None:
        ylabel = f"Error (rank={rank}, normalize={normalize})"
    plt.ylabel(ylabel)
    plt.title("Mean Error over DSB iterations")
    plt.show()


def clean_error(error: torch.Tensor, rank: bool=False, normalize: bool=False):
    """
    Computes the error over DSB iterations.
    error of shape (n_exp,L)
    """
    error = error.T.clone() # to make sure we don't modify the original tensor
    if normalize:
        error = error / error[0]
    if rank:
        error = torch.argsort(error, dim=0).float()
    mean, std = error.mean(dim=1), error.std(dim=1)
    return error, mean, std


def save_res(cov_type: str, L: int=20, M: int=250_000, rank: bool=False, normalize: bool=False):
    dims = [1,5,50]
    res = {}
    colors = ["red", "green", "blue"]
    symbols = [r"\Sigma", r"\Sigma'", "C"]
    x = list(range(L))
    print('#'*42 + '\n' + f'Dealing with cov_type={cov_type}')
    for i, d in enumerate(dims):
        sweep_dir = f"sweeps/g{d}_{cov_type[:3]}"
        errors = assess_performance_sweep(sweep_dir, L, cov_type, M)
        res[d] = dict(zip(symbols,errors))
    
    for i, symbol in enumerate(symbols):
        plt.figure(dpi=150)
        for j, d in enumerate(dims):
            error, mean, std = clean_error(res[d][symbol], rank, normalize)
            plt.errorbar(x, mean, std, fmt="o-", color=colors[j], label=f"d={d}", capsize=6)
        plt.xlabel("DSB iteration $n$")
        plt.xticks(x[::5])
        plt.ylabel(r"$||\hat{" + symbol + r"}-" + symbol + r"||^2_F$")
        plt.legend()
        plt.title(f"Error over DSB iteration ({cov_type})")
        s = symbol.strip("\\")
        fname = f"results/gaussian/{cov_type}/{s}_M={M}"
        if normalize:
            fname += "_normalize"
        if rank:
            fname += "_rank"
        fname += ".png"
        plt.savefig(fname)
        print(f"Saved {fname}")