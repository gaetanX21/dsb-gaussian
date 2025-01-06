from os.path import join
import yaml
import data
import torch

def save_config(args, device):
    config = vars(args)

    config = {
        "general": {
            "device": device,
            "name": args.name,
            "parent_dir": args.parent_dir,
            "host": args.host,
            "dataset": args.dataset,
            "cov_seed": args.cov_seed
        },
        "model": {
            "L": args.L,
            "N": args.N,
            "n_epoch": args.n_epoch,
            "gamma": args.gamma,
            "cache_size": args.cache_size,
            "cache_period": args.cache_period,
            "optimizer": "SGD" if args.use_sgd else "Adam",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "use_ema": args.use_ema,
            "use_sgd": args.use_sgd,
            "gradient_clip": args.gradient_clip,
        }
    }

    # set pprior/pdata config
    dim = int(args.dataset[:-1])
    torch.random.manual_seed(args.cov_seed) # for reproducibility

    config["pprior"] = {
        "type": "gaussian",
        "dim": dim,
        "mean": args.mean_prior,
        "cov_type": args.cov_type,
    }
    config["pdata"] = {
        "type": "gaussian",
        "dim": dim,
        "mean": args.mean_data,
        "cov_type": args.cov_type,
    }

    if args.cov_type == "spherical":
        config["pprior"]["std"] = rand_1_10(1).tolist()
        config["pdata"]["std"] = rand_1_10(1).tolist()
    elif args.cov_type == "diagonal":
        config["pprior"]["D"] = rand_1_10(dim).tolist()
        config["pdata"]["D"] = rand_1_10(dim).tolist()
    else:
        config["pprior"]["L"] = random_L(args.cov_type, dim).tolist()
        config["pdata"]["L"] = random_L(args.cov_type, dim).tolist()

    # save config to YAML
    config_file = join(args.parent_dir, args.name, 'config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_file # just for simplicity with main.py


def rand_1_10(dim: int):
    """
    Returns a list of size dim filled with x ~ U([1,10]) i.i.d.
    """
    r = torch.rand((dim,))
    r = r*9 + 1 # [0,1) to [1,10) --> we prefer [1,10) for stability!
    return r


def random_L(cov_type: str, dim: int) -> torch.Tensor:
    """Sigma=LL^T (Cholesky decomposition) and L is sampled randomly based on cov_type"""
    if cov_type == "spherical":
        std = rand_1_10(1)
        L = std * torch.eye(dim)
    elif cov_type == "diagonal":
        sigma_i = rand_1_10(dim)
        L = torch.diag(sigma_i)
    elif cov_type == "general":
        Z = torch.randn((dim,dim))
        svd = torch.linalg.svd(Z)
        O = svd[0] @ svd[2] # random orthogonal matrix
        sigma_i = rand_1_10(dim)
        L = O @ torch.diag(sigma_i) # L=O@D^(1/2) such that Sigma = O@D^O.T
    else:
        raise ValueError(f'Invalid cov_type {cov_type}') 
    return L