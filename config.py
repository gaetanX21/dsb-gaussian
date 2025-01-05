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
        "L": data.random_L(args.cov_type, dim).tolist()
    }
    config["pdata"] = {
        "type": "gaussian",
        "dim": dim,
        "mean": args.mean_data,
        "cov_type": args.cov_type,
        "L": data.random_L(args.cov_type, dim).tolist()
    }

    # save config to YAML
    config_file = join(args.parent_dir, args.name, 'config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_file # just for simplicity with main.py
