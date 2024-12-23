from os.path import join
import yaml

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
        "pdata": {
            "n_samples": args.n_data_samples,
        },
        "pprior": {
            "n_samples": args.n_prior_samples,
        },
        "model": {
            "L": args.L,
            "N": args.N,
            "n_epoch": args.n_epoch,
            "gamma0": args.gamma0,
            "gamma_bar": args.gamma_bar,
            "cache_size": args.cache_size,
            "cache_period": args.cache_period,
            "optimizer": "SGD" if args.use_sgd else "Adam",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "use_ema": args.use_ema
        }
    }

    # set pdata config
    if args.dataset == "mnist":
        config["pdata"]["type"] = "MNIST"
    else:
        if args.data_tag:
            config["pdata"]["type"] = "tag"
            config["pdata"]["tag"] = args.data_tag
        elif args.data_image:
            config["pdata"]["type"] = "image"
            config["pdata"]["image"] = args.data_image
        else:
            raise ValueError(f'Unknown pdata type')      

    # set pprior config
    if args.prior_tag:
        config["pprior"]["type"] = "tag"
        config["pprior"]["tag"] = args.prior_tag
    elif args.prior_image:
        config["pprior"]["type"] = "image"
        config["pprior"]["image"] = args.prior_image
    else:
        config["pprior"]["type"] = "gaussian"

    # save config to YAML
    config_file = join(args.parent_dir, args.name, 'config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_file # just for simplicity with main.py
