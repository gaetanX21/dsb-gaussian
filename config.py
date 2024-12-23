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
            "gamma": args.gamma,
            "cache_size": args.cache_size,
            "cache_period": args.cache_period,
            "lr": args.lr,
            "batch_size": args.batch_size,
        }
    }

    # set pdata config
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
