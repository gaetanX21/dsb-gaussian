import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Trains the Diffusion Schrödinger Bridge (DSB) model.")

    parser.add_argument("--config_file", help="optional path to configuration file") # overrides all the other args
    parser.add_argument("--host", default="unknown" , help="optionally specify SSH host") # (useful for monitoring)
    
    parser.add_argument("--config_only", action="store_true" , help="the script will only generate the config file without training the model") # (useful for creating sweep configs by hand)

    group_verbosity = parser.add_mutually_exclusive_group()
    group_verbosity.add_argument("-d", "--debug", action="store_true", help="run program in debug mode")
    group_verbosity.add_argument("-q", "--quiet", action="store_true", help="run program in quiet mode")

    parser.add_argument("--dataset", required=True, help="experiment's name (config+weights+log will be stored in experiments/name/ folder)")
    parser.add_argument("--name", required=True, help="experiment's name (config+weights+log will be stored in experiments/name/ folder)")
    parser.add_argument("--parent_dir", default="experiments", help="parent directory to store the results (will be stored in directory/name)")
    parser.add_argument("-L", type=int, default=20, help="number of IPF iterations")
    parser.add_argument("-N", type=int, default=20, help="number of steps from pdata to pprior i.e. length of bridges")
    parser.add_argument("--gamma0", type=float, default=5e-4, help="step size used for sampling, such that T=gamma*N")
    parser.add_argument("--gamma_bar", type=float, default=5e-4, help="step size used for sampling, such that T=gamma*N")
    parser.add_argument("--n_epoch", type=int, default=20_000, help="number of epochs for each IPF iteration")
    parser.add_argument("--cache_size", type=int, default=10_000, help="size of the cache used during training")
    parser.add_argument("--cache_period", type=int, default=1_000, help="number of epochs between each cache renewal")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate used for training")
    parser.add_argument("--use_ema", action="store_true", help="whether to use EMA during training")
    parser.add_argument("--use_sgd", action="store_true", help="whether to use SGD instead of Adam (to reduce memory consumption) for training")
    parser.add_argument("--gradient_clip", action="store_true", help="whether to use gradient clipping during traning")

    group_pdata  = parser.add_mutually_exclusive_group(required=False)
    group_pdata.add_argument("--data_tag", help="specifies the 2d toy distribution (moon, circles) used for pdata")
    group_pdata.add_argument("--data_image", default='torch.webp', help="specifies the filename of the image used for pdata")
    parser.add_argument('--n_data_samples', type=int, default=20000, help="number of samples kept for pdata")

    group_pprior  = parser.add_mutually_exclusive_group(required=False) # default is Gaussian prior
    group_pprior.add_argument("--prior_tag", help="specifies the 2d toy distribution (moon, circles) used for pprior")
    group_pprior.add_argument("--prior_image", help="specifies the filename of the image used for pprior")
    parser.add_argument('--n_prior_samples', type=int, default=20000, help="number of samples kept for pprior")

    args = parser.parse_args()
    return args