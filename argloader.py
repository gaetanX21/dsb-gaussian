import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Trains the Diffusion Schrödinger Bridge (DSB) model.")

    parser.add_argument("--config_file", help="optional path to configuration file") # overrides all the other args
    parser.add_argument("--host", default="unknown" , help="optionally specify SSH host") # (useful for monitoring)
    
    parser.add_argument("--config_only", action="store_true" , help="the script will only generate the config file without training the model") # (useful for creating sweep configs by hand)

    group_verbosity = parser.add_mutually_exclusive_group()
    group_verbosity.add_argument("-d", "--debug", action="store_true", help="run program in debug mode")
    group_verbosity.add_argument("-q", "--quiet", action="store_true", help="run program in quiet mode")

    parser.add_argument("--mean_prior", type=float, default=0, help="mean of prior distribution")
    parser.add_argument("--mean_data", type=float, default=0, help="mean of data distribution")

    parser.add_argument("--cov_type", required=False, help="random 'spherical' or 'diagonal' or 'general' covariance matrix for both pprior and pdata")
    parser.add_argument("--cov_seed", type=int, default=42, help="seed value for RNG")
    # group_cov.add_argument("--random_spherical", action="store_true", help="Sigma=std*I_dim and Sigma'=std'*I_dim with std, std' ~ U([0,1])")
    # group_cov.add_argument("--random_diagonal", action="store_true", help="Sigma=Diag(sigma_i) and Sigma'=Diag(sigma_i') with sigma_i, sigma_' ~ U([0,1])")
    # group_cov.add_argument("--random_general", action="store_true", help="Sigma, Sigma' ~ random SPD matrix")
    
    parser.add_argument("--random_diagonal")
    parser.add_argument("--dataset", required=False, help="experiment's name (config+weights+log will be stored in experiments/name/ folder)")
    parser.add_argument("--name", required=False, help="experiment's name (config+weights+log will be stored in experiments/name/ folder)")
    parser.add_argument("--parent_dir", default="experiments", help="parent directory to store the results (will be stored in directory/name)")
    parser.add_argument("-L", type=int, default=20, help="number of IPF iterations")
    parser.add_argument("-N", type=int, default=20, help="number of steps from pdata to pprior i.e. length of bridges")
    parser.add_argument("--gamma", type=float, default=1/40, help="step size used for sampling, such that T=gamma*N")
    parser.add_argument("--n_epoch", type=int, default=10_000, help="number of epochs for each IPF iteration")
    parser.add_argument("--cache_size", type=int, default=10_000, help="size of the cache used during training")
    parser.add_argument("--cache_period", type=int, default=1_000, help="number of epochs between each cache renewal")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate used for training")
    parser.add_argument("--use_ema", action="store_true", help="whether to use EMA during training")
    parser.add_argument("--use_sgd", action="store_true", help="whether to use SGD instead of Adam (to reduce memory consumption) for training")
    parser.add_argument("--gradient_clip", action="store_true", help="whether to use gradient clipping during training")

    args = parser.parse_args()
    return args