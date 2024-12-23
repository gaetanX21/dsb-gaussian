import models
import torch
import argloader
import logloader
import logging
import config
import os

def main():
    # parse arguments
    args = argloader.get_args()

    if args.config_file:
            config_file = args.config_file
            # create custom logger
            # e.g. config_file = mysweeps/sweep/exp1/config.yaml
            config_dir = os.path.dirname(args.config_file) # returns mysweeps/sweep/exp1
            exp_name = os.path.basename(config_dir) # returns exp1
            parent_folder = os.path.dirname(config_dir) # returns mysweeps/sweep
            log_file = os.path.join(config_dir, "train.log")
            logger = logging.getLogger(exp_name)
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.setLevel(logging.DEBUG)
            logger.info(f"Logger initialized for {exp_name}")
    else:
        # create dirs
        os.makedirs(os.path.join(args.parent_dir, args.name, "weights"), exist_ok=True)
        os.makedirs(os.path.join(args.parent_dir, args.name, "weights_EMA"), exist_ok=True)
        # set logging level
        if args.debug:
            verbose_level = logging.DEBUG
        elif args.quiet:
            verbose_level = logging.ERROR
        else:
            verbose_level = logging.INFO
        # create logger
        log_dir = os.path.join(args.parent_dir, args.name, "train.log")
        logger = logloader.setup_logger(__name__, verbose_level, log_dir)
        # check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # create config file
        config_file = config.save_config(args, str(device))
        if args.config_only:
             print(f"Successfully generated config for {args.name} at {config_file}")
             return
        
    # instantiate DSB
    dsb = models.CachedDSB.from_config(config_file, logger, host=args.host)
    # train DSB
    dsb.train_model()


if __name__ == "__main__":
    main()