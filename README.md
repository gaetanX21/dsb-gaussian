# Study of Diffusion Schrödinger Bridge model in Gaussian case

<img src="results/troll2torch.png" alt="DSB example in 2D" width=75% style="display: block; margin: 0 auto"/>

<br>

This repository contains the implementation of Diffusion Schrödinger Bridge as proposed by <a href="https://arxiv.org/abs/2106.01357">De Bortoli</a>. Three branches coexist:

* **main** for 2d distributions
* **image-only** for image distributions (e.g. MNIST)
* **gaussian** for Gaussian experiments


Table of contents


## Introduction


## How to run an experiment
1. Select the appropriate branch depending on your case.
2. Run ```python main.py``` with your choice of arguments to fit the DSB model.
3. Access the **experiment folder** which contains a config file (```config.yaml```), a log file (```train.log```) and the model weights in (```/weights```). If you set the flag ```--use_ema```, the EMA weights will be stored in ```/weights_EMA```.
4. Generate data using the various functions in ```utils.py``` or by calling DSB's own methods !

## Distributed Training
To distribute training across several SSH hosts:
0. [Setup] List your SSH hosts (Host IP + Host Name on each line) in a ```hosts.txt``` file, create an empty ```used_hosts.txt``` file, and create a ```secret.py``` file containing your SSH username ```USER``` and password ```PSWD``` as well as the directory where you source code is stored ```PROJECT_DIR```.
1. Create a folder ```sweep_dir``` (or any name of your choice) and and fill it with several initial experiment folder by calling ```python main.py --config_only``` with your choice of arguments for each experiment you wish to add to the sweep OR Use utils.create_sweep (branch **gaussian** only).
2. Distribute your tasks to your SSH hosts using ```python distribute.py sweep_dir```.
3. Monitor progress in real-time using ```python monitor.py sweep_dir``` in another window.

