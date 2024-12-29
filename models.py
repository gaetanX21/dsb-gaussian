import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torchvision.ops import MLP
import math
from data import DataSampler
import logging
import time
import yaml
import data
from os.path import join
from ema import EMA


class CachedDSB(nn.Module):
    """
    Implements Cached Diffusion Schrödinger Bridge as proposed by De Bortoli et al. (cf. https://arxiv.org/abs/2106.01357)
    """
    def __init__(self, dataset: str, name: str, host: str, parent_dir: str, device: torch.device, L: int, N: int, n_epoch: int, cache_size: int, cache_period: int,  lr: float, batch_size: int, gamma: float, model: nn.Module, pprior: DataSampler, pdata: DataSampler, logger: logging.Logger, use_ema: bool=False, use_sgd: bool=False):
        super().__init__()

        # set logger
        self.logger = logger
        self.logger.info("-"*42)
        self.logger.info("Initializing CachedDSB with the following parameters:")
        self.logger.info(f"name={name}, parent_dir={parent_dir}, host={host}, device={device}, L={L}, N={N}, n_epoch={n_epoch} cache_size={cache_size}, cache_period={cache_period}, lr={lr}, batch_size={batch_size}, gamma={gamma}")
        n_model_params = sum(p.numel() for p in model(data_dim=int(dataset[:-1]), max_len=1).parameters())
        self.logger.info(f"model={model.__name__}, n_model_params={n_model_params}")
        self.logger.info(f"prior={pprior}, data={pdata}")
        self.logger.info("-"*42)

        # set parameters
        self.dataset = dataset
        self.name = name
        self.parent_dir = parent_dir
        self.status_file = join(parent_dir, name, "status.yaml")
        self.host = host
        self.device = device
        self.L = L
        self.N = N
        self.n_epoch = n_epoch
        self.cache_size = cache_size
        self.cache_period = cache_period
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.pprior = pprior
        self.pdata = pdata
        self.model = model
        self.data_dim = int(dataset[:-1])
        self.use_ema = use_ema
        self.optim = SGD if use_sgd else Adam

        # instantiate f and b
        self.type_to_net = {"alpha": "f", "beta": "b"} # just for clarity
        self.nets = {
            "f": model(self.data_dim, init_zero=True, max_len=N+1).to(device), # forward network, parametrized by alpha
            "b": model(self.data_dim, max_len=N+1).to(device) # backward network, parametrized by beta
        }
        if self.use_ema:
            self.ema = EMA(self.nets["f"])
        self.save_weights('alpha', 0) # initialize alpha_0 as 0 --> useless but ok
        self.optimizer = None # optimizer for training (used later)
        self.cache = None # cache for training (used later)

    @classmethod
    def from_config(cls, config_file, logger, host: str="unknown"):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # build pdata & pprior
        pdata = data.config_to_p(config['pdata'])
        pprior = data.config_to_p(config['pprior'])
        # create CachedDSB instance
        model_config = config['model']
        instance = cls(
            dataset = config["general"]["dataset"],
            device = torch.device(config["general"]["device"]),
            name = config["general"]["name"],
            host = host,
            parent_dir = config["general"]["parent_dir"],
            L = model_config["L"],
            N = model_config["N"],
            n_epoch = model_config["n_epoch"],
            gamma = model_config["gamma"],
            cache_size = model_config["cache_size"],
            cache_period = model_config["cache_period"],
            lr = model_config["lr"],
            batch_size = model_config["batch_size"],
            model = PositionalMLP,
            pprior = pprior,
            pdata = pdata,
            logger = logger,
            use_ema = config["model"]["use_ema"],
            use_sgd = config["model"]["use_sgd"],
        )
        return instance

    def track_memory(func):
        """
        Logs Memory (Allocated & Reserved) before and after calling a function.
        """
        def wrapper(self, *args, **kwargs):
            mem_alloc_before = int(torch.cuda.memory_allocated() / 1024**2) # MiB
            mem_res_before = int(torch.cuda.memory_reserved() / 1024**2) # MiB
            output = func(self, *args, **kwargs)
            mem_alloc_after = int(torch.cuda.memory_allocated() / 1024**2) # MiB
            mem_res_before = int(torch.cuda.memory_reserved() / 1024**2) # MiB
            self.logger.debug(f'[{func.__name__}] Mem. Alloc. {mem_alloc_before}->{mem_alloc_after} MiB | Mem. Res. {mem_res_before}->{mem_alloc_after} MiB.')
            return output
        return wrapper

    def load_model(self, type: str, n: int, ema: bool=False):
        """
        sets self.b = model(beta_n) if type=beta or self.f = model(alpha_n) if type=alpha        
        """
        net = self.type_to_net[type] # "f" or "b"
        self.logger.info(f"Loading network {net} with weights {type}_n")
        filepath = join(self.parent_dir, self.name, "weights"+"_EMA"*ema, f"{type}_{n}.pt")
        weights = torch.load(filepath, weights_only=True)
        self.nets[net].load_state_dict(weights)

    def save_weights(self, type: str, n: int):
        """
        if type=beta saves beta_n=self.b.state_dict() as .pt file
        if type=alpha saves alpha_n=self.f.state_dict() as .pt file
        """
        net = self.type_to_net[type]
        self.logger.debug(f"Saving {type}_{n}")
        # save weights
        weights = self.nets[net].state_dict()
        filepath = join(self.parent_dir, self.name, "weights", f"{type}_{n}.pt")
        torch.save(weights, filepath)
        # save EMA weights too, if any
        if self.use_ema:
            weights_ema = self.ema.state_dict()
            filepath_ema = join(self.parent_dir, self.name, "weights_EMA", f"{type}_{n}.pt")
            torch.save(weights_ema, filepath_ema)
        
    # @track_memory
    def set_optim(self, type: str) -> torch.optim:
        """
        Sets a new adam optimizer to train b (type=beta) or f (type=alpha)
        """
        net = self.type_to_net[type]
        self.logger.debug(f"Getting optimizer {self.optim.__name__} for {net}")
        self.optimizer = self.optim(self.nets[net].parameters(), lr=self.lr)

    def set_ema(self, type: str) -> EMA:
        """
        Sets a new EMA for the training of b (type=beta) or f (type=alpha)
        """
        net = self.type_to_net[type]
        self.logger.debug(f"Creating EMA for {net}")
        self.ema = EMA(self.nets[net])

    # @track_memory
    def init_cache(self):
        self.cache = torch.zeros((self.N+1, self.cache_size, self.data_dim), device=self.device)
        self.Z = torch.zeros((self.N, self.cache_size, self.data_dim), device=self.device)
        self.ones = torch.ones((self.cache_size,), dtype=torch.int, device=self.device)

    def train_model(self):
        """
        Trains the DSB network for L iterations.
        """
        self.t0 = time.time() # so that update_status can access t0 to display time elapsed
        self.logger.info(f'Training started for {self.L} IPF iterations')
        self.logger.debug(f'IPF iteration n=0')
        self.init_cache()
        self.ipf_step('beta', 0) # for the first iteration we want alpha_0=0 so we do it apart
        self.nets["f"] = self.model(self.data_dim, max_len=self.N+1).to(self.device) # now alpha_1 can be initialized to noise (we then train it!)
        # we need to do this separation otherwise a network with ReLU activations and weights=0 have gradients=0 and thus doesn't train!
        for n in range(1, self.L):
            self.logger.info(f'IPF iteration n={n}')
            self.ipf_step('alpha', n)
            self.ipf_step('beta', n)
        self.update_status("alpha", self.L) # indicate we have terminated training
        dt = int(time.time()-self.t0) // 60
        self.logger.info(f'Training finished in {dt//60}h{dt%60:02d}min')

    def update_status(self, type: str, n: int):
        status = {
            "host": self.host,
            "ipf_step": f"{type} {n}/{self.L}" if n<self.L else "-",
            "time_elapsed": int(time.time() - self.t0),
            "status": "training" if n<self.L else "completed"
        }
        with open(self.status_file, "w") as f:
            yaml.safe_dump(status, f)

    def ipf_step(self, type: str, n: int):
        """
        Performs 1 IPF step i.e. trains (half) the DSB network for level n i.e. solves a half-bridge.
        """
        self.update_status(type, n)
        self.logger.debug(f"IPF step {type} {n}/{self.L}")
        self.set_optim(type) # warm-start since use previous weights BUT we DO reset the optim
        if self.use_ema:
            self.set_ema(type) # EMA for network
        for step in range(self.n_epoch):
            if step % self.cache_period == 0:
                self.refresh_cache(type)
            k, Xk, Xkp = self.sample_cache()
            self.gradient_step(type, k, Xk, Xkp)
        self.save_weights(type, n)

    # @track_memory
    def refresh_cache(self, type: str):
        """
        Refreshes the cache i.e. creates cache_size paths.
        - type beta = we generate forward paths using f
        - type alpha = we generate reverse paths using b
        """
        self.logger.debug(f'Refreshing cache of type {type}')
        X, Z, ones = self.cache, self.Z, self.ones # for naming simplicity
        Z.normal_() # refresh Z with N(0,1) values!
        self.generate_path_(type, X, Z, ones)
    
    def generate_path(self, type: str, M: int, remove_last_noise: bool=False):
        """
        Creates M paths AFTER TRAINING.
        - type beta = we generate forward paths using f
        - type alpha = we generate reverse paths using b
        """
        X = torch.zeros((self.N+1, M, self.data_dim), device=self.device)
        Z = torch.randn((self.N, M, self.data_dim), device=self.device)
        if remove_last_noise:
            if type == "beta":
                Z[-1].zero_()
            else:
                Z[0].zero_()
        ones = torch.ones((M,), dtype=torch.int, device=self.device)
        self.generate_path_(type, X, Z, ones)
        return X
        
    def generate_path_(self, type: str, X: torch.tensor, Z: torch.tensor, ones: torch.IntTensor):
        """
        Given empty tensor X and noise Z and "ones", creates the corresponding path by modifying the tensors in-place.
        """
        M = X.shape[1] # number of paths generated
        f, b, gamma = self.nets["f"], self.nets["b"], self.gamma # for conciseness
        if type == "beta":
            X[0].copy_(self.pdata.sample(M))
            with torch.no_grad():
                for k in range(self.N):
                    X[k+1].copy_(X[k] + gamma*f(X[k], k*ones) + math.sqrt(2*gamma) * Z[k])
        else:
            X[-1].copy_(self.pprior.sample(M))
            with torch.no_grad():
                for k in range(self.N, 0, -1):
                    X[k-1].copy_(X[k] + gamma*b(X[k], k*ones) + math.sqrt(2*gamma) * Z[k-1])

    def gradient_step(self, type: str, k: torch.IntTensor, Xk: torch.Tensor, Xkp: torch.Tensor):
        """
        Performs a gradient step on the model for type (alpha or beta) on the selected points X_{k(j)}^j using the optimizer and batch_size.
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(type, k, Xk, Xkp)
        loss.backward()
        self.optimizer.step() # weights update --> frees a lot of Memory Allocated since the gradients can be thrown away
        if self.use_ema:
            self.ema.update(self.nets[self.type_to_net[type]]) # weights_EMA update too!

    def compute_loss(self, type: str, k: torch.Tensor, Xk: torch.Tensor, Xkp: torch.Tensor) -> float:
        """
        Computes the loss for the model for type (alpha or beta) using the cache of size cache_size.
        """
        b, f = self.nets["b"], self.nets["f"] # for clarity
        if type == 'beta':
            pred = b(Xkp, k+1)
            with torch.no_grad():
                actual = (Xk-Xkp)/self.gamma + (f(Xk, k) - f(Xkp, k))
        else:
            pred = f(Xk, k)
            with torch.no_grad():
                actual = (Xkp-Xk)/self.gamma + (b(Xkp, k+1) - b(Xk, k+1))
        loss = F.mse_loss(pred, actual)
        return loss

    def sample_cache(self) -> tuple[torch.Tensor]:
        """
        Samples a batch (k,X_k,Xkp) of shape batch_size, (batch_size,N+1,d), (batch_size,N+1,d) from the cache
        """
        k = torch.randint(0, self.N, (self.batch_size,)) # "batch_size" samples from U({0,...,N-1})
        j = torch.randint(0, self.cache_size, (self.batch_size,)) # "batch_size" sampels from U({0,...,cache_size-1})
        Xk = self.cache[k, j, :] # all the X_{k(j)}^j
        Xkp = self.cache[k+1, j, :] # all the X_{k(j)+1}^j
        return k, Xk, Xkp

    
    # def full_generation(self, M: int, remove_last_noise: bool=True) -> torch.Tensor:
    #     X_init = self.pprior.sample(M) # initial noise
    #     X = torch.zeros((2*self.L, self.N+1, *X_init.shape))
    #     for n in range(self.L):
    #         self.load_model("alpha", n)
    #         X_forward = self.generate_path("beta", X_init=X[2*n-1,0], remove_last_noise=remove_last_noise) if n>0 else self.generate_path("beta", X_init=X_init, remove_last_noise=remove_last_noise)
    #         X[2*n] = X_forward
    #         self.load_model("beta", n)
    #         X_backward = self.generate_path("alpha", X_init=X[2*n][-1], remove_last_noise=remove_last_noise)
    #         X[2*n+1] = X_backward
    #     return X


def build_pe(d_model: int, max_len: int) -> torch.Tensor:
    """
    Builds positional encodings PositionalMLP2d.
    """
    pe = torch.zeros(max_len, d_model) # positional encodings
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PositionalMLP(nn.Module):
    """
    Positional MLP for 2d input data, following the architecture proposed by De Bortoli et al. (cf. https://arxiv.org/abs/2106.01357)
    """
    def __init__(self, data_dim: int, max_len: int, d_embed: int=64, init_zero: bool=False, logger: logging.Logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.logger.debug(f'PositionalMLP in dimension {data_dim} created with init_zero={init_zero} and d_model={d_embed} and max_len={max_len}')
        pe = build_pe(d_model=d_embed, max_len=max_len)
        self.register_buffer('pe', pe) # to make sure it's non-trainable
        self.mlp_1a = MLP(in_channels=data_dim, hidden_channels=[16, 32])
        self.mlp_1b = MLP(in_channels=d_embed, hidden_channels=[16, 32])
        self.mlp_2 = MLP(in_channels=64, hidden_channels=[128, 128, data_dim])
        if init_zero:
            for param in self.parameters():
                param.data.zero_()

    def forward(self, x: torch.Tensor, k: torch.IntTensor) -> torch.Tensor:
        with torch.autograd.set_detect_anomaly(True):
            hx = self.mlp_1a(x)
            hk = self.mlp_1b(self.pe[k])
            h = torch.cat([hx,hk], dim=-1)
            out = self.mlp_2(h)
            return out
