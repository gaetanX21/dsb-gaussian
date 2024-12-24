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
    def __init__(self, dataset: str, name: str, host: str, parent_dir: str, device: torch.device, L: int, N: int, n_epoch: int, cache_size: int, cache_period: int,  lr: float, batch_size: int, gamma0: float, gamma_bar: float, model: nn.Module, data_shape: tuple[int], pprior: DataSampler, pdata: DataSampler, logger: logging.Logger, use_ema: bool=False, use_sgd: bool=False):
        super().__init__()

        # set logger
        self.logger = logger
        self.logger.info("-"*42)
        self.logger.info("Initializing CachedDSB with the following parameters:")
        self.logger.info(f"name={name}, parent_dir={parent_dir}, host={host}, device={device}, L={L}, N={N}, n_epoch={n_epoch} cache_size={cache_size}, cache_period={cache_period}, lr={lr}, batch_size={batch_size}, gamma0={gamma0}, gamma_bar={gamma_bar}")
        n_model_params = sum(p.numel() for p in model(max_len=1).parameters())
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
        self.build_gamma(gamma0, gamma_bar)
        self.pprior = pprior
        self.pdata = pdata
        self.model = model
        self.data_shape = data_shape
        self.use_ema = use_ema
        self.optim = SGD if use_sgd else Adam

        # instantiate f and b
        self.type_to_net = {"alpha": "f", "beta": "b"} # just for clarity
        self.memory_summary("Before initializing f & b")
        self.nets = {
            "f": model(init_zero=True, max_len=N+1).to(device), # forward network, parametrized by alpha
            "b": model(max_len=N+1).to(device) # backward network, parametrized by beta
        }
        self.memory_summary("After initializing f & b")
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
        if config["pprior"]["type"] == "gaussian":
            pprior = data.GaussianSampler(mean=pdata.mean, std=pdata.std) # following paper's recommendations
        else:
            pprior = data.config_to_p(config["pprior"])

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
            gamma0 = model_config["gamma0"],
            gamma_bar = model_config["gamma_bar"],
            cache_size = model_config["cache_size"],
            cache_period = model_config["cache_period"],
            lr = model_config["lr"],
            batch_size = model_config["batch_size"],
            model = PositionalMLP2d if config["general"]["dataset"] == "2d" else PositionalUNet,
            data_shape = (2,) if config["general"]["dataset"] == "2d" else (1,28,28),
            pprior = pprior,
            pdata = pdata,
            logger = logger,
            use_ema = config["model"]["use_ema"],
            use_sgd = config["model"]["use_sgd"],
        )
        return instance

    def memory_summary(self, title: str):
        self.logger.debug(f"[{title}] Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.0f} MiB | Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.0f} MiB")

    def build_gamma(self, gamma0: float, gamma_bar: float):
        if self.dataset == '2d':
            self.gamma = gamma0 * torch.ones((self.N+1,))
        else:
            k = torch.arange(0, self.N//2) # 0 to N//2 excluded
            gamma_half = gamma0 + 2*k/self.N  * (gamma_bar - gamma0)
            self.gamma = torch.cat([gamma_half, torch.Tensor([gamma_bar]), torch.flip(gamma_half, dims=[0])])
        self.gamma = self.gamma.to(self.device)
        self.logger.info(f"Using gamma {self.gamma} of length {len(self.gamma)} (should be {self.N+1})")

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
        
    def set_optim(self, type: str) -> torch.optim:
        """
        Sets a new adam optimizer to train b (type=beta) or f (type=alpha)
        """
        net = self.type_to_net[type]
        self.logger.debug(f"Getting optimizer {self.optim.__name__} for {net}")
        self.memory_summary("Before loading optimizer")
        self.optimizer = self.optim(self.nets[net].parameters(), lr=self.lr)
        self.memory_summary("After loading optimizer")

    def set_ema(self, type: str) -> EMA:
        """
        Sets a new EMA for the training of b (type=beta) or f (type=alpha)
        """
        net = self.type_to_net[type]
        self.logger.debug(f"Creating EMA for {net}")
        self.ema = EMA(self.nets[net])

    def init_cache(self):
        self.cache = torch.zeros((self.N+1, self.cache_size, *self.data_shape), device=self.device)
        self.Z = torch.zeros((self.N, self.cache_size, *self.data_shape), device=self.device)
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
        self.nets["f"] = self.model(max_len=self.N+1).to(self.device) # now alpha_1 can be initialized to noise (we then train it!)
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
            self.memory_summary(title=f"{type}_{n} step {step}/{self.n_epoch}")
            if step % self.cache_period == 0:
                self.refresh_cache(type)
            k, Xk, Xkp = self.sample_cache()
            self.gradient_step(type, k, Xk, Xkp)
        self.save_weights(type, n)

    def refresh_cache(self, type: str):
        self.logger.debug(f'Refreshing cache of type {type}')
        self.memory_summary("Before loading cache")
        # self.cache = None
        # torch.cuda.empty_cache()
        self.generate_path(type, M=self.cache_size, cache_mode=True) # refresh cache
        self.memory_summary("After loading cache")

    def gradient_step(self, type: str, k: torch.IntTensor, Xk: torch.Tensor, Xkp: torch.Tensor):
        """
        Performs a gradient step on the model for type (alpha or beta) on the selected points X_{k(j)}^j using the optimizer and batch_size.
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(type, k, Xk, Xkp)
        loss.backward()
        self.optimizer.step() # weights update
        if self.use_ema:
            self.ema.update(self.nets[self.type_to_net[type]]) # weights_EMA update too!

    def compute_loss(self, type: str, k: torch.Tensor, Xk: torch.Tensor, Xkp: torch.Tensor) -> float:
        """
        Computes the loss for the model for type (alpha or beta) using the cache of size cache_size.
        """
        b, f = self.nets["b"], self.nets["f"] # for clarity
        if type == 'beta':
            pred = b(Xkp,k+1)
            with torch.no_grad():
                actual = (Xk-Xkp)/self.gamma[k].view(-1,1) + (f(Xk,k) - f(Xkp,k)) if self.dataset == "2d" else Xkp + (f(Xk,k)-f(Xkp,k))
        else:
            pred = f(Xk,k)
            with torch.no_grad():
                actual = (Xkp-Xk)/self.gamma[k+1].view(-1,1) + (b(Xkp,k+1) - b(Xk,k+1)) if self.dataset == "2d" else Xk + (f(Xkp,k+1)-f(Xk,k+1))
        loss = F.mse_loss(pred, actual)
        return loss

    def sample_cache(self) -> tuple[torch.Tensor]:
        """
        Samples a batch (k,X_k,Xkp) of shape batch_size, (batch_size,N+1,d), (batch_size,N+1,d) from the cache
        """
        k = torch.randint(0, self.N, (self.batch_size,)) # "batch_size" samples from U({0,...,N-1})
        j = torch.randint(0, self.cache_size, (self.batch_size,)) # "batch_size" sampels from U({0,...,cache_size-1})
        Xk = self.cache[k,j,:] # all the X_{k(j)}^j
        Xkp = self.cache[k+1,j,:] # all the X_{k(j)+1}^j
        return k, Xk, Xkp
    
    def generate_path(self, type: str, M: int=None, X_init: torch.Tensor=None, remove_last_noise: bool=False, cache_mode: bool=True) -> torch.Tensor:
        """
        Generates M forward/reverse paths based on type.
        - beta generates forward paths (using f!)
        - alpha generates reverse paths (using b!)
        """
        if ((X_init is None) and (M is None)) or ((X_init is not None) and (M is not None)):
            raise ValueError("You must specify exactly one of M (number of points) or X_init (initial positions).")
        if X_init is None:
            # X_init = self.pprior.sample(M) if type=="alpha" else self.pdata.sample(M)
            X_init = self.pprior.sample(M) if type=="alpha" else self.pdata.sample(M)
        X = self.reverse_path(X_init, remove_last_noise, cache_mode) if type=="alpha" else self.forward_path(X_init, remove_last_noise, cache_mode)
        return X

    def forward_path(self, X0: torch.Tensor, remove_last_noise: bool=False, cache_mode: bool=True) -> torch.Tensor:
        """
        Uses current self.f !
        Takes tensor X0 of dim (M,d) of initial positions and samples the corresponding paths P^i starting from X0^i such that P is of shape (N+1,M,d)
        """
        print(f'cache mode= {cache_mode}')
        X = self.cache if cache_mode else torch.zeros(self.N+1, *X0.shape, device=self.device) # shape (N+1,M,d)
        M = X0.shape[0]
        X[0] = X0
        if cache_mode: self.Z.normal_() # in-place!
        Z = self.Z if cache_mode else torch.randn(self.N, *X0.shape, device=self.device) # shape (N,M,d)
        if remove_last_noise:
            Z[-1].zero_()
        ones = self.ones if cache_mode else torch.ones(M, dtype=torch.int, device=self.device)
        if self.dataset == '2d':
            with torch.no_grad():
                for k in range(self.N):
                    X[k+1] = X[k] + self.gamma[k]*self.nets["f"](X[k], k*ones) + math.sqrt(2*self.gamma[k]) * Z[k] # drift-matching for 2d dataset (because the underlying model does not have residual structure)
        else:
            with torch.no_grad():
                for k in range(self.N):
                    X[k+1] = self.nets["f"](X[k], k*ones) + math.sqrt(2*self.gamma[k]) * Z[k] # mean-matching for image datasets (the underlying  U-Net model does have a residual structure)
        self.memory_summary("after loop")
        return X
    
    def reverse_path(self, XN: torch.Tensor, remove_last_noise: bool=False, cache_mode: bool=True) -> torch.Tensor:
        """
        Uses current self.b !
        Takes tensor XN of dim (M,d) of final positions and samples the corresponding paths P^i starting from XN^i such that P is of shape (N+1,M,d)
        """
        X = self.cache if cache_mode else torch.zeros(self.N+1, *XN.shape, device=self.device) # shape (N+1,M,d)
        M = XN.shape[0]
        X[-1] = XN
        if cache_mode: self.Z.normal_() # in-place!
        Z = self.Z if cache_mode else torch.randn(self.N, *XN.shape, device=self.device) # shape (N,M,d)
        if remove_last_noise:
            Z[0].zero_()
        ones = self.ones if cache_mode else torch.ones(M, dtype=torch.int, device=self.device)
        if self.dataset == "2d":
            with torch.no_grad():
                for k in range(self.N,0,-1):
                    X[k-1] = X[k] + self.gamma[k]*self.nets["b"](X[k],k*ones) + math.sqrt(2*self.gamma[k]) * Z[k-1] # drift-matching for 2d dataset
        else:
            with torch.no_grad():
                for k in range(self.N,0,-1):
                    X[k-1] = self.nets["b"](X[k],k*ones) + math.sqrt(2*self.gamma[k]) * Z[k-1] # mean-matching for image dataset
        return X
    
    def full_generation(self, M: int, remove_last_noise: bool=True) -> torch.Tensor:
        X_init = self.pprior.sample(M) # initial noise
        X = torch.zeros((2*self.L, self.N+1, *X_init.shape))
        for n in range(self.L):
            self.load_model("alpha", n)
            X_forward = self.generate_path("beta", X_init=X[2*n-1,0], remove_last_noise=remove_last_noise) if n>0 else self.generate_path("beta", X_init=X_init, remove_last_noise=remove_last_noise)
            X[2*n] = X_forward
            self.load_model("beta", n)
            X_backward = self.generate_path("alpha", X_init=X[2*n][-1], remove_last_noise=remove_last_noise)
            X[2*n+1] = X_backward
        return X


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


class PositionalMLP2d(nn.Module):
    """
    Positional MLP for 2d input data, following the architecture proposed by De Bortoli et al. (cf. https://arxiv.org/abs/2106.01357)
    """
    def __init__(self, max_len: int, d_model: int=64, init_zero: bool=False, logger: logging.Logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.logger.debug(f'PositionalMLP2d created with init_zero={init_zero} and d_model={d_model} and max_len={max_len}')
        pe = build_pe(d_model=d_model, max_len=max_len)
        self.register_buffer('pe', pe) # to make sure it's non-trainable
        self.mlp_1a = MLP(in_channels=2, hidden_channels=[16, 32])
        self.mlp_1b = MLP(in_channels=d_model, hidden_channels=[16, 32])
        self.mlp_2 = MLP(in_channels=64, hidden_channels=[128, 128, 2])
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


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = build_pe(d_model=d_model, max_len=max_len)
        self.register_buffer('pe', pe) # to make sure it's non-trainable
    
    def forward(self, x: torch.IntTensor) -> torch.Tensor:
        return self.pe[x]

#######################################""
# U-Net code (from PGM project)
class Dense(nn.Module):
  """Fully connected layer + reshape outputs to feature maps (i.e. with width and height of 1 each)."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[...,None,None] # same as [:,:,None,None]


class PositionalUNet(nn.Module):
  "Position-conditional U-Net to estimate the score network."""
  def __init__(self, n_channels: int=1, channels: list[int]=[32,64,128,256], embed_dim: int=64, max_len: int=51, init_zero: bool=False):
    """Initialize a time-dependent score-based network.
    Args:
      channels: The number of channels for feature maps for each layer.
      embed_dim: The dimensionality of the Gaussian random feature embedding.
    """
    super().__init__()
    # embed time
    pe = build_pe(d_model=embed_dim, max_len=max_len)
    self.register_buffer('pe', pe) # to make sure it's non-trainable
    self.lin_embed = nn.Linear(embed_dim, embed_dim)
    # encoder
    self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=channels[0], kernel_size=3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(num_groups=4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(num_groups=32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(num_groups=32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(num_groups=32, num_channels=channels[3])
    # decoder
    self.tconv4 = nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[2], kernel_size=3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(num_groups=32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(in_channels=channels[2]*2, out_channels=channels[1], kernel_size=3, stride=2, bias=False, output_padding=1)
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(num_groups=32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(in_channels=channels[1]*2, out_channels=channels[0], kernel_size=3, stride=2, bias=False, output_padding=1)
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(num_groups=32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(in_channels=channels[0]*2, out_channels=n_channels, kernel_size=3, stride=1)
    # activation
    self.activation = lambda x: x*torch.sigmoid(x) # swish activation function

    if init_zero:
        for param in self.parameters():
            param.data.zero_()

  def forward(self, x: torch.Tensor, t: torch.IntTensor):
    # embed time
    embedding = self.activation(self.lin_embed(self.pe[t]))
    # encoder
    h1 = self.activation( self.gnorm1( self.conv1(x)+self.dense1(embedding) ) )
    h2 = self.activation( self.gnorm2( self.conv2(h1)+self.dense2(embedding) ) )
    h3 = self.activation ( self.gnorm3( self.conv3(h2)+self.dense3(embedding) ) )
    h4 = self.activation ( self.gnorm4( self.conv4(h3)+self.dense4(embedding) ) )
    # decoder
    h5 = self.activation ( self.tgnorm4( self.tconv4(h4)+self.dense5(embedding) ) )
    h6 = self.activation ( self.tgnorm3( self.tconv3(torch.cat([h5,h3],dim=1))+self.dense6(embedding) ) )
    h7 = self.activation ( self.tgnorm2 ( self.tconv2(torch.cat([h6,h2],dim=1))+self.dense7(embedding) ) )
    h8 = self.tconv1(torch.cat([h7,h1], dim=1))
    # normalize output
    h = h8 # / self.marginal_prob_std(t)[:,None,None,None]
    return h