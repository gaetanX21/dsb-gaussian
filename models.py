import torch
import torch.nn as nn
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
        self.logger.info("Initializing CachedDSB *IMAGE-ONLY* with the following parameters:")
        self.logger.info(f"name={name}, parent_dir={parent_dir}, host={host}, device={device}, L={L}, N={N}, n_epoch={n_epoch} cache_size={cache_size}, cache_period={cache_period}, lr={lr}, batch_size={batch_size}, gamma0={gamma0}, gamma_bar={gamma_bar}, data_shape={data_shape}")
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
        self.init_gamma(gamma0, gamma_bar)
        self.pprior = pprior
        self.pdata = pdata
        self.model = model
        self.data_shape = data_shape
        self.use_ema = use_ema
        self.optim = SGD if use_sgd else Adam

        # instantiate f and b
        self.init_networks()
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
            model = PositionalUNet,
            data_shape = (2,) if config["general"]["dataset"] == "2d" else (1,28,28),
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

            self.logger.debug(f'[{func.__name__}] Mem. Alloc. {mem_alloc_before} -> {mem_alloc_after} MiB | Mem. Res. {mem_res_before} -> {mem_alloc_after} MiB.')
            return output
        return wrapper

    @track_memory
    def init_networks(self):
        """
        Initialize forward network F_alpha and reverse network B_beta.
        """
        self.nets = {
            "F": self.model(init_zero=True, max_len=self.N+1).to(self.device), # forward network, parametrized by alpha
            "B": self.model(max_len=self.N+1).to(self.device) # reverse network, parametrized by beta
        }
        self.type_to_net = {
            "alpha": "F",
            "beta": "B"
        }

    @track_memory
    def init_gamma(self, gamma0: float, gamma_bar: float):
        self.gamma = torch.zeros((self.N+1), device=self.device)
        k = torch.arange(0, self.N//2) # 0 to N//2 (excluded)
        self.gamma[:self.N//2] = gamma0 + 2*k/self.N  * (gamma_bar - gamma0)
        self.gamma[self.N//2] = gamma_bar
        self.gamma[self.N//2+1:] = torch.flip(self.gamma[:self.N//2], dims=[0])

    @track_memory
    def load_model(self, type: str, n: int, ema: bool=False):
        """
        sets self.b = model(beta_n) if type=beta or self.f = model(alpha_n) if type=alpha        
        """
        net = self.type_to_net[type] # "f" or "b"
        self.logger.info(f"Loading network {net} with weights {type}_n")
        filepath = join(self.parent_dir, self.name, "weights"+"_EMA"*ema, f"{type}_{n}.pt")
        weights = torch.load(filepath, weights_only=True)
        self.nets[net].load_state_dict(weights)

    @track_memory
    def save_weights(self, type: str, n: int):
        """
        if type=beta saves beta_n=self.b.state_dict() as .pt file
        if type=alpha saves alpha_n=self.f.state_dict() as .pt file
        """
        self.logger.debug(f"Saving {type}_{n}")
        # save weights
        net = self.type_to_net[type]
        weights = self.nets[net].state_dict()
        filepath = join(self.parent_dir, self.name, "weights", f"{type}_{n}.pt")
        torch.save(weights, filepath)
        # save EMA weights too, if any
        if self.use_ema:
            weights_ema = self.ema.state_dict()
            filepath_ema = join(self.parent_dir, self.name, "weights_EMA", f"{type}_{n}.pt")
            torch.save(weights_ema, filepath_ema)

    @track_memory 
    def set_optim(self, type: str) -> torch.optim:
        """
        Sets a new adam optimizer to train b (type=beta) or f (type=alpha)
        """
        net = self.type_to_net[type]
        self.logger.debug(f"Getting optimizer {self.optim.__name__} for {net}")
        self.optimizer = self.optim(self.nets[net].parameters(), lr=self.lr)

    @track_memory
    def set_ema(self, type: str) -> EMA:
        """
        Sets a new EMA for the training of b (type=beta) or f (type=alpha)
        """
        net = self.type_to_net[type]
        self.logger.debug(f"Creating EMA for {net}")
        self.ema = EMA(self.nets[net])

    @track_memory
    def init_cache(self):
        self.cache = torch.zeros((self.N+1, self.cache_size, *self.data_shape), device=self.device)
        self.Z = torch.zeros((self.N, self.cache_size, *self.data_shape), device=self.device)
        self.ones = torch.ones((self.cache_size,), dtype=torch.int, device=self.device)

    @track_memory
    def init_loss(self):
        self.actual = torch.zeros((self.batch_size, *self.data_shape), device=self.device)

    def train_model(self):
        """
        Trains the DSB network for L iterations.
        """
        self.t0 = time.time() # so that update_status can access t0 to display time elapsed
        self.logger.info(f'Training started for {self.L} IPF iterations')
        self.logger.debug(f'IPF iteration n=0')
        self.init_cache()
        self.init_loss()
        self.ipf_step('beta', 0) # for the first iteration we want alpha_0=0 so we do it apart
        self.nets["F"] = self.model(max_len=self.N+1).to(self.device) # now alpha_1 can be initialized to noise (we then train it!)
        # we need to do this separation otherwise a network with ReLU activations and weights=0 have gradients=0 and thus doesn't train!
        for n in range(1, self.L):
            self.logger.info(f'IPF iteration n={n}')
            self.ipf_step('alpha', n)
            self.ipf_step('beta', n)
        self.update_status("alpha", self.L) # indicate we have terminated training
        dt = int(time.time()-self.t0) // 60
        self.logger.info(f'Training finished in {dt//60}h{dt%60:02d}min')

    def update_status(self, type: str, n: int):
        """
        Write current status (training or completed) in status.yaml file.
        """
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
        Performs one IPF step i.e. trains (half) the DSB network for level n i.e. solves a half-bridge.
        """
        self.update_status(type, n)
        self.logger.debug(f"IPF step {type} {n}/{self.L}")
        self.set_optim(type) # warm-start since use previous weights BUT we DO reset the optim
        if self.use_ema:
            self.set_ema(type) # EMA for network
        for step in range(self.n_epoch):
            if step % self.cache_period == 0:
                self.refresh_cache(type)
                self.refresh_cuda_cache() # to clear all the Memory Reservd unnecessarily created by Torch because we created a bunch of intermediary tensors by calling f/b so many times...
            k, Xk, Xkp = self.sample_cache()
            self.gradient_step(type, k, Xk, Xkp)
        self.save_weights(type, n)

    @track_memory
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
        X = torch.zeros((self.N+1, M, *self.data_shape), device=self.device)
        Z = torch.randn((self.N, M, *self.data_shape), device=self.device)
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
        if type == "beta":
            X[0].copy_(self.pdata.sample(M))
            with torch.no_grad():
                for k in range(self.N):
                    X[k+1].copy_(self.nets["F"](X[k], k*ones) + math.sqrt(2*self.gamma[k+1]) * Z[k])
        else:
            X[-1].copy_(self.pprior.sample(M))
            with torch.no_grad():
                for k in range(self.N, 0, -1):
                    X[k-1].copy_(self.nets["B"](X[k], k*ones) + math.sqrt(2*self.gamma[k]) * Z[k-1])

    @track_memory
    def refresh_cuda_cache(self):
        torch.cuda.empty_cache()

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
        B, F = self.nets["B"], self.nets["F"] # for clarity
        if type == 'beta':
            pred = B(Xkp, k+1)
            with torch.no_grad():
                self.actual.copy_(Xkp + (F(Xk, k) - F(Xkp, k))) # using equation (12)
        else:
            pred = F(Xk, k)
            with torch.no_grad():
                self.actual.copy_(Xk + (B(Xkp, k+1) - B(Xk, k+1))) # using equation (13)
        loss = nn.functional.mse_loss(pred, self.actual)
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


######################################################################
# networks used for prediction
######################################################################
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


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = build_pe(d_model=d_model, max_len=max_len)
        self.register_buffer('pe', pe) # to make sure it's non-trainable
    
    def forward(self, x: torch.IntTensor) -> torch.Tensor:
        return self.pe[x]


class Dense(nn.Module):
  """Fully connected layer + reshape outputs to feature maps (i.e. with width and height of 1 each)."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[...,None,None] # same as [:,:,None,None]


class PositionalUNet(nn.Module):
  "Position-conditional U-Net to estimate the score network."""
  def __init__(self, max_len: int, n_channels: int=1, channels: list[int]=[32,64,128,256], embed_dim: int=128, init_zero: bool=False):
    """Initialize a time-dependent score-based network.
    Args:
      channels: The number of channels for feature maps for each layer.
      embed_dim: The dimensionality of the Gaussian random feature embedding.
    """
    super().__init__()
    # embed time
    pe = build_pe(d_model=embed_dim, max_len=max_len)
    self.register_buffer('pe', pe) # to make sure it's non-trainable
    self.embed = nn.Sequential(
        nn.Linear(embed_dim, embed_dim),
        nn.SiLU(),
        nn.Linear(embed_dim, embed_dim)
    )
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
    embedding = self.activation(self.embed(self.pe[t]))
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



# use time embedding instead
class TimeEmbedding(nn.Module):
  """Gaussian random features for encoding time."""
  def __init__(self, n_channels: int=1, channels: list[int]=[32,64,128,256], embed_dim: int=128, max_len: int=51, init_zero: bool=False):
    """Initialize Gaussian random features.
    Args:
      embed_dim: the dimensionality of the output embedding.
      scale: the scale of the Gaussian random features.
    """
    super().__init__()
    # self.W = nn.Parameter(torch.randn(embed_dim//2)*scale, requires_grad=False)
    self.W = nn.Parameter(torch.randn(embed_dim//2), requires_grad=False)
  def forward(self, x):
    theta = x[:,None]*self.W[None,:] * 2 * torch.pi
    return torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
  

class TimeUNet(nn.Module):
  "Position-conditional U-Net to estimate the score network."""
  def __init__(self, n_channels: int=1, channels: list[int]=[32,64,128,256], embed_dim: int=64, max_len: int=51, init_zero: bool=False):
    """Initialize a time-dependent score-based network.
    Args:
      channels: The number of channels for feature maps for each layer.
      embed_dim: The dimensionality of the Gaussian random feature embedding.
    """
    super().__init__()
    # embed time
    self.embed = nn.Sequential(TimeEmbedding(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
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

  def forward(self, x: torch.Tensor, t: float):
    # embed time
    embedding = self.activation(self.embed(t))
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