""""
Todo:
    - dropout? After token embeddings, final classifier layer
    - 24 layers?
    - Label smoothing?
"""
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------

class SelfAttention(nn.Module):

    def __init__(self, config, is_causal):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # Dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)  # Dropout after attention
        self.resid_dropout = nn.Dropout(config.resid_pdrop)  # Dropout after projection
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.is_causal = is_causal
        

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.attn_dropout(y) # dropout after attention
        # output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y) # dropout after projection
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.fc_dropout = nn.Dropout(config.mlp_pdrop)
        self.proj_dropout = nn.Dropout(config.mlp_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.fc_dropout(x) # Dropout after activation
        x = self.c_proj(x)
        x = self.proj_dropout(x) # Dropout after projection
        return x
    
class EncoderBlock(nn.Module):

    def __init__(self, encoder_config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(encoder_config.n_embd)
        self.self_attn = SelfAttention(encoder_config, is_causal=False)
        self.ln_2 = nn.LayerNorm(encoder_config.n_embd)
        self.mlp = MLP(encoder_config)

    def forward(self, encoder_x):
        encoder_x = encoder_x + self.self_attn(self.ln_1(encoder_x))
        encoder_x = encoder_x + self.mlp(self.ln_2(encoder_x))
        return encoder_x

@dataclass
class EncoderConfig:
    block_size: int = 16 # max sequence length (articles)
    vocab_size: int = 50258 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    n_classes: int = 2 # number of sentiment classes
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    mlp_pdrop: float = 0.1

class BERT(nn.Module):

    def __init__(self, encoder_config):
        super().__init__()
        self.encoder_config = encoder_config

        self.transformer = nn.ModuleDict(dict(
            encoder_wte = nn.Embedding(self.encoder_config.vocab_size, self.encoder_config.n_embd),
            encoder_wpe = nn.Embedding(self.encoder_config.block_size, self.encoder_config.n_embd),
            encoder_h = nn.ModuleList([EncoderBlock(self.encoder_config) for _ in range(self.encoder_config.n_layer)]),
            ln_f = nn.LayerNorm(self.encoder_config.n_embd),
        ))
        self.lm_head = nn.Linear(self.encoder_config.n_embd, self.encoder_config.n_classes, bias=False)

        # weight sharing scheme
        # self.transformer.encoder_wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.encoder_config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, encoder_idx, targets=None):
        # encoder_idx is of shape (B, T)
        B, encoder_T = encoder_idx.size()
        assert encoder_T <= self.encoder_config.block_size, f"Cannot forward sequence of length {encoder_T}, block size is only {self.encoder_config.block_size}"
        # forward the token and posisition embeddings
        encoder_pos = torch.arange(0, encoder_T, dtype=torch.long, device=encoder_idx.device) # shape (encoder_T)
        encoder_pos_emb = self.transformer.encoder_wpe(encoder_pos) # position embeddings of shape (encoder_T, n_embd)
        encoder_tok_emb = self.transformer.encoder_wte(encoder_idx) # token embeddings of shape (B, encoder_T, n_embd)
        encoder_x = encoder_tok_emb + encoder_pos_emb

        for i, encoder_block in enumerate(self.transformer.encoder_h):
            encoder_x = encoder_block(encoder_x)

        # forward the final layernorm and the classifier
        encoder_x = self.transformer.ln_f(encoder_x[:, :1, :]) # (64, 1, 768)
        logits = self.lm_head(encoder_x) # (B, 1(CLS token), 3(classes))
        loss = None
        accuracy = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.encoder_config.n_classes), targets) # logits.view(-1, 2): (64, 1, 2) -> (64, 2)
            predicted_labels = torch.argmax(logits.view(-1, self.encoder_config.n_classes), dim=1)
            accuracy = (predicted_labels == targets).float().mean()
        return logits, loss, accuracy

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    ptt = torch.load(filename, weights_only=True)
    return ptt

class DataLoaderLite:
    def __init__(self, B, encoder_T, process_rank, num_processes, split):
        self.B = B  # Number of articles per batch
        self.encoder_T = encoder_T  # Tokens per article (1024)
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        if self.split == "train":
            self.posts_file_path = r"sentiment140/tweets.pt"
            self.labels_file_path = r"sentiment140/labels.pt"
        elif self.split == "test":
            self.posts_file_path = r"sentiment140/test_tweets.pt"
            self.labels_file_path = r"sentiment140/test_labels.pt"

        assert os.path.exists(self.posts_file_path), f"File {self.posts_file_path} not found"
        assert os.path.exists(self.labels_file_path), f"File {self.labels_file_path} not found"
        
        self.reset()

    def reset(self):
        self.posts_tokens = load_tokens(self.posts_file_path).to(device)
        self.labels_tokens = load_tokens(self.labels_file_path).to(device)
        self.current_posts_position = self.B * self.encoder_T * self.process_rank
        self.current_labels_position = self.B * self.process_rank

    def next_batch(self):
        # Get posts and labels
        posts_buf = self.posts_tokens[self.current_posts_position : self.current_posts_position+B*self.encoder_T]
        labels_buf = self.labels_tokens[self.current_labels_position : self.current_labels_position+B]
        # Posts
        encoder_x = (posts_buf[:]).view(self.B, self.encoder_T) # reshape encoder (posts) inputs to 64 * 1024
        # Labels
        # y = (labels_buf[:]).view(self.B)
        y = labels_buf

        # advance the position in the posts and labels tensors
        self.current_posts_position += self.B * self.encoder_T * self.num_processes
        self.current_labels_position += self.B * self.num_processes
        # if loading the next batch would be out of bounds, reset to the start of the posts array
        if self.current_posts_position + (self.B * self.encoder_T * self.num_processes + 1) > len(self.posts_tokens):
            self.current_posts_position = self.B * self.encoder_T * self.process_rank
        if self.current_labels_position + (self.B * self.num_processes + 1) > len(self.labels_tokens):
            self.current_labels_position = self.B * self.process_rank
        
        return encoder_x, y

# -----------------------------------------------------------------------------
# simple launch:
# python trainSummarizer.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 trainSentiment.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

mode = input("Would you like to train or generate? (Enter t or g): ").lower()

if mode == "t":
    total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
    B = 64 # micro batch size
    encoder_T = EncoderConfig.block_size # article sequence length (1024)
    assert total_batch_size % (B * encoder_T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * encoder_T * ddp_world_size)
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, encoder_T=encoder_T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    test_loader = DataLoaderLite(B=B, encoder_T=encoder_T, process_rank=ddp_rank, num_processes=ddp_world_size, split="test")

torch.set_float32_matmul_precision('high')

# create model
model = BERT(EncoderConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 18e-4
min_lr = max_lr * 0.1
warmup_steps = 15
max_steps = 512 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# Load the checkpoint if it exists, otherwise the model will train from scratch 
checkpoint_path = "checkpoints/checkpoint_1000.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Load model parameters
    state_dict = checkpoint['model_state_dict']
    raw_model_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("module."):
            new_key = key[len("module."):]  # strip the prefix
        raw_model_state_dict[new_key] = value
    raw_model.load_state_dict(raw_model_state_dict)

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Retrieve additional training state (e.g., step count)

    step = checkpoint['step']
    start_step = step + 1

    if master_process:
        print(f"Checkpoint loaded successfully! Resuming from step {step}.")
else:
    start_step = 0  # Starting from scratch if no checkpoint is found
    print("No checkpoint found. Initializing model from scratch.")

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log1.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

def generate(tweet=None):
    model.eval()

    if tweet == None:
        tweet = input("Enter a tweet to classify: ")

    tweet_tokens = enc.encode_ordinary(tweet)
    eot = enc._special_tokens['<|endoftext|>']  # Special <|endoftext|> token
    enc._special_tokens['<|pad|>'] = eot + 1
    pad = enc._special_tokens['<|pad|>'] # pad token
    if len(tweet_tokens) < 15: # Pad with special token if shorter than 15
        tweet_tokens = [eot] + tweet_tokens + [pad] * (15 - len(tweet_tokens))
    else: # Truncate if longer than 15
        tweet_tokens = [eot] + tweet_tokens[:15]
    
    tweet_tokens = torch.tensor(tweet_tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # forward the model to get the logits
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss, accuracy = model(tweet_tokens) # logits: (1, 1(CLS token), 2(classes))
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs)
        if pred == 0:
            label = "negative"
        elif pred == 1:
            label = "positive"
        print(f"Prediction: {label}")

# Saving model and optimizer state at a checkpoint
def save_checkpoint(model, optimizer, step, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at step {step} to {checkpoint_path}")

def train():
    train_losses = []
    local_dir = "train_loss"
    LOSS_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(LOSS_DIR, exist_ok=True)
    for step in range(start_step, max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # once in a while evaluate our validation loss
        if step % 10 == 0 or last_step:
            model.eval()
            test_loader.reset()
            with torch.no_grad():
                test_loss_accum = 0.0
                test_loss_steps = 20
                test_accuracy_accum = 0.0
                for _ in range(test_loss_steps):
                    x, y = test_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss, test_accuracy = model(x, y)
                    test_accuracy = test_accuracy / test_loss_steps
                    test_accuracy_accum += test_accuracy
                    loss = loss / test_loss_steps
                    test_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(test_loss_accum, op=dist.ReduceOp.AVG)
                dist.all_reduce(test_accuracy_accum, op=dist.ReduceOp.AVG)

            if master_process:
                print(f"accuracy: {test_accuracy_accum.item():.2%} | validation loss: {test_loss_accum.item():.6f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {test_loss_accum.item():.4f}\n")

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        accuracy_accum = 0.0
        for micro_step in range(grad_accum_steps):
            encoder_x, y = train_loader.next_batch()
            encoder_x, y = encoder_x.to(device), y.to(device)
            # added after video, this field is also used by the forward pass.
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss, accuracy = model(encoder_x, y)
            accuracy = accuracy / grad_accum_steps
            accuracy_accum += accuracy
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(accuracy_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work
        if (master_process and (step > 0 and step % 50 == 0) or last_step):
            save_checkpoint(model, optimizer, step)
            train_loss_tensor = torch.tensor(train_losses, dtype=torch.long)
            train_loss_path = os.path.join(LOSS_DIR, "train_loss1.pt")
            torch.save(train_loss_tensor, train_loss_path)
            plt.plot(train_losses)
            plt.xlabel("Steps")
            plt.ylabel("Train Loss")
            plt.title("Training Loss")
            plt.savefig("training_loss_curve2.png")

        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.encoder_T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | accuracy: {accuracy_accum.item():.2%} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
            train_losses += [loss_accum.item()]

if mode == "t":
    train()
elif mode == "g":
    while True:
        generate()

if ddp:
    destroy_process_group()
