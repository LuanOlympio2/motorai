import math
import mmap
import os
import random
import struct
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from tokenizers import ByteLevelBPETokenizer
    HAS_TOKENIZER = True
except ImportError:
    ByteLevelBPETokenizer = None
    HAS_TOKENIZER = False


CONFIG = {
    "lattice_size": 256,
    "c": 1.0,
    "gamma": 0.01,
    "dt": 0.01,
    "sub_steps": 1,
    "tau_init": 0.5,
    "threshold": 0.32,
    "vocab_size": 2000,
    "embed_dim": 128,
    "seq_len": 128,
    "data_stride": 64,
    "chunk_size": 64,
    "batch_size": 64,
    "lr": 0.00001,
    "grad_clip": 1.0,
    "accum_steps": 1,
    "warmup_steps": 200,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 1337,
    "ab_runs": [256],
    "ab_target_updates": 151000,
    "val_interval_updates": 1000,
    "val_max_batches": 50,
    "val_token_count": 65536,
    "driver2_scale": 0.05,
    "driver2_ab_runs": [0.05, 0.02],
    "driver2_ab_target_updates": 1000,
    "driver2_ab_val_interval_updates": 100,
    "driver2_ab_seed": 1337,
    "speech_interval_updates": 5000,
    "speech_gen_steps": 64,
    "speech_temperature": 0.8,
    "speech_top_k": 40,
    "log_dir": "./logs",
    "save_checkpoints": True,
    "run_log_name": "logs/train_v4_toddler_sft_run1.log",
    "oom_retry_from_batch_size": 0,
    "resume_from_checkpoint": "motor_toddler_v3_step150000.pth",
    "corpus_path": "sft_corpus_v2.bin",
    "save_interval_updates": 100,
    "label_smoothing": 0.0,
    "eos_id": 3,
    "eos_weight": 3.0,
    "short_seq_len": 64,
    "short_ratio": 0.15,
}

SPEECH_PROMPTS = [
    "Olá, como você está?",
    "O Brasil é um país…",
    "Explique o que é astronomia.",
]


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * x / rms


class TokenShift(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mix = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x_prev = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        alpha = torch.sigmoid(self.mix)
        return alpha * x + (1.0 - alpha) * x_prev


class _FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x * 2.0)
        return grad * 2.0 * s * (1.0 - s)


spike_fn = _FastSigmoid.apply


class MotorCore(nn.Module):
    def __init__(self, lattice_size, c=1.0, gamma=0.01, dt=0.01, dilation=17):
        super().__init__()
        self.c = c
        self.gamma = gamma
        self.dt = dt
        self.dilation = dilation
        self.register_buffer("lap_local", torch.tensor([[[1.0, -2.0, 1.0]]]))
        self.register_buffer("biharm_kernel", torch.tensor([[[1.0, -4.0, 6.0, -4.0, 1.0]]]))
        self.lap_mix = nn.Parameter(torch.tensor(0.5))
        self.hyperviscosity = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("spectral_norm", torch.tensor(0.2))

    def _laplacian(self, u):
        u_pad = torch.cat([u[:, -1:], u, u[:, :1]], dim=1).unsqueeze(1)
        lap_loc = F.conv1d(u_pad, self.lap_local).squeeze(1)
        d = self.dilation
        lap_dil = torch.roll(u, -d, dims=1) - 2.0 * u + torch.roll(u, d, dims=1)
        alpha = torch.sigmoid(self.lap_mix)
        return (alpha * lap_loc + (1.0 - alpha) * lap_dil) * self.spectral_norm

    def _biharmonic(self, u):
        u_pad = torch.cat([u[:, -2:], u, u[:, :2]], dim=1).unsqueeze(1)
        return F.conv1d(u_pad, self.biharm_kernel).squeeze(1)

    def forward(self, u, v, forcing, sub_steps=1):
        dt_sub = self.dt / sub_steps
        for _ in range(sub_steps):
            lap = self._laplacian(u)
            biharm = self._biharmonic(u)
            hv = F.softplus(self.hyperviscosity)
            ke = (v ** 2).mean(dim=1, keepdim=True).clamp(min=1e-6)
            agc = 1.0 / (ke.sqrt() + 1.0)
            modulated_forcing = forcing * agc
            accel = (self.c ** 2) * lap - self.gamma * v - hv * biharm + modulated_forcing
            v = v + accel * dt_sub
            u = u + v * dt_sub
        grad_u = torch.diff(u, dim=-1, prepend=u[:, -1:])
        energy = 0.5 * (v ** 2 + (self.c ** 2) * grad_u ** 2)
        return u, v, energy


class ALIFCell(nn.Module):
    def __init__(self, input_size, hidden_size, tau_init=0.5, threshold_init=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau = nn.Parameter(torch.tensor(tau_init))
        self.thr = nn.Parameter(torch.tensor(threshold_init))
        self.rho = nn.Parameter(torch.tensor(0.97))
        self.alpha = nn.Parameter(torch.tensor(0.15))
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_rec.weight)
        with torch.no_grad():
            self.W_in.weight.mul_(0.1)
            self.W_rec.weight.mul_(0.1)

    def forward(self, x, state):
        spike_prev, mem, theta = state
        x_norm = self.norm(x)
        current = self.W_in(x_norm) + self.W_rec(spike_prev)
        decay = torch.sigmoid(self.tau)
        mem = mem * decay + current * (1.0 - decay)
        thr_base = F.softplus(self.thr)
        rho = torch.sigmoid(self.rho)
        alpha = F.softplus(self.alpha)
        theta = rho * theta + alpha * spike_prev
        spike = spike_fn(mem - thr_base - theta)
        mem = mem * (1.0 - spike)
        return spike, mem, theta

    def zero_state(self, batch_size, device):
        z = torch.zeros(batch_size, self.W_in.out_features, device=device)
        theta = torch.zeros(batch_size, self.W_in.out_features, device=device)
        return z, z.clone(), theta


class MotorAI_baby(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        L = cfg["lattice_size"]
        E = cfg["embed_dim"]
        V = cfg["vocab_size"]
        self.token_shift = TokenShift(E)
        self.embedding = nn.Embedding(V, E)
        self.driver1 = nn.Linear(E, L)
        self.motor1 = MotorCore(L, cfg["c"], cfg["gamma"], cfg["dt"])
        self.u_proj1 = nn.Linear(L, L)
        self.e_proj1 = nn.Linear(L, L)
        self.lif1 = ALIFCell(L, L, cfg["tau_init"], cfg["threshold"])
        self.driver2 = nn.Linear(L, L)
        self.motor2 = MotorCore(L, cfg["c"], cfg["gamma"], cfg["dt"])
        self.u_proj2 = nn.Linear(L, L)
        self.e_proj2 = nn.Linear(L, L)
        self.lif2 = ALIFCell(L, L, cfg["tau_init"], cfg["threshold"])
        with torch.no_grad():
            self.lif1.W_rec.weight.mul_(0.1)
            self.lif2.W_rec.weight.mul_(0.1)
        self.out_norm = RMSNorm(L)
        self.fc = nn.Linear(L, V)
        self.cfg = cfg
        nn.init.xavier_uniform_(self.driver1.weight)
        with torch.no_grad():
            self.driver1.weight.mul_(0.03)
        nn.init.zeros_(self.driver1.bias)
        driver2_scale = float(cfg.get("driver2_scale", 0.05))
        nn.init.xavier_uniform_(self.driver2.weight)
        with torch.no_grad():
            self.driver2.weight.mul_(driver2_scale)
        nn.init.zeros_(self.driver2.bias)
        nn.init.xavier_uniform_(self.u_proj1.weight)
        nn.init.xavier_uniform_(self.e_proj1.weight)
        nn.init.xavier_uniform_(self.u_proj2.weight)
        nn.init.xavier_uniform_(self.e_proj2.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _zero_state(self, batch_size, device):
        L = self.cfg["lattice_size"]
        u1 = torch.zeros(batch_size, L, device=device)
        v1 = torch.zeros(batch_size, L, device=device)
        s1 = self.lif1.zero_state(batch_size, device)
        u2 = torch.zeros(batch_size, L, device=device)
        v2 = torch.zeros(batch_size, L, device=device)
        s2 = self.lif2.zero_state(batch_size, device)
        return (u1, v1, s1), (u2, v2, s2)

    def forward(self, x, state=None):
        batch_size, seq_len = x.size()
        device = x.device
        emb = self.token_shift(self.embedding(x))
        if state is None:
            state1, state2 = self._zero_state(batch_size, device)
        else:
            state1, state2 = state
        u1, v1, s1 = state1
        u2, v2, s2 = state2
        outputs = []
        spike_rates1 = []
        spike_rates2 = []
        for t in range(seq_len):
            inp_t = emb[:, t, :]
            f1 = self.driver1(inp_t)
            u1, v1, e1 = self.motor1(u1, v1, f1)
            u_in1 = torch.tanh(self.u_proj1(u1))
            e_in1 = torch.tanh(self.e_proj1(e1))
            spike1, mem1, theta1 = self.lif1(u_in1 + e_in1, s1)
            s1 = (spike1, mem1, theta1)
            f2 = self.driver2(spike1)
            u2, v2, e2 = self.motor2(u2, v2, f2)
            u_in2 = torch.tanh(self.u_proj2(u2))
            e_in2 = torch.tanh(self.e_proj2(e2))
            spike2, mem2, theta2 = self.lif2(u_in2 + e_in2, s2)
            s2 = (spike2, mem2, theta2)
            outputs.append(spike2)
            spike_rates1.append(spike1.mean().item())
            spike_rates2.append(spike2.mean().item())
        spike_seq = torch.stack(outputs, dim=1)
        logits = self.fc(self.out_norm(spike_seq))
        u1 = u1.detach()
        v1 = v1.detach()
        s1 = (s1[0].detach(), s1[1].detach(), s1[2].detach())
        u2 = u2.detach()
        v2 = v2.detach()
        s2 = (s2[0].detach(), s2[1].detach(), s2[2].detach())
        spike_rate1 = sum(spike_rates1) / len(spike_rates1)
        spike_rate2 = sum(spike_rates2) / len(spike_rates2)
        return logits, ((u1, v1, s1), (u2, v2, s2)), (spike_rate1, spike_rate2)


class MotorAI_toddler(MotorAI_baby):
    def __init__(self, cfg):
        super().__init__(cfg)
        for layer in (self.lif1.W_in, self.lif1.W_rec, self.lif2.W_in, self.lif2.W_rec):
            nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_in", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)


class BPETextDataset(Dataset):
    def __init__(self, seq_len, stride, bin_path=None, start_token=0, end_token=None):
        self.seq_len = int(seq_len)
        self.stride = max(1, int(stride))
        self.bin_path = bin_path or find_bin_path()
        self.token_bytes = token_bytes_for_path(self.bin_path)
        self.total_bytes = os.path.getsize(self.bin_path)
        self.total_tokens = self.total_bytes // self.token_bytes
        self.start_token = max(0, int(start_token))
        self.end_token = self.total_tokens if end_token is None else min(self.total_tokens, int(end_token))
        max_start = self.end_token - self.seq_len - 1
        if max_start < self.start_token:
            self.offsets = []
        else:
            self.offsets = list(range(self.start_token, max_start + 1, self.stride))
        self._f = None
        self._mm = None
        self._open_handles()

    def _open_handles(self):
        self._f = open(self.bin_path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def _close_handles(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._f is not None:
            self._f.close()
            self._f = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_f"] = None
        state["_mm"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_handles()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        start = self.offsets[int(idx)]
        raw = self._mm[start * self.token_bytes : (start + self.seq_len + 1) * self.token_bytes]
        if self.token_bytes == 4:
            tokens = struct.unpack(f"<{self.seq_len + 1}i", raw)
        else:
            tokens = struct.unpack(f"<{self.seq_len + 1}H", raw)
        data = torch.tensor(tokens, dtype=torch.long)
        return data[:-1], data[1:]

    def __del__(self):
        try:
            self._close_handles()
        except Exception:
            pass


class InMemoryTokenDataset(Dataset):
    def __init__(self, token_ids, seq_len, stride):
        self.seq_len = int(seq_len)
        self.stride = max(1, int(stride))
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        max_start = len(token_ids) - self.seq_len - 1
        if max_start < 0:
            self.offsets = []
        else:
            self.offsets = list(range(0, max_start + 1, self.stride))

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        start = self.offsets[int(idx)]
        data = self.tokens[start : start + self.seq_len + 1]
        return data[:-1], data[1:]


class LargeBPETextDataset(Dataset):
    def __init__(self, seq_len, stride, bin_path=None, start_token=0, end_token=None):
        self.seq_len = int(seq_len)
        self.stride = max(1, int(stride))
        self.bin_path = bin_path or resolve_corpus_path()
        self.token_bytes = token_bytes_for_path(self.bin_path)
        self.total_bytes = os.path.getsize(self.bin_path)
        self.total_tokens = self.total_bytes // self.token_bytes
        self.start_token = max(0, int(start_token))
        self.end_token = self.total_tokens if end_token is None else min(self.total_tokens, int(end_token))
        max_start = self.end_token - self.seq_len - 1
        if max_start < self.start_token:
            self.length = 0
        else:
            self.length = ((max_start - self.start_token) // self.stride) + 1
        self._f = None
        self._mm = None
        self._open_handles()

    def _open_handles(self):
        self._f = open(self.bin_path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def _close_handles(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._f is not None:
            self._f.close()
            self._f = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_f"] = None
        state["_mm"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_handles()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = self.start_token + int(idx) * self.stride
        raw = self._mm[start * self.token_bytes : (start + self.seq_len + 1) * self.token_bytes]
        if self.token_bytes == 4:
            tokens = struct.unpack(f"<{self.seq_len + 1}i", raw)
        else:
            tokens = struct.unpack(f"<{self.seq_len + 1}H", raw)
        data = torch.tensor(tokens, dtype=torch.long)
        return data[:-1], data[1:]

    def __del__(self):
        try:
            self._close_handles()
        except Exception:
            pass


class MixedSeqDataset(Dataset):
    def __init__(self, long_dataset, short_dataset):
        self.long_dataset = long_dataset
        self.short_dataset = short_dataset
        self.short_base = len(long_dataset)

    def __len__(self):
        return self.short_base + len(self.short_dataset)

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < self.short_base:
            return self.long_dataset[idx]
        return self.short_dataset[idx - self.short_base]


class MixedSeqBatchSampler:
    def __init__(self, long_length, short_length, batch_size, short_ratio):
        self.long_length = int(long_length)
        self.short_length = int(short_length)
        self.batch_size = int(batch_size)
        self.short_ratio = float(short_ratio)
        if self.long_length <= 0 and self.short_length <= 0:
            self.num_batches = 0
        else:
            self.num_batches = max(1, self.long_length // max(self.batch_size, 1))
        if self.short_length > 0 and self.short_ratio > 0.0 and self.num_batches > 0:
            self.short_batches = min(self.num_batches, int(round(self.num_batches * self.short_ratio)))
        else:
            self.short_batches = 0

    def __len__(self):
        return self.num_batches

    def _sample_batch(self, dataset_length, offset):
        if dataset_length <= 0:
            return None
        if dataset_length <= self.batch_size:
            return list(range(offset, offset + dataset_length))
        max_start = dataset_length - self.batch_size
        start = int(torch.randint(0, max_start + 1, (1,)).item())
        return list(range(offset + start, offset + start + self.batch_size))

    def __iter__(self):
        if self.num_batches <= 0:
            return
        remaining_short = self.short_batches
        remaining_total = self.num_batches
        short_offset = self.long_length
        for _ in range(self.num_batches):
            use_short = False
            if self.short_length > 0 and remaining_short > 0:
                short_prob = remaining_short / max(remaining_total, 1)
                use_short = bool(torch.rand(1).item() < short_prob)
            remaining_total -= 1
            if use_short:
                remaining_short -= 1
                batch = self._sample_batch(self.short_length, short_offset)
            else:
                batch = self._sample_batch(self.long_length, 0)
                if batch is None:
                    batch = self._sample_batch(self.short_length, short_offset)
            if batch:
                yield batch


def token_bytes_for_path(bin_path):
    name = os.path.basename(bin_path).lower()
    if "4000" in name:
        return 4
    return 2


def find_bin_path():
    for path in [
        "src/data/corpus_bpe_v6.bin",
        "data/corpus_bpe_v6.bin",
        "corpus_bpe_v6.bin",
        "src/data/corpus_bpe_v6_clean.bin",
        "data/corpus_bpe_v6_clean.bin",
        "corpus_bpe_v6_clean.bin",
        "corpus_bpe4000.bin",
        "src/data/corpus_bpe4000.bin",
        "data/corpus_bpe4000.bin",
    ]:
        if os.path.exists(path):
            return path
    sys.exit()


def find_tokenizer_dir():
    for path in [
        "tokenizer",
        "src/data/bpe_tokenizer_v5",
        "src/data/bpe_tokenizer_v6",
        "data/bpe_tokenizer_v6",
        "bpe_tokenizer_v6",
        "tokenizer_bpe4000",
        "src/data/tokenizer_bpe4000",
        "data/tokenizer_bpe4000",
    ]:
        if os.path.exists(path):
            return path
    return None


def resolve_corpus_path(cfg=None):
    corpus_path = CONFIG["corpus_path"] if cfg is None else str(cfg.get("corpus_path", CONFIG["corpus_path"]))
    candidates = [
        os.path.abspath(corpus_path),
        os.path.abspath(os.path.join("src", "data", corpus_path)),
        os.path.abspath(os.path.join("data", corpus_path)),
    ]
    original_dir = os.path.dirname(os.path.abspath(find_bin_path()))
    candidates.append(os.path.abspath(os.path.join(original_dir, os.path.basename(corpus_path))))
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(corpus_path)


class RunLogger:
    def __init__(self, path):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def log(self, msg):
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            safe = msg.encode(enc, errors="replace").decode(enc, errors="replace")
            print(safe, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


class NullLogger:
    def log(self, msg):
        return None

    def close(self):
        return None


def safe_print(text=""):
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = str(text).encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe, flush=True)


global_tokenizer = None


def checkpoint_name_for_step(step):
    return f"motor_toddler_v4_step{int(step)}.pth"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_env_overrides(cfg):
    out = dict(cfg)
    if "MOTOR_SEED" in os.environ:
        out["seed"] = int(os.environ["MOTOR_SEED"])
    if "MOTOR_AB_RUNS" in os.environ:
        out["ab_runs"] = [int(x.strip()) for x in os.environ["MOTOR_AB_RUNS"].split(",") if x.strip()]
    if "MOTOR_AB_TARGET_UPDATES" in os.environ:
        out["ab_target_updates"] = int(os.environ["MOTOR_AB_TARGET_UPDATES"])
    if "MOTOR_VAL_INTERVAL_UPDATES" in os.environ:
        out["val_interval_updates"] = int(os.environ["MOTOR_VAL_INTERVAL_UPDATES"])
    if "MOTOR_VAL_MAX_BATCHES" in os.environ:
        out["val_max_batches"] = int(os.environ["MOTOR_VAL_MAX_BATCHES"])
    if "MOTOR_SPEECH_INTERVAL_UPDATES" in os.environ:
        out["speech_interval_updates"] = int(os.environ["MOTOR_SPEECH_INTERVAL_UPDATES"])
    if "MOTOR_SPEECH_GEN_STEPS" in os.environ:
        out["speech_gen_steps"] = int(os.environ["MOTOR_SPEECH_GEN_STEPS"])
    if "MOTOR_SPEECH_TEMPERATURE" in os.environ:
        out["speech_temperature"] = float(os.environ["MOTOR_SPEECH_TEMPERATURE"])
    if "MOTOR_SPEECH_TOP_K" in os.environ:
        out["speech_top_k"] = int(os.environ["MOTOR_SPEECH_TOP_K"])
    if "MOTOR_DRIVER2_SCALE" in os.environ:
        out["driver2_scale"] = float(os.environ["MOTOR_DRIVER2_SCALE"])
    if "MOTOR_DRIVER2_AB_RUNS" in os.environ:
        out["driver2_ab_runs"] = [float(x.strip()) for x in os.environ["MOTOR_DRIVER2_AB_RUNS"].split(",") if x.strip()]
    if "MOTOR_DRIVER2_AB_TARGET_UPDATES" in os.environ:
        out["driver2_ab_target_updates"] = int(os.environ["MOTOR_DRIVER2_AB_TARGET_UPDATES"])
    if "MOTOR_DRIVER2_AB_VAL_INTERVAL_UPDATES" in os.environ:
        out["driver2_ab_val_interval_updates"] = int(os.environ["MOTOR_DRIVER2_AB_VAL_INTERVAL_UPDATES"])
    if "MOTOR_DRIVER2_AB_SEED" in os.environ:
        out["driver2_ab_seed"] = int(os.environ["MOTOR_DRIVER2_AB_SEED"])
    if "MOTOR_THRESHOLD" in os.environ:
        out["threshold"] = float(os.environ["MOTOR_THRESHOLD"])
    if "MOTOR_EMBED_DIM" in os.environ:
        out["embed_dim"] = int(os.environ["MOTOR_EMBED_DIM"])
    if "MOTOR_RUN_LOG_NAME" in os.environ:
        out["run_log_name"] = os.environ["MOTOR_RUN_LOG_NAME"]
    if "MOTOR_RESUME_CHECKPOINT" in os.environ:
        out["resume_from_checkpoint"] = os.environ["MOTOR_RESUME_CHECKPOINT"]
    if "MOTOR_CORPUS_PATH" in os.environ:
        out["corpus_path"] = os.environ["MOTOR_CORPUS_PATH"]
    if "MOTOR_LABEL_SMOOTHING" in os.environ:
        out["label_smoothing"] = float(os.environ["MOTOR_LABEL_SMOOTHING"])
    if "MOTOR_EOS_ID" in os.environ:
        out["eos_id"] = int(os.environ["MOTOR_EOS_ID"])
    if "MOTOR_EOS_WEIGHT" in os.environ:
        out["eos_weight"] = float(os.environ["MOTOR_EOS_WEIGHT"])
    if "MOTOR_SHORT_SEQ_LEN" in os.environ:
        out["short_seq_len"] = int(os.environ["MOTOR_SHORT_SEQ_LEN"])
    if "MOTOR_SHORT_RATIO" in os.environ:
        out["short_ratio"] = float(os.environ["MOTOR_SHORT_RATIO"])
    if "MOTOR_SAVE_INTERVAL_UPDATES" in os.environ:
        out["save_interval_updates"] = int(os.environ["MOTOR_SAVE_INTERVAL_UPDATES"])
    if "MOTOR_BATCH_SIZE" in os.environ:
        out["batch_size"] = int(os.environ["MOTOR_BATCH_SIZE"])
    if "MOTOR_ACCUM_STEPS" in os.environ:
        out["accum_steps"] = int(os.environ["MOTOR_ACCUM_STEPS"])
    if "MOTOR_CHUNK_SIZE" in os.environ:
        out["chunk_size"] = int(os.environ["MOTOR_CHUNK_SIZE"])
    return out


def build_criterion(cfg):
    weights = torch.ones(int(cfg["vocab_size"]), device=cfg["device"])
    eos_id = int(cfg.get("eos_id", 3))
    if 0 <= eos_id < weights.numel():
        weights[eos_id] = float(cfg.get("eos_weight", 3.0))
    return nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )


def build_train_val_loaders(cfg, stride, logger):
    bin_path = resolve_corpus_path(cfg)
    token_bytes = token_bytes_for_path(bin_path)
    seq_len = int(cfg["seq_len"])
    short_seq_len = int(cfg.get("short_seq_len", 64))
    short_ratio = float(cfg.get("short_ratio", 0.15))
    batch_size = int(cfg["batch_size"])

    val_txt = os.path.join("src", "data", "val.txt")
    tokenizer_dir = find_tokenizer_dir() if HAS_TOKENIZER else None

    if os.path.exists(val_txt) and tokenizer_dir is not None:
        tokenizer = ByteLevelBPETokenizer.from_file(
            os.path.join(tokenizer_dir, "vocab.json"),
            os.path.join(tokenizer_dir, "merges.txt"),
        )
        with open(val_txt, "r", encoding="utf-8", errors="ignore") as f:
            val_text = f.read()
        val_ids = tokenizer.encode(val_text).ids
        val_ds = InMemoryTokenDataset(val_ids, seq_len, stride)
        long_train_ds = LargeBPETextDataset(seq_len, stride, bin_path=bin_path)
        short_train_ds = LargeBPETextDataset(short_seq_len, stride, bin_path=bin_path)
        logger.log(f"[VAL] usando src/data/val.txt | val_samples={len(val_ds):,}")
    else:
        total_tokens = os.path.getsize(bin_path) // token_bytes
        val_tokens = min(int(cfg["val_token_count"]), max(seq_len * 4, total_tokens // 10))
        split_start = max(0, total_tokens - val_tokens)
        long_train_ds = LargeBPETextDataset(
            seq_len,
            stride,
            bin_path=bin_path,
            start_token=0,
            end_token=split_start,
        )
        short_train_ds = LargeBPETextDataset(
            short_seq_len,
            stride,
            bin_path=bin_path,
            start_token=0,
            end_token=split_start,
        )
        val_ds = LargeBPETextDataset(
            seq_len,
            stride,
            bin_path=bin_path,
            start_token=split_start,
            end_token=total_tokens,
        )
        logger.log(
            f"[VAL] split deterministico do binario | train_tokens=[0,{split_start}) val_tokens=[{split_start},{total_tokens})"
        )

    train_ds = MixedSeqDataset(long_train_ds, short_train_ds)
    train_batch_sampler = MixedSeqBatchSampler(
        len(long_train_ds),
        len(short_train_ds),
        batch_size,
        short_ratio,
    )

    train_kwargs = {
        "batch_sampler": train_batch_sampler,
        "num_workers": int(cfg.get("num_workers", 0)),
        "pin_memory": bool(cfg.get("pin_memory", False)),
    }
    if train_kwargs["num_workers"] > 0:
        train_kwargs["persistent_workers"] = True

    val_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": bool(cfg.get("pin_memory", False)),
    }

    train_loader = DataLoader(train_ds, **train_kwargs)
    val_loader = DataLoader(val_ds, **val_kwargs)

    logger.log(
        f"[TRAIN] corpus={bin_path} | long_seq_len={seq_len} | short_seq_len={short_seq_len} "
        f"| short_ratio={short_ratio:.3f} | long_samples={len(long_train_ds):,} "
        f"| short_samples={len(short_train_ds):,} | train_batches={len(train_batch_sampler):,} "
        f"| short_batches={train_batch_sampler.short_batches:,}"
    )
    return train_loader, val_loader


def evaluate(model, val_loader, criterion, chunk_size, vocab_size, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for bidx, (x, y) in enumerate(val_loader):
            if max_batches is not None and bidx >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, t = x.size()
            state = None
            for start in range(0, t, chunk_size):
                end = min(start + chunk_size, t)
                logits, state, _ = model(x[:, start:end], state)
                y_chunk = y[:, start:end]
                loss = criterion(logits.reshape(-1, vocab_size), y_chunk.reshape(-1))
                n_tok = y_chunk.numel()
                total_loss += loss.item() * n_tok
                total_tokens += n_tok

    model.train()
    if total_tokens == 0:
        return float("nan"), float("nan")
    val_loss = total_loss / total_tokens
    val_ppl = math.exp(min(val_loss, 20.0))
    return val_loss, val_ppl


def save_checkpoint(model, optimizer, scheduler, cfg, update_step, global_step, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "cfg": dict(cfg),
            "update_step": int(update_step),
            "global_step": int(global_step),
        },
        path,
    )


@torch.no_grad()
def generate_speech(model, prompt, cfg):
    global global_tokenizer
    model.eval()
    device = next(model.parameters()).device

    tokenizer = global_tokenizer
    if tokenizer is None and HAS_TOKENIZER:
        tokenizer_dir = find_tokenizer_dir()
        if tokenizer_dir:
            tokenizer = ByteLevelBPETokenizer.from_file(
                os.path.join(tokenizer_dir, "vocab.json"),
                os.path.join(tokenizer_dir, "merges.txt"),
            )
            global_tokenizer = tokenizer

    if tokenizer is not None:
        token_ids = tokenizer.encode(prompt).ids
        if not token_ids:
            token_ids = [tokenizer.token_to_id("<unk>") or 0]
        eos_id = tokenizer.token_to_id("</s>")
        blocked_ids = {
            tokenizer.token_to_id("<pad>"),
            tokenizer.token_to_id("<s>"),
            tokenizer.token_to_id("<mask>"),
        }
        blocked_ids = {idx for idx in blocked_ids if idx is not None}
    else:
        token_ids = [45, 1152]
        eos_id = None
        blocked_ids = set()

    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    out_ids = list(token_ids)
    state = None
    _, state, _ = model(x, state)
    current_id = token_ids[-1]
    eos_generated = False

    for _ in range(int(cfg["speech_gen_steps"])):
        x = torch.tensor([[current_id]], dtype=torch.long, device=device)
        logits, state, _ = model(x, state)
        logits_last = logits[:, -1, :] / max(float(cfg["speech_temperature"]), 1e-6)
        for sid in blocked_ids:
            logits_last[:, sid] = float("-inf")
        top_k = int(cfg["speech_top_k"])
        if top_k > 0:
            vals, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
            logits_last = logits_last.masked_fill(logits_last < vals[:, -1:], float("-inf"))
        probs = F.softmax(logits_last, dim=-1)
        current_id = torch.multinomial(probs, 1).item()
        out_ids.append(current_id)
        if eos_id is not None and current_id == eos_id:
            eos_generated = True
            break

    model.train()
    if tokenizer is not None:
        text = tokenizer.decode(out_ids, skip_special_tokens=True)
    else:
        text = f"[Token IDs]: {out_ids}"
    if text.startswith(prompt):
        text = text[len(prompt):].lstrip()
    return text, eos_generated, len(out_ids) - len(token_ids)


def log_speech_checkpoints(model, logger, cfg, update_step):
    logger.log(f"[SPEECH] update_step={update_step}")
    for prompt in SPEECH_PROMPTS:
        out, eos_generated, generated_count = generate_speech(model, prompt, cfg)
        eos_text = "sim" if eos_generated else "nao"
        logger.log(
            f"[SPEECH] prompt={prompt} | eos={eos_text} | tokens_gerados={generated_count} -> {out[:240]}"
        )


def train_single_stride(base_cfg, stride, target_updates, log_path):
    cfg = dict(base_cfg)
    cfg["data_stride"] = int(stride)
    cfg["target_updates"] = int(target_updates)

    logger = RunLogger(log_path)
    try:
        set_seed(cfg["seed"])
        threshold_effective = float(F.softplus(torch.tensor(cfg["threshold"])).item())
        threshold_ref_baseline = float(F.softplus(torch.tensor(0.4)).item())
        threshold_ref_run = float(F.softplus(torch.tensor(cfg["threshold"])).item())
        checkpoint_path = str(cfg.get("resume_from_checkpoint", "")).strip()
        checkpoint_status = "disabled"
        logger.log(
            f"MOTOR AI TODDLER V4 | stride={cfg['data_stride']} | lattice_size={cfg['lattice_size']} "
            f"| embed_dim={cfg['embed_dim']} | batch_size={cfg['batch_size']} "
            f"| target_updates={cfg['target_updates']} | driver2_scale={cfg['driver2_scale']:.3f} "
            f"| threshold={cfg['threshold']:.4f} | threshold_effective={threshold_effective:.4f} "
            f"| checkpoint_load=pending"
        )
        logger.log("batch_size: " + str(int(cfg["batch_size"])))
        logger.log("accum_steps: " + str(int(cfg["accum_steps"])))
        logger.log("chunk_size: " + str(int(cfg["chunk_size"])))
        logger.log("driver2_scale: " + str(float(cfg["driver2_scale"])))
        logger.log("threshold_effective: " + str(float(threshold_effective)))
        logger.log("corpus_path: " + resolve_corpus_path(cfg))
        logger.log("label_smoothing: " + str(float(cfg.get("label_smoothing", 0.0))))
        logger.log("eos_id: " + str(int(cfg.get("eos_id", 3))))
        logger.log("eos_weight: " + str(float(cfg.get("eos_weight", 3.0))))
        logger.log("save_interval_updates: " + str(int(cfg.get("save_interval_updates", 1000))))
        logger.log("short_seq_len: " + str(int(cfg.get("short_seq_len", 64))))
        logger.log("short_ratio: " + str(float(cfg.get("short_ratio", 0.15))))
        if int(cfg.get("oom_retry_from_batch_size", 0)) > 0:
            logger.log(
                f"oom_retry_from_batch_size: {int(cfg['oom_retry_from_batch_size'])} -> {int(cfg['batch_size'])}"
            )
        logger.log(f"threshold_reference_baseline_softplus_0.4: {threshold_ref_baseline:.6f}")
        logger.log(f"threshold_reference_run_softplus_cfg: {threshold_ref_run:.6f}")

        train_loader, val_loader = build_train_val_loaders(cfg, cfg["data_stride"], logger)
        logger.log(f"Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")

        model = MotorAI_toddler(cfg).to(cfg["device"])
        logger.log("model_embed_dim: " + str(int(model.cfg["embed_dim"])))
        logger.log("model_lattice_size: " + str(int(model.cfg["lattice_size"])))
        logger.log("model_threshold_configured: " + str(float(model.cfg["threshold"])))
        logger.log("thr_lif1_init: " + str(float(F.softplus(model.lif1.thr).detach().cpu())))
        logger.log("thr_lif2_init: " + str(float(F.softplus(model.lif2.thr).detach().cpu())))
        logger.log("driver2_w_abs_mean: " + str(float(model.driver2.weight.detach().abs().mean().cpu())))

        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
        criterion = build_criterion(cfg)

        def lr_lambda(step):
            warmup = int(cfg["warmup_steps"])
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(1, 50000 - warmup)
            progress = min(progress, 1.0)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        global_step = 0
        update_step = 0
        chunk_size = int(cfg["chunk_size"])

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=cfg["device"])
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" not in checkpoint:
                raise KeyError("optimizer_state_dict ausente no checkpoint")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update_step = int(checkpoint.get("update_step", checkpoint.get("global_step", 0)))
            global_step = int(checkpoint.get("global_step", update_step))
            checkpoint_status = f"loaded:{checkpoint_path}"
            logger.log(
                f"checkpoint_load: loaded | path={checkpoint_path} | resume_update_step={update_step} | resume_global_step={global_step}"
            )
        else:
            logger.log("checkpoint_load: disabled | run started from step 0")

        optimizer.zero_grad(set_to_none=True)
        start_time = time.time()

        for epoch in range(100000):
            epoch_loss, epoch_steps = 0.0, 0
            logger.log(f"Epoch {epoch} iniciada")

            for x, y in train_loader:
                x, y = x.to(cfg["device"]), y.to(cfg["device"])
                _, t = x.size()
                state = None
                batch_loss = 0.0
                spike_rate_l1 = 0.0
                spike_rate_l2 = 0.0
                n_chunks = 0
                logits_stats = None

                for start in range(0, t, chunk_size):
                    end = min(start + chunk_size, t)
                    logits, state, (sr1, sr2) = model(x[:, start:end], state)
                    loss = criterion(logits.reshape(-1, int(cfg["vocab_size"])), y[:, start:end].reshape(-1))
                    loss = loss / int(cfg["accum_steps"])
                    loss.backward()
                    batch_loss += loss.item() * int(cfg["accum_steps"])
                    spike_rate_l1 += sr1
                    spike_rate_l2 += sr2
                    n_chunks += 1
                    with torch.no_grad():
                        logits_stats = (
                            logits.mean().item(),
                            logits.std().item(),
                            logits.min().item(),
                            logits.max().item(),
                        )

                did_update = False
                if (global_step + 1) % int(cfg["accum_steps"]) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_step += 1
                    did_update = True
                    logger.log(f"OptimizerStep | epoch={epoch} global_step={global_step+1} update_step={update_step}")

                batch_loss /= max(n_chunks, 1)
                spike_rate_l1 /= max(n_chunks, 1)
                spike_rate_l2 /= max(n_chunks, 1)
                global_step += 1
                epoch_loss += batch_loss
                epoch_steps += 1

                if global_step % 100 == 0:
                    dt = time.time() - start_time
                    speed = (100 * int(cfg["batch_size"])) / max(dt, 1e-6)
                    lr = optimizer.param_groups[0]["lr"]
                    lm, ls, lmin, lmax = logits_stats if logits_stats is not None else (0.0, 0.0, 0.0, 0.0)
                    logger.log(
                        f"epoch={epoch} global_step={global_step} update_step={update_step} "
                        f"| lattice_size={cfg['lattice_size']} | embed_dim={cfg['embed_dim']} | batch_size={cfg['batch_size']} "
                        f"| accum_steps={cfg['accum_steps']} | driver2_scale={cfg['driver2_scale']:.3f} "
                        f"| threshold={cfg['threshold']:.4f} | threshold_effective={threshold_effective:.4f} "
                        f"| checkpoint_load={checkpoint_status} | train_loss={batch_loss:.4f} | lr={lr:.2e} "
                        f"| speed={speed:.1f} seq/s | spike_l1={spike_rate_l1:.3f} | spike_l2={spike_rate_l2:.3f} "
                        f"| logits_std={ls:.3f} | logits={lm:.3f}/{ls:.3f}/{lmin:.3f}/{lmax:.3f}"
                    )
                    start_time = time.time()

                if did_update and update_step % int(cfg["val_interval_updates"]) == 0:
                    val_loss, val_ppl = evaluate(
                        model,
                        val_loader,
                        criterion,
                        chunk_size=int(cfg["chunk_size"]),
                        vocab_size=int(cfg["vocab_size"]),
                        max_batches=int(cfg["val_max_batches"]),
                    )
                    logger.log(
                        f"[VAL] epoch={epoch} global_step={global_step} update_step={update_step} "
                        f"| lattice_size={cfg['lattice_size']} | embed_dim={cfg['embed_dim']} | batch_size={cfg['batch_size']} "
                        f"| accum_steps={cfg['accum_steps']} | driver2_scale={cfg['driver2_scale']:.3f} "
                        f"| threshold={cfg['threshold']:.4f} | threshold_effective={threshold_effective:.4f} "
                        f"| checkpoint_load={checkpoint_status} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.3f}"
                    )

                if did_update and update_step % int(cfg.get("save_interval_updates", 1000)) == 0:
                    if cfg.get("save_checkpoints", False):
                        checkpoint_file = checkpoint_name_for_step(update_step)
                        save_checkpoint(model, optimizer, scheduler, cfg, update_step, global_step, checkpoint_file)
                        logger.log(f"checkpoint_save: {checkpoint_file}")

                if did_update and update_step % int(cfg["speech_interval_updates"]) == 0:
                    log_speech_checkpoints(model, logger, cfg, update_step)

                if update_step >= int(cfg["target_updates"]):
                    break

            if update_step >= int(cfg["target_updates"]):
                break

            if global_step % int(cfg["accum_steps"]) != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                logger.log(f"OptimizerStep | epoch={epoch} global_step={global_step} update_step={update_step}")

                if update_step % int(cfg["val_interval_updates"]) == 0:
                    val_loss, val_ppl = evaluate(
                        model,
                        val_loader,
                        criterion,
                        chunk_size=int(cfg["chunk_size"]),
                        vocab_size=int(cfg["vocab_size"]),
                        max_batches=int(cfg["val_max_batches"]),
                    )
                    logger.log(
                        f"[VAL] epoch={epoch} global_step={global_step} update_step={update_step} "
                        f"| lattice_size={cfg['lattice_size']} | embed_dim={cfg['embed_dim']} | batch_size={cfg['batch_size']} "
                        f"| accum_steps={cfg['accum_steps']} | driver2_scale={cfg['driver2_scale']:.3f} "
                        f"| threshold={cfg['threshold']:.4f} | threshold_effective={threshold_effective:.4f} "
                        f"| checkpoint_load={checkpoint_status} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.3f}"
                    )

                if update_step % int(cfg.get("save_interval_updates", 1000)) == 0:
                    if cfg.get("save_checkpoints", False):
                        checkpoint_file = checkpoint_name_for_step(update_step)
                        save_checkpoint(model, optimizer, scheduler, cfg, update_step, global_step, checkpoint_file)
                        logger.log(f"checkpoint_save: {checkpoint_file}")

                if update_step % int(cfg["speech_interval_updates"]) == 0:
                    log_speech_checkpoints(model, logger, cfg, update_step)

            avg = epoch_loss / max(epoch_steps, 1)
            logger.log(f"Epoch {epoch} | loss_medio={avg:.4f}")

        if cfg.get("save_checkpoints", False) and update_step > 0:
            checkpoint_file = checkpoint_name_for_step(update_step)
            save_checkpoint(model, optimizer, scheduler, cfg, update_step, global_step, checkpoint_file)
            logger.log(f"checkpoint_save: {checkpoint_file}")
        logger.log(f"FIM | stride={cfg['data_stride']} global_step={global_step} update_step={update_step}")
    finally:
        logger.close()


def smoke_test():
    cfg = apply_env_overrides(CONFIG)
    set_seed(cfg["seed"])
    logger = NullLogger()
    train_loader, val_loader = build_train_val_loaders(cfg, cfg["data_stride"], logger)
    model = MotorAI_toddler(cfg).to(cfg["device"])
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    criterion = build_criterion(cfg)

    def lr_lambda(step):
        warmup = int(cfg["warmup_steps"])
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(1, 50000 - warmup)
        progress = min(progress, 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    checkpoint_path = str(cfg.get("resume_from_checkpoint", "")).strip()
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=cfg["device"])
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    global_step = 0
    chunk_size = int(cfg["chunk_size"])

    for _ in range(100000):
        for x, y in train_loader:
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            _, t = x.size()
            state = None
            for start in range(0, t, chunk_size):
                end = min(start + chunk_size, t)
                logits, state, _ = model(x[:, start:end], state)
                loss = criterion(logits.reshape(-1, int(cfg["vocab_size"])), y[:, start:end].reshape(-1))
                loss = loss / int(cfg["accum_steps"])
                loss.backward()

            if (global_step + 1) % int(cfg["accum_steps"]) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                if update_step >= 10:
                    break

            global_step += 1

        if update_step >= 10:
            break

    val_loss, _ = evaluate(
        model,
        val_loader,
        criterion,
        chunk_size=int(cfg["chunk_size"]),
        vocab_size=int(cfg["vocab_size"]),
        max_batches=int(cfg["val_max_batches"]),
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_mb = int(torch.cuda.memory_allocated() / (1024 * 1024))
    else:
        vram_mb = 0

    safe_print(f"vram_usada={vram_mb}mb | val_loss={val_loss:.4f} | smoke_ok")


def train():
    cfg = apply_env_overrides(CONFIG)
    os.makedirs(cfg["log_dir"], exist_ok=True)
    for lattice_size in cfg["ab_runs"]:
        run_cfg = dict(cfg)
        run_cfg["seed"] = int(cfg["seed"])
        run_cfg["data_stride"] = int(cfg["data_stride"])
        run_cfg["target_updates"] = int(cfg["ab_target_updates"])
        run_cfg["val_interval_updates"] = int(cfg["val_interval_updates"])
        run_cfg["lattice_size"] = int(lattice_size)

        log_name = str(cfg["run_log_name"]).strip()
        if len(cfg["ab_runs"]) == 1 and log_name:
            if os.path.isabs(log_name) or os.path.dirname(log_name):
                log_path = log_name
            else:
                log_path = os.path.join(cfg["log_dir"], log_name)
        else:
            log_path = os.path.join(cfg["log_dir"], f"train_lattice{int(lattice_size):03d}.log")

        train_single_stride(run_cfg, run_cfg["data_stride"], run_cfg["target_updates"], log_path)


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke_test()
    else:
        train()
