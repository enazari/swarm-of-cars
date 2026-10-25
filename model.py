import math
import torch
import torch.nn as nn
import torch.nn.functional as F

HC, AD = 8, 2          # chunk length, action dim
VEC, KT, TD = 18, 6, 8
STEP_W = None          # optional per-step loss weights (length HC)


def wmse(pred, target):
    if STEP_W is None:
        return F.mse_loss(pred, target)
    w = STEP_W.to(pred.device)
    return (((pred - target) ** 2).view(-1, HC, AD).mean(-1) * w).mean()


def sc(n, ec):
    """channel count n scaled by ec (ec=1 keeps the original widths); multiples of 4"""
    return n if ec == 1 else max(4, int(round(n * ec / 4)) * 4)


class Encoder(nn.Module):
    def __init__(self, d=256, ec=1.0, pool=0):
        super().__init__()
        self.nocnn = pool == 2
        g = 8 if ec == 1 else 4
        c1, c2, c3, c4, fa, fv, ft = [sc(n, ec) for n in (32, 64, 96, 128, 256, 128, 64)]
        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(g, o), nn.SiLU())
        if pool == 0:
            self.cnn = nn.Sequential(blk(4, c1, 1), blk(c1, c2, 2), blk(c2, c3, 2), blk(c3, c4, 2), nn.Flatten(), nn.Linear(c4 * 16, fa), nn.SiLU())
        elif pool == 1:     # tiny: the 4x4 feature map is averaged to 2x2 before the linear layer
            self.cnn = nn.Sequential(blk(4, c1, 1), blk(c1, c2, 2), blk(c2, c3, 2), blk(c3, c4, 2), nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(c4 * 4, fa), nn.SiLU())
        else:               # no map at all: route vector and neighbour tokens only
            self.cnn = None; fa = 0
        self.vec = nn.Sequential(nn.Linear(VEC, fv), nn.SiLU(), nn.Linear(fv, fv), nn.SiLU())
        self.tok = nn.Sequential(nn.Linear(TD, ft), nn.SiLU(), nn.Linear(ft, ft), nn.SiLU())
        self.tok_out = nn.Sequential(nn.Linear(2 * ft, fv), nn.SiLU())
        self.mix = nn.Sequential(nn.Linear(fa + fv + fv, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, crop, vec, tok, mask):
        a = self.cnn(crop) if self.cnn is not None else crop.new_zeros(crop.shape[0], 0)
        b = self.vec(vec)
        t = self.tok(tok)                                   # (B,K,64)
        m = mask.unsqueeze(-1)
        mean = (t * m).sum(1) / m.sum(1).clamp(min=1)
        mx = (t + (m - 1) * 1e4).max(1).values * (m.sum(1) > 0).float()
        c = self.tok_out(torch.cat([mean, mx], -1))
        return self.mix(torch.cat([a, b, c], -1))


class TimeEmb(nn.Module):
    def __init__(self, d=64, h=128):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h))

    def forward(self, t):
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        e = t[:, None] * 1000.0 * f[None]
        return self.mlp(torch.cat([e.sin(), e.cos()], -1))


class Block(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        self.n = nn.LayerNorm(w, elementwise_affine=False)
        self.film = nn.Linear(d, 2 * w)
        self.f = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, x, c):
        g, b = self.film(c).chunk(2, -1)
        return x + self.f(self.n(x) * (1 + g) + b)


class Policy(nn.Module):
    """kind='diff': flow-matching denoiser of an action chunk. kind='bc': direct regression of the chunk."""
    def __init__(self, kind="diff", w=512, nblk=6, d=256, ec=1.0, pool=0):
        super().__init__()
        self.kind = kind
        self.cfg = dict(w=w, nblk=nblk, d=d) if ec == 1 else dict(w=w, nblk=nblk, d=d, ec=ec)    # old checkpoints have no ec
        if pool: self.cfg['pool'] = pool
        self.td = sc(128, ec)
        self.enc = Encoder(d, ec, pool)
        self.temb = TimeEmb(64, self.td)
        self.inp = nn.Linear(HC * AD + (0 if kind == "bc" else 0), w)
        self.cproj = nn.Linear(d + self.td, d)
        self.blocks = nn.ModuleList([Block(w, d) for _ in range(nblk)])
        self.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, HC * AD))

    def cond(self, crop, vec, tok, mask, t=None):
        e = self.enc(crop, vec, tok, mask)
        te = self.temb(t) if t is not None else torch.zeros(e.shape[0], self.td, device=e.device)
        return self.cproj(torch.cat([e, te], -1))

    def net(self, x, c):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, c)
        return self.out(h)

    def loss(self, batch):
        crop, vec, tok, mask, act = batch
        x1 = act.flatten(1)
        if self.kind == "bc":
            c = self.cond(crop, vec, tok, mask)
            return wmse(self.net(torch.zeros_like(x1), c), x1)
        B = x1.shape[0]
        t = torch.sigmoid(torch.randn(B, device=x1.device) * 0.8)          # emphasise mid noise levels
        x0 = torch.randn_like(x1)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        c = self.cond(crop, vec, tok, mask, t)
        v = self.net(xt, c)
        return wmse(v, x1 - x0)

    @torch.no_grad()
    def sample(self, crop, vec, tok, mask, steps=8, return_traj=False, x=None):
        B = crop.shape[0]
        if self.kind == "bc":
            c = self.cond(crop, vec, tok, mask)
            return self.net(torch.zeros(B, HC * AD, device=crop.device), c).view(B, HC, AD).clamp(-1, 1)
        x = torch.randn(B, HC * AD, device=crop.device) if x is None else x
        # encode the observation once; only the time embedding changes between steps
        e = self.enc(crop, vec, tok, mask)
        traj = [x.view(B, HC, AD).clone()]
        for k in range(steps):
            t = torch.full((B,), k / steps, device=crop.device)
            c = self.cproj(torch.cat([e, self.temb(t)], -1))
            x = x + self.net(x, c) / steps
            if return_traj:
                traj.append(x.view(B, HC, AD).clone())
        out = x.view(B, HC, AD).clamp(-1, 1)
        return (out, traj) if return_traj else out


def to_tensors(crop, vec, tok, mask, device):
    """numpy arrays from the env / dataset -> normalised float tensors."""
    c = torch.as_tensor(crop, device=device)
    c = c.float() / 100.0 if c.dtype == torch.int8 else c.float()
    return c, torch.as_tensor(vec, device=device).float(), torch.as_tensor(tok, device=device).float(), torch.as_tensor(mask, device=device).float()


def mirror(crop, vec, tok, act, flip):
    """Left-right mirror of a batch where flip (bool tensor, B) is True."""
    f = flip
    crop = torch.where(f[:, None, None, None], crop.flip(2), crop)
    cs = crop.clone(); cs[:, 3] = -cs[:, 3]
    crop = torch.where(f[:, None, None, None], cs, crop)
    sign = torch.ones(VEC, device=vec.device); sign[1] = -1; sign[5] = -1
    for k in range(6):
        sign[7 + 2 * k] = -1
    vec = torch.where(f[:, None], vec * sign, vec)
    ts = torch.ones(TD, device=tok.device); ts[1] = -1; ts[3] = -1; ts[5] = -1
    tok = torch.where(f[:, None, None], tok * ts, tok)
    a = act.clone(); a[..., 1] = -a[..., 1]
    act = torch.where(f[:, None, None], a, act)
    return crop, vec, tok, act
