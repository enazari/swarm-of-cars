import argparse, os, time, threading, queue, math, copy
import numpy as np
import torch
import model as M

p = argparse.ArgumentParser()
p.add_argument("--kind", default="diff")
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--data", nargs="+", default=["data/r0"])
p.add_argument("--steps", type=int, default=60000)
p.add_argument("--bs", type=int, default=512)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--out", default="ckpt/diff_r0")
p.add_argument("--init", default=None)
p.add_argument("--first_w", type=float, default=1.0)
p.add_argument("--wsteps", default=None)
p.add_argument("--width", type=int, default=512)
p.add_argument("--nblk", type=int, default=6)
p.add_argument("--d", type=int, default=256)
p.add_argument("--ec", type=float, default=1.0)   # encoder channel scale
p.add_argument("--pool", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
dev = torch.device(f"cuda:{a.gpu}")
if a.wsteps:
    w = torch.ones(M.HC); ws = [float(x) for x in a.wsteps.split(',')]; w[:len(ws)] = torch.tensor(ws); M.STEP_W = w / w.mean()
elif a.first_w != 1.0:
    w = torch.ones(M.HC); w[:2] = a.first_w; M.STEP_W = w / w.mean()
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

arrs = {k: [np.load(os.path.join(d, k + ".npy"), mmap_mode="r") for d in a.data] for k in ["crop", "vec", "tok", "mask", "act"]}
lens = [len(x) for x in arrs["act"]]
N = sum(lens); NV = 30000
offs = np.cumsum([0] + lens)
print("samples", N, flush=True)
# validation = the last NV samples of the first dataset (a different set of scenes)
val_idx = np.arange(lens[0] - NV, lens[0])
train_ranges = [(offs[0], offs[0] + lens[0] - NV)] + [(offs[i], offs[i] + lens[i]) for i in range(1, len(lens))]
train_pool = np.concatenate([np.arange(s, e) for s, e in train_ranges])


def gather(idx):
    idx = np.sort(idx)
    out = {k: [] for k in arrs}
    for di in range(len(lens)):
        sel = idx[(idx >= offs[di]) & (idx < offs[di + 1])] - offs[di]
        if len(sel) == 0:
            continue
        for k in arrs:
            out[k].append(arrs[k][di][sel])
    return {k: np.concatenate(v) for k, v in out.items()}


def to_dev(b, aug):
    crop, vec, tok, mask = M.to_tensors(b["crop"], b["vec"], b["tok"], b["mask"], dev)
    act = torch.as_tensor(b["act"], device=dev).float()
    if aug:
        flip = torch.rand(crop.shape[0], device=dev) < 0.5
        crop, vec, tok, act = M.mirror(crop, vec, tok, act, flip)
    return crop, vec, tok, mask, act


q = queue.Queue(maxsize=8)
def producer():
    rng = np.random.default_rng(int(time.time()))
    while True:
        q.put(gather(rng.choice(train_pool, a.bs, replace=False)))
for _ in range(3):
    threading.Thread(target=producer, daemon=True).start()

torch.manual_seed(a.seed)
net = M.Policy(a.kind, w=a.width, nblk=a.nblk, d=a.d, ec=a.ec, pool=a.pool).to(dev)
if a.init:
    net.load_state_dict(torch.load(a.init, map_location=dev)["ema"])
ema = copy.deepcopy(net).eval()
for pp in ema.parameters():
    pp.requires_grad_(False)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))
vb = to_dev(gather(val_idx[:4096]), False)

def val(model):
    g = torch.Generator(device=dev); g.manual_seed(0)
    torch.manual_seed(0)
    with torch.no_grad():
        return model.loss(vb).item()

t0 = time.time(); run = 0.0
for step in range(1, a.steps + 1):
    b = to_dev(q.get(), True)
    loss = net.loss(b)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step(); sched.step()
    d = min(0.999, (1 + step) / (10 + step))
    with torch.no_grad():
        for pe, pn in zip(ema.parameters(), net.parameters()):
            pe.lerp_(pn, 1 - d)
    run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
    if step % 500 == 0:
        msg = f"step {step} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {step/(time.time()-t0):.1f} it/s"
        if step % 2000 == 0:
            msg += f" val {val(ema):.4f}"
        print(msg, flush=True)
    if step % 5000 == 0 or step == a.steps:
        torch.save({"ema": ema.state_dict(), "kind": a.kind, "step": step, "cfg": net.cfg}, a.out + ".pt")
print("done", flush=True)
