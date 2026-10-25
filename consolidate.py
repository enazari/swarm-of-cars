import sys, glob, os, numpy as np
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
sizes = []
for f in files:
    with np.load(f) as z:
        sizes.append(len(z["act"]))
N = sum(sizes); print(d, "shards", len(files), "samples", N)
spec = {"crop": ((4, 32, 32), np.int8), "vec": ((18,), np.float32), "tok": ((6, 8), np.float16), "mask": ((6,), np.uint8), "act": ((8, 2), np.float16)}
mm = {k: np.lib.format.open_memmap(os.path.join(d, k + ".npy"), mode="w+", dtype=dt, shape=(N,) + sh) for k, (sh, dt) in spec.items()}
o = 0
for f, n in zip(files, sizes):
    with np.load(f) as z:
        for k in spec:
            mm[k][o:o + n] = z[k]
    o += n
for v in mm.values():
    v.flush()
print("done")
