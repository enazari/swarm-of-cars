import sys, numpy as np, torch, torch.nn as nn
import model as M

ckpt, out = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
d = torch.load(ckpt, map_location="cpu")
net = M.Policy(d["kind"], **d.get("cfg", {})); net.load_state_dict(d["ema"]); net.eval()


class Wrap(nn.Module):
    """(crop, vec, tok, mask, noise) -> final chunk (B, 8, 2), the noisy chunk at every step xs (B, steps+1, 16) and the model's
    current guess of the clean chunk at every step xhat (B, steps, 16), all denoising steps unrolled."""
    def __init__(self, net, steps):
        super().__init__(); self.net, self.steps = net, steps

    def forward(self, crop, vec, tok, mask, noise):
        e = self.net.enc(crop, vec, tok, mask)
        x = noise
        xs, xh = [x], []
        for k in range(self.steps):
            t = torch.full((x.shape[0],), k / self.steps)
            c = self.net.cproj(torch.cat([e, self.net.temb(t)], -1))
            v = self.net.net(x, c)
            xh.append((x + (1 - k / self.steps) * v).clamp(-1, 1))
            x = x + v / self.steps
            xs.append(x)
        return x.view(-1, M.HC, M.AD).clamp(-1, 1), torch.stack(xs, 1), torch.stack(xh, 1)


w = Wrap(net, steps).eval()
B = 3
ex = (torch.randn(B, 4, 32, 32), torch.randn(B, 18), torch.randn(B, 6, 8), torch.ones(B, 6), torch.randn(B, 16))
torch.onnx.export(w, ex, out, input_names=["crop", "vec", "tok", "mask", "noise"], output_names=["chunk", "xs", "xhat"],
                  dynamic_axes={k: {0: "b"} for k in ["crop", "vec", "tok", "mask", "noise", "chunk", "xs", "xhat"]}, opset_version=17, dynamo=False)
import onnxruntime as ort
s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
r = s.run(None, {k: v.numpy() for k, v in zip(["crop", "vec", "tok", "mask", "noise"], ex)})[0]
with torch.no_grad():
    ref = w(*ex)[0].numpy()
print("onnx vs torch max abs diff", float(np.abs(r - ref).max()), "size KB", __import__("os").path.getsize(out) // 1024)
