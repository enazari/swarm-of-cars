import ast
t = open("results.txt").read()
rows = []
for line in t.splitlines():
    if line.startswith("#####"): rows.append(line)
    elif line.startswith("== "): name = line[3:]
    elif line.startswith("{"):
        d = ast.literal_eval(line.split("} wall")[0] + "}")
        nm = name.replace("--ckpt ckpt/", "").replace(".pt", "").replace("--n 192", "").replace("--n 96 --hard", "").replace("--safety2", "+ safety rule").strip()
        if "safety rule" not in nm and not nm.startswith("teacher"): nm += " (raw)"
        rows.append(f"| {nm} | {d['arrived_pct']:.2f}% | {d['cc']} | {d['cd']} | {d['override']} | {d['mean_t']:.2f} | {d['cars']} |")
print("\n".join(rows))
