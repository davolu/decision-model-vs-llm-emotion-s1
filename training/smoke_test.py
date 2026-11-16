"""Local CPU smoke test: executes every notebook cell in order with EMOTION_S1_TINY=1 (a handful of rows, 3 training
steps). It only proves the notebook runs end to end; the numbers it prints mean nothing. Never use it for real training."""
import nbformat, os, sys, time, pathlib
os.environ["EMOTION_S1_TINY"] = "1"
nb = nbformat.read(str(pathlib.Path(__file__).parent / "emotion_s1_t4.ipynb"), 4)
work = pathlib.Path("/tmp/emotion_s1_smoke_work"); work.mkdir(exist_ok=True); os.chdir(work)
g = {"__name__": "__main__"}; T0 = time.time()
for i, c in enumerate(nb.cells):
    if c.cell_type != "code": continue
    tag = (c.metadata.get("tags") or [""])[0]; t = time.time()
    if tag == "install": print(f"-- [{i}] skip install"); continue
    src = c.source
    if src.startswith("%%writefile"):
        first, body = src.split("\n", 1); pathlib.Path(first.split()[1]).write_text(body + "\n")
        print(f"-- [{i}] wrote {first.split()[1]}"); continue
    exec(compile(src, f"<cell {i} {tag}>", "exec"), g)
    print(f"-- [{i}] ran {tag} in {time.time()-t:.1f}s", flush=True)
print(f"ALL CELLS OK in {time.time()-T0:.0f}s (TINY smoke mode, 3 training steps; metrics are meaningless)")
