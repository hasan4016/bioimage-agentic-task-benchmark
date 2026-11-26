import shutil, sys
from pathlib import Path
from importlib.metadata import version

ROOT = Path("release")
if ROOT.exists():
    shutil.rmtree(ROOT)
(ROOT / "task").mkdir(parents=True)
(ROOT / "hidden").mkdir()

# agent-visible
shutil.copytree("task_build/data", ROOT / "task" / "data")
shutil.copy("task_build/instructions.md", ROOT / "task" / "instructions.md")

# hidden: answer key, expert solution (self-contained), grader
shutil.copy("task_build/ground_truth/truth.json", ROOT / "hidden" / "truth.json")

ref = Path("scripts/ref_method.py").read_text()
sol = Path("task_build/solution.py").read_text()
sol = sol.replace('sys.path.insert(0, "scripts")\n', "")
sol = sol.replace("from ref_method import segment_ref, contrast   # (will be inlined when packaging)\n", "")
sol = sol.replace('Path("task_build/data")', 'Path("data")')
sol = sol.replace('Path("task_build/output_ref")', 'Path("output")')
assert "ref_method" not in sol and "task_build" not in sol, "packaging replace failed"
(ROOT / "hidden" / "solution.py").write_text(ref + "\n\n" + sol)

g = Path("scripts/grader.py").read_text()
assert "task_build/ground_truth/truth.json" in g
(ROOT / "hidden" / "grader.py").write_text(
    g.replace('"task_build/ground_truth/truth.json"',
              'str(Path(__file__).resolve().parent / "truth.json")'))

pk = ["numpy", "pandas", "scipy", "scikit-image", "tifffile"]
pins = " ".join(f"{p}=={version(p)}" for p in pk)
py = f"{sys.version_info.major}.{sys.version_info.minor}"
(ROOT / "Dockerfile").write_text(
    f"FROM python:{py}-slim\n"
    f"RUN pip install --no-cache-dir {pins}\n"
    "WORKDIR /workspace\n"
    "COPY task/ /workspace/\n")
print("packaged ->", ROOT, "| python", py, "|", pins)
