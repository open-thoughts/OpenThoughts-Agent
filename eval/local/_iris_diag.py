"""Diagnostic for pydantic namespace-package shadowing on iris workers.

Run via the iris launcher in place of run_eval.py to dump the worker's
view of sys.path, site-packages contents, and pydantic's import spec.
"""
import sys
import os
import importlib.util
import pathlib
import traceback


def banner(s):
    print("=" * 8, s, "=" * 8, flush=True)


banner("env (selected)")
for k in ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT",
          "UV_LINK_MODE", "UV_NO_EDITABLE", "PWD"):
    print(f"  {k}={os.environ.get(k)!r}", flush=True)

banner("sys.path")
for p in sys.path:
    print(" ", p, flush=True)

banner("pydantic dir + init")
pd = pathlib.Path("/app/.venv/lib/python3.12/site-packages/pydantic")
print("dir exists:", pd.is_dir(), flush=True)
print("is symlink:", pd.is_symlink(), flush=True)
if pd.exists():
    try:
        print("dir resolved:", pd.resolve(), flush=True)
    except Exception as e:
        print("dir resolve err:", e, flush=True)
    try:
        print("dir lstat:", os.lstat(pd), flush=True)
    except Exception as e:
        print("dir lstat err:", e, flush=True)
    try:
        children = sorted(p.name for p in pd.iterdir())
        print(f"dir entries ({len(children)}):", children[:30], flush=True)
    except Exception as e:
        print("iterdir err:", e, flush=True)
init = pd / "__init__.py"
print("init exists:", init.exists(), flush=True)
print("init is_symlink:", init.is_symlink(), flush=True)
if init.exists():
    try:
        print("init resolved:", init.resolve(), flush=True)
    except Exception as e:
        print("init resolve err:", e, flush=True)
    try:
        text = init.read_text()
        print(f"init read OK, len={len(text)}; first 300 chars:", flush=True)
        print(text[:300], flush=True)
    except Exception as e:
        print("init read err:", type(e).__name__, e, flush=True)
elif init.is_symlink():
    try:
        target = os.readlink(init)
        print("init symlink target (broken):", target, flush=True)
    except Exception as e:
        print("readlink err:", e, flush=True)

banner("editable finders in site-packages")
sp = pathlib.Path("/app/.venv/lib/python3.12/site-packages")
if sp.is_dir():
    for f in sorted(sp.glob("__editable__*")):
        print(" ", f.name, flush=True)
        if f.suffix == ".py":
            try:
                print("    contents:", flush=True)
                txt = f.read_text()
                for line in txt.splitlines():
                    print("    " + line, flush=True)
            except Exception as e:
                print("    read err:", e, flush=True)
        elif f.suffix == ".pth":
            try:
                print("    contents:", f.read_text(), flush=True)
            except Exception as e:
                print("    read err:", e, flush=True)
    # Also dump any .pth files at top of site-packages
    print("  -- all .pth files in site-packages --", flush=True)
    for f in sorted(sp.glob("*.pth")):
        try:
            print(" ", f.name, "->", f.read_text()[:400].replace("\n", " | "), flush=True)
        except Exception as e:
            print(" ", f.name, "read err:", e, flush=True)

banner("/app top-level")
app = pathlib.Path("/app")
if app.is_dir():
    for f in sorted(app.iterdir()):
        print(" ", f.name, "DIR" if f.is_dir() else "file", flush=True)

banner("importlib spec for pydantic")
spec = importlib.util.find_spec("pydantic")
print("  spec:", spec, flush=True)
print("  spec.origin:", getattr(spec, "origin", None), flush=True)
print("  spec.submodule_search_locations:",
      getattr(spec, "submodule_search_locations", None), flush=True)
print("  spec.loader:", getattr(spec, "loader", None), flush=True)

banner("try import")
try:
    import pydantic
    print("  pydantic.__file__:", getattr(pydantic, "__file__", None), flush=True)
    print("  pydantic.__path__:", getattr(pydantic, "__path__", None), flush=True)
    from pydantic import BaseModel
    print("  BaseModel:", BaseModel, flush=True)
    print("  PYDANTIC IMPORT OK", flush=True)
except Exception as e:
    print("  IMPORT ERROR:", type(e).__name__, e, flush=True)
    traceback.print_exc()

banner("done")
