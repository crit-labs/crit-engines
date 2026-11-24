"""crit run build's entry: runs build.cmd (MSVC's environment, then gguf-projections' build_kernels.py) by its
absolute path, since a Windows process start does not find a script by a relative name."""
import os
import subprocess
import sys
from pathlib import Path

script = Path(__file__).resolve().parent / "build.cmd"
# build.cmd runs `python`: this interpreter (crit's, with torch), whatever PATH held before
env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
sys.exit(subprocess.run(["cmd", "/c", str(script)], cwd=script.parent, env=env).returncode)
