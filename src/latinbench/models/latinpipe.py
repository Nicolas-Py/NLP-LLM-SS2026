"""ÚFAL LatinPipe via subprocess into the vendored repo.

LatinPipe is Keras 3 with the PyTorch backend (KERAS_BACKEND=torch must be set
in the env). It writes predictions as <test_stem>.predicted.conllu inside the
--exp directory; we move that to the canonical out_path.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
from pathlib import Path

from ..core import Model
from ..data import CHECKPOINTS_DIR, LATINPIPE_DIR


class LatinpipeModel(Model):
    name = "latinpipe"

    def __init__(
        self,
        weights: Path | None = None,
        venv_python: Path | None = None,
    ) -> None:
        self.weights = (weights or CHECKPOINTS_DIR / "latinpipe-evalatin24-240520" / "model.weights.h5").absolute()
        # .absolute() (not .resolve()) — keeps the symlink to the venv binary so
        # the venv's site-packages stays on sys.path.
        self.venv_python = (venv_python or LATINPIPE_DIR / ".venv" / "bin" / "python").absolute()

        if not self.weights.exists():
            raise FileNotFoundError(
                f"LatinPipe checkpoint not found at {self.weights}.\n"
                f"Download from https://hdl.handle.net/11234/1-5671 and extract "
                f"the contents into {self.weights.parent}."
            )
        if not self.venv_python.exists():
            raise FileNotFoundError(
                f"LatinPipe venv not found at {self.venv_python}.\n"
                f"Create it with: python3 -m venv {LATINPIPE_DIR}/.venv && "
                f"{LATINPIPE_DIR}/.venv/bin/pip install -e ."
            )

    def predict(self, test_path: Path, out_path: Path) -> None:
        print(f"[latinpipe] using checkpoint {self.weights.parent.name}")
        env = {**os.environ, "KERAS_BACKEND": "torch"}
        with tempfile.TemporaryDirectory() as td:
            exp = Path(td)
            cmd = [
                str(self.venv_python),
                "latinpipe_evalatin24.py",
                "--load", str(self.weights),
                "--exp", str(exp),
                "--test", str(test_path),
            ]
            r = subprocess.run(cmd, cwd=LATINPIPE_DIR, capture_output=True, text=True, env=env)
            if r.returncode:
                raise RuntimeError(
                    f"LatinPipe failed (exit {r.returncode}).\nSTDERR:\n{r.stderr[-3000:]}"
                )
            # LatinPipe writes <stem>.predicted.conllu inside --exp.
            produced = exp / f"{test_path.stem}.predicted.conllu"
            if not produced.exists():
                # Fall back: pick whatever .conllu got written.
                conllus = list(exp.rglob("*.conllu"))
                if not conllus:
                    raise RuntimeError(f"LatinPipe produced no .conllu files. Logs:\n{r.stdout[-2000:]}")
                produced = conllus[0]
            out_path.write_bytes(produced.read_bytes())
