"""Convert src/*.py (Jupytext percent format) to executed notebooks in notebooks/."""
import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
only = set(sys.argv[1:])
for src in sorted((ROOT / "src").glob("[0-9]*.py")):
    if only and src.stem not in only:
        continue
    nb = jupytext.reads(src.read_text(), fmt="py:percent")
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    nbformat.write(nb, ROOT / "notebooks" / f"{src.stem}.ipynb")
    print("executed", src.stem)
