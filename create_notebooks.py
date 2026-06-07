"""Create Jupyter notebooks for the project."""
import nbformat as nbf
import os

os.makedirs("notebooks", exist_ok=True)


def code(src): return nbf.v4.new_code_cell(src)
def md(src):   return nbf.v4.new_markdown_cell(src)


def make_nb(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def save(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  {path}")


# ── Notebook 1 ──────────────────────────────────
if __name__ == "__main__":
    print("Notebooks module loaded")
