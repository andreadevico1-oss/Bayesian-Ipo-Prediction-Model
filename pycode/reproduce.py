"""Execute the research notebook from a fresh kernel."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parent.parent
    notebook_path = root / "final_output.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(notebook, timeout=180, kernel_name="python3",
                            resources={"metadata": {"path": str(root)}})
    client.execute()
    nbformat.write(notebook, notebook_path)
    figures = sum("image/png" in output.get("data", {})
                  for cell in notebook.cells if cell.cell_type == "code"
                  for output in cell.outputs)
    print(f"Notebook reproduced successfully: {figures} figures.")


if __name__ == "__main__":
    main()
