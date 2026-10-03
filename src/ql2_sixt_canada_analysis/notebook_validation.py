"""Execute a notebook copy from a clean kernel and check its execution order.

Used by the automated tests and by local validation. The tracked notebook is
only read; execution happens on an in-memory copy with a fresh kernel, and
the executed copy is written only where the caller asks (e.g. a pytest
``tmp_path``) or not at all. Importing this module performs no I/O.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbformat import NotebookNode

__all__ = [
    "NOTEBOOK_FORMAT_MAJOR",
    "NotebookExecutionError",
    "NotebookExecutionResult",
    "NotebookOrderError",
    "check_execution_order",
    "execute_notebook_copy",
    "read_notebook",
]

NOTEBOOK_FORMAT_MAJOR = 4


class NotebookExecutionError(Exception):
    """A cell raised during execution. The nbclient error is the cause.

    Only the zero-based index of the failing code cell is reported; the
    cell's output (which could contain data) is left on the cause.
    """

    def __init__(self, notebook: Path, cell_index: int) -> None:
        super().__init__(f"{notebook.name}: code cell {cell_index} raised during execution.")
        self.notebook = notebook
        self.cell_index = cell_index


class NotebookOrderError(Exception):
    """Executed code cells do not have consecutive, in-order execution counts."""


@dataclass(frozen=True, slots=True)
class NotebookExecutionResult:
    """Outcome of :func:`execute_notebook_copy`."""

    source: Path
    executed: NotebookNode
    execution_counts: tuple[int, ...]


def read_notebook(path: str | os.PathLike[str]) -> NotebookNode:
    """Read and validate a notebook file without modifying it."""
    notebook = nbformat.read(Path(path), as_version=NOTEBOOK_FORMAT_MAJOR)
    nbformat.validate(notebook)
    return notebook


def check_execution_order(notebook: NotebookNode) -> tuple[int, ...]:
    """Return the code-cell execution counts, requiring them to be 1, 2, 3, ...

    Raises:
        NotebookOrderError: A code cell was not executed, or the counts are
            not strictly consecutive in notebook order.
    """
    counts = []
    for index, cell in enumerate(c for c in notebook.cells if c.cell_type == "code"):
        count = cell.get("execution_count")
        if count is None:
            raise NotebookOrderError(f"code cell {index} has no execution count")
        counts.append(int(count))
    if counts != list(range(1, len(counts) + 1)):
        raise NotebookOrderError("execution counts are not consecutive in notebook order")
    return tuple(counts)


def execute_notebook_copy(
    source: str | os.PathLike[str],
    *,
    workdir: str | os.PathLike[str],
    env: Mapping[str, str] | None = None,
    timeout_seconds: int = 120,
    kernel_name: str | None = None,
    output_path: str | os.PathLike[str] | None = None,
) -> NotebookExecutionResult:
    """Execute a fresh copy of ``source`` from a clean kernel, top to bottom.

    Args:
        source: Tracked notebook to read (never written).
        workdir: Working directory for the kernel; use a temporary directory
            so any stray writes stay out of the repository.
        env: Extra environment variables for the kernel process, e.g. the
            raw-directory override from :mod:`ql2_sixt_canada_analysis.paths`.
        timeout_seconds: Per-cell timeout.
        kernel_name: Kernel to start; defaults to the notebook's kernelspec.
        output_path: Where to write the executed copy; omitted means the
            executed notebook is kept in memory only.

    Raises:
        NotebookExecutionError: A cell raised; ``__cause__`` has the details.
        NotebookOrderError: Execution counts are not consecutive afterwards.
    """
    source_path = Path(source)
    notebook = read_notebook(source_path)
    kernel_env = {**os.environ, **(env or {})}
    client = NotebookClient(
        notebook,
        timeout=timeout_seconds,
        kernel_name=kernel_name or notebook.metadata.get("kernelspec", {}).get("name", "python3"),
        allow_errors=False,
        record_timing=False,
        resources={"metadata": {"path": str(workdir)}},
    )
    try:
        # ``env`` is forwarded to the kernel manager so the kernel process
        # (not this process) sees the override variables.
        client.execute(env=kernel_env)
    except CellExecutionError as exc:
        index = next(
            (i for i, c in enumerate(cc for cc in notebook.cells if cc.cell_type == "code")
             if any(o.get("output_type") == "error" for o in c.get("outputs", []))),
            -1,
        )
        raise NotebookExecutionError(source_path, index) from exc
    counts = check_execution_order(notebook)
    if output_path is not None:
        nbformat.write(notebook, Path(output_path))
    return NotebookExecutionResult(source=source_path, executed=notebook, execution_counts=counts)
