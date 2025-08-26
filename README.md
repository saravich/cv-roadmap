# IP Toolkit: Roadmap for Computer Vision & Image Processing

A clean, well-tested reference toolkit for image processing and classic computer vision asks.


## Highlights
- Python package (`iptk`) with typed, unit-tested modules
- Typer-based CLI for quick demos and batch jobs
- Classic CV algorithms: filtering, edges, features/matching, homography, calibration, optical flow,
  segmentation, tracking
- Lightweight dependencies: OpenCV, NumPy, Typer, Rich
- GitHub Actions CI (ruff + mypy + pytest)
- Dockerfile (CPU) + devcontainer
- MkDocs site with API/usage guides

## Quickstart
```bash
# 1) Clone & install (dev mode)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2) Run the CLI (see available commands)
iptk --help

# Example: edges
iptk edges path/to/image.jpg --out out_edges.png

# 3) Run tests & linters
pytest -q
ruff check .
mypy src
```

## Folder layout
```
ip-toolkit/
├─ src/iptk/                # Python package
├─ tests/                   # Unit tests
├─ docs/                    # MkDocs documentation
├─ cpp/                     # Minimal C++ sample (OpenCV)
├─ .github/workflows/       # CI
├─ .devcontainer/           # Codespaces/VS Code dev container
├─ Dockerfile               # CPU Docker image
├─ pyproject.toml           # Build + deps + tooling
└─ README.md
```

## License
MIT — see [LICENSE](LICENSE).
