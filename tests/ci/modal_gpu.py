import pathlib
import subprocess

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REMOTE_REPO = "/root/rushlite"

app = modal.App("rushlite-ci-gpu")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11"
    )
    .apt_install(
        "build-essential",
        "cmake",
        "git",
        "python3-venv",
        "python3-pip",
        "pybind11-dev",
        "python3-pybind11",
        "libboost-all-dev",
    )
    .pip_install("uv")
    .add_local_dir(str(REPO_ROOT), REMOTE_REPO, copy=True)
)


def _run(cmd: str) -> None:
    print(f"\n+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=REMOTE_REPO, check=True)


@app.function(gpu="T4", image=image, timeout=60 * 60)
def run_gpu_ci() -> None:
    _run("nvidia-smi")
    _run(
        "cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug "
        "-DLMP_ENABLE_CUDA=ON -DLMP_ENABLE_TEST=OFF "
        "-DLMP_ENABLE_COVERAGE=OFF -DLMP_ENABLE_BENCH=OFF"
    )
    _run("cmake --build build --config Debug --parallel")
    _run("uv lock")
    _run("uv sync --extra cu128")
    _run("SKBUILD_CMAKE_DEFINE=LMP_ENABLE_CUDA=ON uv pip install .")
    _run(
        "uv run python -c \""
        "import pylamp; "
        "pylamp.Tensor([[0.0]], requires_grad=False, "
        "device=pylamp.device.cuda, dtype=pylamp.dtype.float64); "
        "print('pylamp CUDA device OK')\""
    )
    _run("uv run pytest tests/stress/pytorch_stress_test.py -v")


@app.local_entrypoint()
def main() -> None:
    run_gpu_ci.remote()
