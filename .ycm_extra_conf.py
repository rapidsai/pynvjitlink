import sysconfig
from pathlib import Path

PREFIX = sysconfig.get_config_var("prefix")
NVJITLINK_DIR = Path(__file__).parent

PREFIX_INCLUDE_DIR = Path(PREFIX, "include")
PYTHON_INCLUDE_DIR = sysconfig.get_config_var("include")
NVJITLINK_INCLUDE_DIR = Path(NVJITLINK_DIR, "include")
CUDA_INCLUDE_DIR = "/usr/local/cuda/include"

flags = [
    "--cuda-gpu-arch=sm_50",
    f"-I{PREFIX_INCLUDE_DIR}",
    f"-I{PYTHON_INCLUDE_DIR}",
    f"-I{NVJITLINK_INCLUDE_DIR}",
    f"-I{CUDA_INCLUDE_DIR}",
]


def Settings(**kwargs):
    return {"flags": flags}


if __name__ == "__main__":
    print(Settings())
