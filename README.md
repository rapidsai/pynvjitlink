> [!IMPORTANT]
> The final release of this project was `v0.7.0`.
> Similar functionality is available in `cuda.core`, and for `numba-cuda>=0.16`,
> `numba-cuda` automatically detecst and enable nvjitlink when needed with no explicit configuration.
> See https://docs.rapids.ai/notices/rsn0052/ for details.

# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;pynvjitlink</div>

The [RAPIDS](https://rapids.ai) pynvjitlink library provides a Python binding for the
[nvJitLink library](https://docs.nvidia.com/cuda/nvjitlink/index.html).

## Installation with pip

```shell
pip install pynvjitlink-cu12
```

## Installation from source

Install with either:

```shell
python -m pip install .
```

or

```shell
python -m pip install -e .
```

for an editable install.

## Installation with Conda

```shell
conda install -c rapidsai pynvjitlink
```

## Contributing Guide

Review the
[CONTRIBUTING.md](https://github.com/rapidsai/pynvjitlink/blob/main/CONTRIBUTING.md)
file for information on how to contribute code and issues to the project.
