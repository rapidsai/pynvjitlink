#!/usr/bin/env python
# Copyright (c) 2024, NVIDIA CORPORATION

from pynvjitlink import patch

patch.patch_numba_linker()

if __name__ == "__main__":
    import sys

    from numba.testing._runtests import _main

    sys.exit(0 if _main(sys.argv) else 1)
