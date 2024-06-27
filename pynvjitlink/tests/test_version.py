# Copyright (c) 2024, NVIDIA CORPORATION.

import pynvjitlink


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(pynvjitlink.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(pynvjitlink.__version__, str)
    assert len(pynvjitlink.__version__) > 0


def test_nvjitlink_version():
    nvjitlink_version = pynvjitlink.nvjitlink_version()
    assert len(nvjitlink_version) == 2
    assert nvjitlink_version[0] >= 12
    assert nvjitlink_version[1] >= 0
