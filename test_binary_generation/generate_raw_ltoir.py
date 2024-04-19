# Copyright (c) 2024, NVIDIA CORPORATION.

import argparse
import pathlib
import subprocess
import sys

from cuda import nvrtc

# Magic number found at the start of an LTO-IR file
LTOIR_MAGIC = 0x7F4E43ED


def check(args):
    """
    Abort and print an error message in the presence of an error result.

    Otherwise:
    - Return None if there were no more arguments,
    - Return the singular argument if there was only one further argument,
    - Return the tuple of arguments if multiple followed.
    """

    result, *args = args
    value = result.value

    if value:
        error_string = check(nvrtc.nvrtcGetErrorString(result)).decode()
        msg = f"NVRTC error, code {value}: {error_string}"
        print(msg, file=sys.stderr)
        sys.exit(1)

    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        return args


def determine_include_path():
    # Inspired by the logic in FindCUDA.cmake. We need the CUDA include path
    # because NVRTC doesn't add it by default, and we can compile a much
    # broader set of test files if the CUDA headers are available.

    # We get NVCC to tell us the location of the CUDA Toolkit.

    cmd = ["nvcc", "-v", "__dummy"]
    cp = subprocess.run(cmd, capture_output=True)

    rc = cp.returncode
    if rc != 1:
        print(f"Unexpected return code ({rc}) from `nvcc -v`. Expected 1.")
        return None

    output = cp.stderr.decode()
    lines = output.splitlines()

    top_lines = [line for line in lines if line.startswith("#$ TOP=")]
    if len(top_lines) != 1:
        print(f"Expected exactly one TOP line. Got {len(top_lines)}.")
        return None

    # Parse out the path following "TOP="

    top_dir = top_lines[0].split("TOP=")[1].strip()
    include_dir = f"{top_dir}/include"
    print(f"Using CUDA include dir {include_dir}")

    # Sanity check the include dir

    include_path = pathlib.Path(include_dir)
    if not include_path.exists():
        print("Include path appears not to exist", file=sys.stdout)
        return None

    if not include_path.is_dir():
        print("Include path appears not to be a directory", file=sys.stdout)
        return None

    cuda_h = include_path / "cuda.h"

    if not cuda_h.exists():
        print("cuda.h not found in CUDA in CUDA include location", file=sys.stderr)
        return None

    if not cuda_h.is_file():
        print("cuda.h is not a file", file=sys.stdout)
        return None

    # All is now well!

    return include_dir


def get_ltoir(source, name, arch):
    """Given a CUDA C/C++ source, compile it and return the LTO-IR."""

    program = check(nvrtc.nvrtcCreateProgram(source.encode(), name.encode(), 0, [], []))

    cuda_include_path = determine_include_path()
    if cuda_include_path is None:
        print("Error determining CUDA include path. Exiting.", file=sys.stderr)
        sys.exit(1)

    options = [
        f"--gpu-architecture={arch}",
        "-dlto",
        "-rdc",
        "true",
        f"-I{cuda_include_path}",
    ]
    options = [o.encode() for o in options]

    result = nvrtc.nvrtcCompileProgram(program, len(options), options)

    # Report compilation errors back to the user
    if result[0] == nvrtc.nvrtcResult.NVRTC_ERROR_COMPILATION:
        log_size = check(nvrtc.nvrtcGetProgramLogSize(program))
        log = b" " * log_size
        check(nvrtc.nvrtcGetProgramLog(program, log))
        print("NVRTC compilation error:\n", file=sys.stderr)
        print(log.decode(), file=sys.stderr)
        sys.exit(1)

    # Handle other errors in the standard way
    check(result)

    ltoir_size = check(nvrtc.nvrtcGetLTOIRSize(program))
    ltoir = b" " * ltoir_size
    check(nvrtc.nvrtcGetLTOIR(program, ltoir))

    # Check that the output looks like an LTO-IR container
    header = int.from_bytes(ltoir[:4], byteorder="little")
    if header != LTOIR_MAGIC:
        print(
            f"Unexpected header value 0x{header:X}.\n"
            f"Expected LTO-IR magic number 0x{LTOIR_MAGIC:X}."
            "\nExiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    return ltoir


def main(sourcepath, outputpath, arch):
    with open(sourcepath) as f:
        source = f.read()

    name = pathlib.Path(sourcepath).name
    ltoir = get_ltoir(source, name, arch)

    print(f"Writing {outputpath}...")

    with open(outputpath, "wb") as f:
        f.write(ltoir)


if __name__ == "__main__":
    description = "Compiles CUDA C/C++ to LTO-IR using NVRTC."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("sourcepath", help="path to source file")
    parser.add_argument("-o", "--output", help="path to output file", default=None)
    parser.add_argument(
        "-a",
        "--arch",
        help="compute arch to target (e.g. sm_87). " "Defaults to sm_50.",
        default="sm_50",
    )

    args = parser.parse_args()
    outputpath = args.output

    if outputpath is None:
        outputpath = pathlib.Path(args.sourcepath).with_suffix(".ltoir")

    main(args.sourcepath, outputpath, args.arch)
