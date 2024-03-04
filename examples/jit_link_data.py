from numba import cuda
from pynvjitlink import patch

import numpy as np

patch.patch_numba_linker()

source = cuda.CUSource("""
typedef unsigned int uint32_t;

extern "C" __device__ int cu_add(uint32_t* result, uint32_t a, uint32_t b)
{
    *result = a + b;
    return 0;
}
""")


cu_add = cuda.declare_device("cu_add", "uint32(uint32, uint32)")


@cuda.jit(link=[source])
def kernel(result, a, b):
    result[0] = cu_add(a, b)

a = 1
b = 2
result = np.zeros(1, dtype=np.uint32)
kernel[1, 1](result, a, b)

print(f"According to a CUDA kernel, {a} + {b} = {result[0]}")
