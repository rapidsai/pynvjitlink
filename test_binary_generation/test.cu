#include <cuda_fp16.h>

extern __device__ bool __heq(__half arg1, __half arg2);

__device__
__half test_add_fp16(__half arg1, __half arg2)
{
  return __hadd(arg1, arg2);
}

__device__
bool test_cmp_fp16(__half arg1, __half arg2)
{
  return __heq(arg1, arg2);
}
