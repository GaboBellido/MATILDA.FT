// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"

// NOTE: The original d_reduce_float kernel and reduce_device_float() wrapper
// that lived here were removed because they were non-functional:
//
//   1. `tot_sum` was passed by value, so all writes inside the kernel
//      (tot_sum = 0.f, atomicAdd(&tot_sum, ...)) modified a thread-local copy
//      that was never visible to the caller.
//
//   2. `cudaFree(&d_sum)` passed the stack address of a pointer variable
//      rather than the pointer itself, corrupting memory on every call.
//
//   3. The only call site (potential.cu:223) was already commented out.
//
// Use thrust::reduce() or thrust::transform_reduce() from <thrust/reduce.h>
// for reliable device-side reductions.
