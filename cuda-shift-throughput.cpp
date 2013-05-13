#include <algorithm>
#include <cstdio>

#include <cuda.h>

int
main()
{
  CUresult result;
  result = cuInit(0);
  CUdevice device;
  result = cuDeviceGet(&device, 0);
  CUcontext ctx;
  result = cuCtxCreate(&ctx, 0, device);
  CUmodule module;
  result = cuModuleLoad(&module, "cuda-shift-throughput.cubin");
  CUfunction kernel;
  result = cuModuleGetFunction(&kernel, module, "kernel");
  int block;
  result = cuFuncGetAttribute(&block,
                              CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                              kernel);
  int grid = 1024 * 1024;
  CUevent event[2];
  for (ptrdiff_t i = 0; i < 2; ++i) {
    result = cuEventCreate(&event[i], 0);
  }
  result = cuEventRecord(event[0], 0);
  result = cuLaunchKernel(kernel, grid, 1, 1, block, 1, 1, 0, 0, 0, 0);
  result = cuEventRecord(event[1], 0);
  result = cuEventSynchronize(event[1]);
  float time;
  result = cuEventElapsedTime(&time, event[0], event[1]);
  int gpuclock;
  result =
    cuDeviceGetAttribute(&gpuclock, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);
  int gpump;
  result =
    cuDeviceGetAttribute(&gpump, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                         device);
  std::printf("Clock: %d KHz, # of MPs: %d\n", gpuclock, gpump);
  std::printf("Elapsed Time: %f milliseconds\n", time);
  std::printf("# of Threads: %d, # of SHLs : %lld\n", block,
              1024ll * block * grid);
  std::printf("Throughput: %f\n",
              1024.0 * block * grid / ((double) gpump * gpuclock * time));
  for (ptrdiff_t i = 0; i < 2; ++i) {
    result = cuEventDestroy(event[i]);
  }
  result = cuModuleUnload(module);
  result = cuCtxDestroy(ctx);
  return 0;
}
