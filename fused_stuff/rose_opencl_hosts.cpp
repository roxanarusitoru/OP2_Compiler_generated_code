#include <CL/cl.h>
#include "op_opencl_rt_support.h"
extern float alpha;
extern float cfl;
extern float eps;
extern float gam;
extern float gm1;
extern float mach;
extern float qinf[4UL];
int threadsPerBlockSize_bres_calc = 512;
int setPartitionSize_bres_calc = 512;
int threadsPerBlockSize_fusedOne = 512;
int setPartitionSize_fusedOne = 512;
int threadsPerBlockSize_fusedTwo = 512;
int setPartitionSize_fusedTwo = 512;
int threadsPerBlockSize_res_calc = 512;
int setPartitionSize_res_calc = 512;
#ifdef OP_WARPSIZE_0
#define OP_WARPSIZE OP_WARPSIZE_0
#endif
int threadsPerBlockSize_update = 512;
void bres_calc_host(const char *userSubroutine,op_set set,op_arg opDat1,op_arg opDat2,op_arg opDat3,op_arg opDat4,op_arg opDat5,op_arg opDat6)
{
#ifdef OP_BLOCK_SIZE_0
  threadsPerBlockSize_bres_calc = OP_BLOCK_SIZE_0;
#endif
#ifdef OP_PART_SIZE_0
  setPartitionSize_bres_calc = OP_PART_SIZE_0;
#endif
  size_t blocksPerGrid;
  size_t threadsPerBlock;
  size_t totalThreadNumber;
  size_t dynamicSharedMemorySize;
  cl_int errorCode;
  cl_event event;
  cl_kernel kernelPointer;
  int i3;
  op_arg opDatArray[6];
  int indirectionDescriptorArray[6];
  op_plan *planRet;
  int blockOffset;
  opDatArray[0] = opDat1;
  opDatArray[1] = opDat2;
  opDatArray[2] = opDat3;
  opDatArray[3] = opDat4;
  opDatArray[4] = opDat5;
  opDatArray[5] = opDat6;
  indirectionDescriptorArray[0] = 0;
  indirectionDescriptorArray[1] = 0;
  indirectionDescriptorArray[2] = 1;
  indirectionDescriptorArray[3] = 2;
  indirectionDescriptorArray[4] = 3;
  indirectionDescriptorArray[5] = -1;
  planRet = op_plan_get(userSubroutine,set,setPartitionSize_bres_calc,6,opDatArray,4,indirectionDescriptorArray);
  cl_mem gm1_d;
  gm1_d = op_allocate_constant(&gm1,sizeof(float ));
  cl_mem qinf_d;
  qinf_d = op_allocate_constant(&qinf,4 * sizeof(float));
  cl_mem eps_d;
  eps_d = op_allocate_constant(&eps,sizeof(float ));
  blockOffset = 0;
  double cpu_t1;
  double cpu_t2;
  double wall_t1;
op_timers(&cpu_t1, &wall_t1);
  double wall_t2;
  for (i3 = 0; i3 < planRet -> ncolors; ++i3) {
    blocksPerGrid = planRet -> ncolblk[i3];
    dynamicSharedMemorySize = planRet -> nshared;
    threadsPerBlock = threadsPerBlockSize_bres_calc;
    totalThreadNumber = threadsPerBlock * blocksPerGrid;
    kernelPointer = getKernel("bres_calc_kernel");
    errorCode = clSetKernelArg(kernelPointer,0,sizeof(cl_mem ),&opDat1.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,1,sizeof(cl_mem ),&opDat3.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,2,sizeof(cl_mem ),&opDat4.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,3,sizeof(cl_mem ),&opDat5.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,4,sizeof(cl_mem ),&opDat6.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,5,sizeof(cl_mem ),&planRet -> ind_maps[0]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,6,sizeof(cl_mem ),&planRet -> ind_maps[1]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,7,sizeof(cl_mem ),&planRet -> ind_maps[2]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,8,sizeof(cl_mem ),&planRet -> ind_maps[3]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,9,sizeof(cl_mem ),&planRet -> loc_maps[0]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,10,sizeof(cl_mem ),&planRet -> loc_maps[1]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,11,sizeof(cl_mem ),&planRet -> loc_maps[2]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,12,sizeof(cl_mem ),&planRet -> loc_maps[3]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,13,sizeof(cl_mem ),&planRet -> loc_maps[4]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,14,sizeof(cl_mem ),&planRet -> ind_sizes);
    errorCode = errorCode | clSetKernelArg(kernelPointer,15,sizeof(cl_mem ),&planRet -> ind_offs);
    errorCode = errorCode | clSetKernelArg(kernelPointer,16,sizeof(cl_mem ),&planRet -> blkmap);
    errorCode = errorCode | clSetKernelArg(kernelPointer,17,sizeof(cl_mem ),&planRet -> offset);
    errorCode = errorCode | clSetKernelArg(kernelPointer,18,sizeof(cl_mem ),&planRet -> nelems);
    errorCode = errorCode | clSetKernelArg(kernelPointer,19,sizeof(cl_mem ),&planRet -> nthrcol);
    errorCode = errorCode | clSetKernelArg(kernelPointer,20,sizeof(cl_mem ),&planRet -> thrcol);
    errorCode = errorCode | clSetKernelArg(kernelPointer,21,sizeof(int ),&blockOffset);
    errorCode = errorCode | clSetKernelArg(kernelPointer,22,dynamicSharedMemorySize,NULL);
    errorCode = errorCode | clSetKernelArg(kernelPointer,23,sizeof(cl_mem ),&gm1_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,24,sizeof(cl_mem ),&qinf_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,25,sizeof(cl_mem ),&eps_d);
    assert_m(errorCode == CL_SUCCESS,"Error setting OpenCL kernel arguments");
    errorCode = clEnqueueNDRangeKernel(cqCommandQueue,kernelPointer,1,NULL,&totalThreadNumber,&threadsPerBlock,0,NULL,&event);
    assert_m(errorCode == CL_SUCCESS,"Error executing OpenCL kernel");
    errorCode = clFinish(cqCommandQueue);
    assert_m(errorCode == CL_SUCCESS,"Error completing device command queue");
    blockOffset += blocksPerGrid;
  }
op_timers(&cpu_t2, &wall_t2);
op_timing_realloc(0);
  OP_kernels[0].name = userSubroutine;
  OP_kernels[0].count = OP_kernels[0].count + 1;
  OP_kernels[0].time = OP_kernels[0].time + (wall_t2 - wall_t1);
  OP_kernels[0].transfer = OP_kernels[0].transfer + planRet -> transfer;
  OP_kernels[0].transfer = OP_kernels[0].transfer + planRet -> transfer2;
}
void fusedOne_host(const char *userSubroutine,op_set set,op_arg opDat1,op_arg opDat2,op_arg opDat3,op_arg opDat4,op_arg opDat5,op_arg opDat6,op_arg opDat7)
{
#ifdef OP_BLOCK_SIZE_1
  threadsPerBlockSize_fusedOne = OP_BLOCK_SIZE_1;
#endif
#ifdef OP_PART_SIZE_1
  setPartitionSize_fusedOne = OP_PART_SIZE_1;
#endif
  size_t blocksPerGrid;
  size_t threadsPerBlock;
  size_t totalThreadNumber;
  size_t dynamicSharedMemorySize;
  cl_int errorCode;
  cl_event event;
  cl_kernel kernelPointer;
  int i3;
  op_arg opDatArray[7];
  int indirectionDescriptorArray[7];
  op_plan *planRet;
  int blockOffset;
  opDatArray[0] = opDat1;
  opDatArray[1] = opDat2;
  opDatArray[2] = opDat3;
  opDatArray[3] = opDat4;
  opDatArray[4] = opDat5;
  opDatArray[5] = opDat6;
  opDatArray[6] = opDat7;
  indirectionDescriptorArray[0] = -1;
  indirectionDescriptorArray[1] = -1;
  indirectionDescriptorArray[2] = 0;
  indirectionDescriptorArray[3] = 0;
  indirectionDescriptorArray[4] = 0;
  indirectionDescriptorArray[5] = 0;
  indirectionDescriptorArray[6] = -1;
  planRet = op_plan_get(userSubroutine,set,setPartitionSize_fusedOne,7,opDatArray,1,indirectionDescriptorArray);
  cl_mem gam_d;
  gam_d = op_allocate_constant(&gam,sizeof(float ));
  cl_mem gm1_d;
  gm1_d = op_allocate_constant(&gm1,sizeof(float ));
  cl_mem cfl_d;
  cfl_d = op_allocate_constant(&cfl,sizeof(float ));
  blockOffset = 0;
  double cpu_t1;
  double cpu_t2;
  double wall_t1;
op_timers(&cpu_t1, &wall_t1);
  double wall_t2;
  for (i3 = 0; i3 < planRet -> ncolors; ++i3) {
    blocksPerGrid = planRet -> ncolblk[i3];
    dynamicSharedMemorySize = planRet -> nshared;
    threadsPerBlock = threadsPerBlockSize_fusedOne;
    totalThreadNumber = threadsPerBlock * blocksPerGrid;
    kernelPointer = getKernel("fusedOne_kernel");
    errorCode = clSetKernelArg(kernelPointer,0,sizeof(cl_mem ),&opDat1.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,1,sizeof(cl_mem ),&opDat2.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,2,sizeof(cl_mem ),&opDat3.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,3,sizeof(cl_mem ),&opDat7.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,4,sizeof(cl_mem ),&planRet -> ind_maps[0]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,5,sizeof(cl_mem ),&planRet -> loc_maps[2]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,6,sizeof(cl_mem ),&planRet -> loc_maps[3]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,7,sizeof(cl_mem ),&planRet -> loc_maps[4]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,8,sizeof(cl_mem ),&planRet -> loc_maps[5]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,9,sizeof(cl_mem ),&planRet -> ind_sizes);
    errorCode = errorCode | clSetKernelArg(kernelPointer,10,sizeof(cl_mem ),&planRet -> ind_offs);
    errorCode = errorCode | clSetKernelArg(kernelPointer,11,sizeof(cl_mem ),&planRet -> blkmap);
    errorCode = errorCode | clSetKernelArg(kernelPointer,12,sizeof(cl_mem ),&planRet -> offset);
    errorCode = errorCode | clSetKernelArg(kernelPointer,13,sizeof(cl_mem ),&planRet -> nelems);
    errorCode = errorCode | clSetKernelArg(kernelPointer,14,sizeof(cl_mem ),&planRet -> nthrcol);
    errorCode = errorCode | clSetKernelArg(kernelPointer,15,sizeof(cl_mem ),&planRet -> thrcol);
    errorCode = errorCode | clSetKernelArg(kernelPointer,16,sizeof(int ),&blockOffset);
    errorCode = errorCode | clSetKernelArg(kernelPointer,17,dynamicSharedMemorySize,NULL);
    errorCode = errorCode | clSetKernelArg(kernelPointer,18,sizeof(cl_mem ),&gam_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,19,sizeof(cl_mem ),&gm1_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,20,sizeof(cl_mem ),&cfl_d);
    assert_m(errorCode == CL_SUCCESS,"Error setting OpenCL kernel arguments");
    errorCode = clEnqueueNDRangeKernel(cqCommandQueue,kernelPointer,1,NULL,&totalThreadNumber,&threadsPerBlock,0,NULL,&event);
    assert_m(errorCode == CL_SUCCESS,"Error executing OpenCL kernel");
    errorCode = clFinish(cqCommandQueue);
    assert_m(errorCode == CL_SUCCESS,"Error completing device command queue");
    blockOffset += blocksPerGrid;
  }
op_timers(&cpu_t2, &wall_t2);
op_timing_realloc(1);
  OP_kernels[1].name = userSubroutine;
  OP_kernels[1].count = OP_kernels[1].count + 1;
  OP_kernels[1].time = OP_kernels[1].time + (wall_t2 - wall_t1);
  OP_kernels[1].transfer = OP_kernels[1].transfer + planRet -> transfer;
  OP_kernels[1].transfer = OP_kernels[1].transfer + planRet -> transfer2;
}
void fusedTwo_host(const char *userSubroutine,op_set set,op_arg opDat1,op_arg opDat2,op_arg opDat3,op_arg opDat4,op_arg opDat5,op_arg opDat6,op_arg opDat7,op_arg opDat8,op_arg opDat9)
{
#ifdef OP_BLOCK_SIZE_2
  threadsPerBlockSize_fusedTwo = OP_BLOCK_SIZE_2;
#endif
#ifdef OP_PART_SIZE_2
  setPartitionSize_fusedTwo = OP_PART_SIZE_2;
#endif
  size_t blocksPerGrid;
  size_t threadsPerBlock;
  size_t totalThreadNumber;
  size_t dynamicSharedMemorySize;
  cl_int errorCode;
  cl_event event;
  cl_kernel kernelPointer;
  int i3;
  op_arg opDatArray[9];
  int indirectionDescriptorArray[9];
  op_plan *planRet;
  int blockOffset;
  int i1;
  int i2;
  int reductionBytes;
  int reductionSharedMemorySize;
  float *reductionArrayHost5;
  reductionBytes = 0;
  reductionSharedMemorySize = 0;
  reductionArrayHost5 = ((float *)opDat5.data);
  reductionBytes += ROUND_UP(blocksPerGrid * sizeof(float ) * 1);
  reductionSharedMemorySize = MAX(reductionSharedMemorySize,sizeof(float ));
  reallocReductArrays(reductionBytes);
  reductionBytes = 0;
  opDat5.data = OP_reduct_h + reductionBytes;
  opDat5.data_d = ((char *)OP_reduct_d) + reductionBytes;
  for (i1 = 0; i1 < blocksPerGrid; ++i1) {
    for (i2 = 0; i2 < 1; ++i2) {
      ((float *)opDat5.data)[i2 + i1 * 1] = 0.00000F;
    }
  }
  reductionBytes += ROUND_UP(blocksPerGrid * sizeof(float ) * 1);
  mvReductArraysToDevice(reductionBytes);
  opDatArray[0] = opDat1;
  opDatArray[1] = opDat2;
  opDatArray[2] = opDat3;
  opDatArray[3] = opDat4;
  opDatArray[4] = opDat5;
  opDatArray[5] = opDat6;
  opDatArray[6] = opDat7;
  opDatArray[7] = opDat8;
  opDatArray[8] = opDat9;
  indirectionDescriptorArray[0] = -1;
  indirectionDescriptorArray[1] = -1;
  indirectionDescriptorArray[2] = -1;
  indirectionDescriptorArray[3] = -1;
  indirectionDescriptorArray[4] = -1;
  indirectionDescriptorArray[5] = 0;
  indirectionDescriptorArray[6] = 0;
  indirectionDescriptorArray[7] = 0;
  indirectionDescriptorArray[8] = 0;
  planRet = op_plan_get(userSubroutine,set,setPartitionSize_fusedTwo,9,opDatArray,1,indirectionDescriptorArray);
  cl_mem gam_d;
  gam_d = op_allocate_constant(&gam,sizeof(float ));
  cl_mem gm1_d;
  gm1_d = op_allocate_constant(&gm1,sizeof(float ));
  cl_mem cfl_d;
  cfl_d = op_allocate_constant(&cfl,sizeof(float ));
  blockOffset = 0;
  double cpu_t1;
  double cpu_t2;
  double wall_t1;
op_timers(&cpu_t1, &wall_t1);
  double wall_t2;
  for (i3 = 0; i3 < planRet -> ncolors; ++i3) {
    blocksPerGrid = planRet -> ncolblk[i3];
    dynamicSharedMemorySize = planRet -> nshared;
    threadsPerBlock = threadsPerBlockSize_fusedTwo;
    totalThreadNumber = threadsPerBlock * blocksPerGrid;
    kernelPointer = getKernel("fusedTwo_kernel");
    errorCode = clSetKernelArg(kernelPointer,0,sizeof(cl_mem ),&opDat1.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,1,sizeof(cl_mem ),&opDat2.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,2,sizeof(cl_mem ),&opDat3.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,3,sizeof(cl_mem ),&opDat4.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,4,sizeof(cl_mem ),&opDat5.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,5,sizeof(cl_mem ),&opDat6.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,6,sizeof(cl_mem ),&planRet -> ind_maps[0]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,7,sizeof(cl_mem ),&planRet -> loc_maps[5]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,8,sizeof(cl_mem ),&planRet -> loc_maps[6]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,9,sizeof(cl_mem ),&planRet -> loc_maps[7]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,10,sizeof(cl_mem ),&planRet -> loc_maps[8]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,11,sizeof(cl_mem ),&planRet -> ind_sizes);
    errorCode = errorCode | clSetKernelArg(kernelPointer,12,sizeof(cl_mem ),&planRet -> ind_offs);
    errorCode = errorCode | clSetKernelArg(kernelPointer,13,sizeof(cl_mem ),&planRet -> blkmap);
    errorCode = errorCode | clSetKernelArg(kernelPointer,14,sizeof(cl_mem ),&planRet -> offset);
    errorCode = errorCode | clSetKernelArg(kernelPointer,15,sizeof(cl_mem ),&planRet -> nelems);
    errorCode = errorCode | clSetKernelArg(kernelPointer,16,sizeof(cl_mem ),&planRet -> nthrcol);
    errorCode = errorCode | clSetKernelArg(kernelPointer,17,sizeof(cl_mem ),&planRet -> thrcol);
    errorCode = errorCode | clSetKernelArg(kernelPointer,18,sizeof(int ),&blockOffset);
    errorCode = errorCode | clSetKernelArg(kernelPointer,19,dynamicSharedMemorySize,NULL);
    errorCode = errorCode | clSetKernelArg(kernelPointer,20,sizeof(cl_mem ),&gam_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,21,sizeof(cl_mem ),&gm1_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,22,sizeof(cl_mem ),&cfl_d);
    assert_m(errorCode == CL_SUCCESS,"Error setting OpenCL kernel arguments");
    errorCode = clEnqueueNDRangeKernel(cqCommandQueue,kernelPointer,1,NULL,&totalThreadNumber,&threadsPerBlock,0,NULL,&event);
    assert_m(errorCode == CL_SUCCESS,"Error executing OpenCL kernel");
    errorCode = clFinish(cqCommandQueue);
    assert_m(errorCode == CL_SUCCESS,"Error completing device command queue");
    blockOffset += blocksPerGrid;
  }
op_timers(&cpu_t2, &wall_t2);
op_timing_realloc(2);
  OP_kernels[2].name = userSubroutine;
  OP_kernels[2].count = OP_kernels[2].count + 1;
  OP_kernels[2].time = OP_kernels[2].time + (wall_t2 - wall_t1);
  OP_kernels[2].transfer = OP_kernels[2].transfer + planRet -> transfer;
  OP_kernels[2].transfer = OP_kernels[2].transfer + planRet -> transfer2;
  mvReductArraysToHost(reductionBytes);
  for (i1 = 0; i1 < blocksPerGrid; ++i1) {
    for (i2 = 0; i2 < 1; ++i2) {
      reductionArrayHost5[i2] += ((float *)opDat5.data)[i2 + i1 * 1];
    }
  }
}
void res_calc_host(const char *userSubroutine,op_set set,op_arg opDat1,op_arg opDat2,op_arg opDat3,op_arg opDat4,op_arg opDat5,op_arg opDat6,op_arg opDat7,op_arg opDat8)
{
#ifdef OP_BLOCK_SIZE_3
  threadsPerBlockSize_res_calc = OP_BLOCK_SIZE_3;
#endif
#ifdef OP_PART_SIZE_3
  setPartitionSize_res_calc = OP_PART_SIZE_3;
#endif
  size_t blocksPerGrid;
  size_t threadsPerBlock;
  size_t totalThreadNumber;
  size_t dynamicSharedMemorySize;
  cl_int errorCode;
  cl_event event;
  cl_kernel kernelPointer;
  int i3;
  op_arg opDatArray[8];
  int indirectionDescriptorArray[8];
  op_plan *planRet;
  int blockOffset;
  opDatArray[0] = opDat1;
  opDatArray[1] = opDat2;
  opDatArray[2] = opDat3;
  opDatArray[3] = opDat4;
  opDatArray[4] = opDat5;
  opDatArray[5] = opDat6;
  opDatArray[6] = opDat7;
  opDatArray[7] = opDat8;
  indirectionDescriptorArray[0] = 0;
  indirectionDescriptorArray[1] = 0;
  indirectionDescriptorArray[2] = 1;
  indirectionDescriptorArray[3] = 1;
  indirectionDescriptorArray[4] = 2;
  indirectionDescriptorArray[5] = 2;
  indirectionDescriptorArray[6] = 3;
  indirectionDescriptorArray[7] = 3;
  planRet = op_plan_get(userSubroutine,set,setPartitionSize_res_calc,8,opDatArray,4,indirectionDescriptorArray);
  cl_mem gm1_d;
  gm1_d = op_allocate_constant(&gm1,sizeof(float ));
  cl_mem eps_d;
  eps_d = op_allocate_constant(&eps,sizeof(float ));
  blockOffset = 0;
  double cpu_t1;
  double cpu_t2;
  double wall_t1;
op_timers(&cpu_t1, &wall_t1);
  double wall_t2;
  for (i3 = 0; i3 < planRet -> ncolors; ++i3) {
    blocksPerGrid = planRet -> ncolblk[i3];
    dynamicSharedMemorySize = planRet -> nshared;
    threadsPerBlock = threadsPerBlockSize_res_calc;
    totalThreadNumber = threadsPerBlock * blocksPerGrid;
    kernelPointer = getKernel("res_calc_kernel");
    errorCode = clSetKernelArg(kernelPointer,0,sizeof(cl_mem ),&opDat1.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,1,sizeof(cl_mem ),&opDat3.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,2,sizeof(cl_mem ),&opDat5.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,3,sizeof(cl_mem ),&opDat7.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,4,sizeof(cl_mem ),&planRet -> ind_maps[0]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,5,sizeof(cl_mem ),&planRet -> ind_maps[1]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,6,sizeof(cl_mem ),&planRet -> ind_maps[2]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,7,sizeof(cl_mem ),&planRet -> ind_maps[3]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,8,sizeof(cl_mem ),&planRet -> loc_maps[0]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,9,sizeof(cl_mem ),&planRet -> loc_maps[1]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,10,sizeof(cl_mem ),&planRet -> loc_maps[2]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,11,sizeof(cl_mem ),&planRet -> loc_maps[3]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,12,sizeof(cl_mem ),&planRet -> loc_maps[4]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,13,sizeof(cl_mem ),&planRet -> loc_maps[5]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,14,sizeof(cl_mem ),&planRet -> loc_maps[6]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,15,sizeof(cl_mem ),&planRet -> loc_maps[7]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,16,sizeof(cl_mem ),&planRet -> ind_sizes);
    errorCode = errorCode | clSetKernelArg(kernelPointer,17,sizeof(cl_mem ),&planRet -> ind_offs);
    errorCode = errorCode | clSetKernelArg(kernelPointer,18,sizeof(cl_mem ),&planRet -> blkmap);
    errorCode = errorCode | clSetKernelArg(kernelPointer,19,sizeof(cl_mem ),&planRet -> offset);
    errorCode = errorCode | clSetKernelArg(kernelPointer,20,sizeof(cl_mem ),&planRet -> nelems);
    errorCode = errorCode | clSetKernelArg(kernelPointer,21,sizeof(cl_mem ),&planRet -> nthrcol);
    errorCode = errorCode | clSetKernelArg(kernelPointer,22,sizeof(cl_mem ),&planRet -> thrcol);
    errorCode = errorCode | clSetKernelArg(kernelPointer,23,sizeof(int ),&blockOffset);
    errorCode = errorCode | clSetKernelArg(kernelPointer,24,dynamicSharedMemorySize,NULL);
    errorCode = errorCode | clSetKernelArg(kernelPointer,25,sizeof(cl_mem ),&gm1_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,26,sizeof(cl_mem ),&eps_d);
    assert_m(errorCode == CL_SUCCESS,"Error setting OpenCL kernel arguments");
    errorCode = clEnqueueNDRangeKernel(cqCommandQueue,kernelPointer,1,NULL,&totalThreadNumber,&threadsPerBlock,0,NULL,&event);
    assert_m(errorCode == CL_SUCCESS,"Error executing OpenCL kernel");
    errorCode = clFinish(cqCommandQueue);
    assert_m(errorCode == CL_SUCCESS,"Error completing device command queue");
    blockOffset += blocksPerGrid;
  }
op_timers(&cpu_t2, &wall_t2);
op_timing_realloc(3);
  OP_kernels[3].name = userSubroutine;
  OP_kernels[3].count = OP_kernels[3].count + 1;
  OP_kernels[3].time = OP_kernels[3].time + (wall_t2 - wall_t1);
  OP_kernels[3].transfer = OP_kernels[3].transfer + planRet -> transfer;
  OP_kernels[3].transfer = OP_kernels[3].transfer + planRet -> transfer2;
}
void update_host(const char *userSubroutine,op_set set,op_arg opDat1,op_arg opDat2,op_arg opDat3,op_arg opDat4,op_arg opDat5)
{
#ifdef OP_BLOCK_SIZE_4
  threadsPerBlockSize_update = OP_BLOCK_SIZE_4;
#endif
  size_t blocksPerGrid;
  size_t threadsPerBlock;
  size_t totalThreadNumber;
  size_t dynamicSharedMemorySize;
  cl_int errorCode;
  cl_event event;
  cl_kernel kernelPointer;
  int sharedMemoryOffset;
  int i1;
  int i2;
  int reductionBytes;
  int reductionSharedMemorySize;
  float *reductionArrayHost5;
  double cpu_t1;
  double cpu_t2;
  double wall_t1;
op_timers(&cpu_t1, &wall_t1);
  double wall_t2;
  blocksPerGrid = 200;
  threadsPerBlock = threadsPerBlockSize_update;
  totalThreadNumber = threadsPerBlock * blocksPerGrid;
  dynamicSharedMemorySize = 0;
  dynamicSharedMemorySize = MAX(dynamicSharedMemorySize,sizeof(float ) * 4);
  dynamicSharedMemorySize = MAX(dynamicSharedMemorySize,sizeof(float ) * 4);
  dynamicSharedMemorySize = MAX(dynamicSharedMemorySize,sizeof(float ) * 4);
  dynamicSharedMemorySize = MAX(dynamicSharedMemorySize,sizeof(float ) * 4);
  sharedMemoryOffset = dynamicSharedMemorySize * OP_WARPSIZE;
  dynamicSharedMemorySize = dynamicSharedMemorySize * threadsPerBlock;
  reductionBytes = 0;
  reductionSharedMemorySize = 0;
  reductionArrayHost5 = ((float *)opDat5.data);
  reductionBytes += ROUND_UP(blocksPerGrid * sizeof(float ) * 1);
  reductionSharedMemorySize = MAX(reductionSharedMemorySize,sizeof(float ));
  reallocReductArrays(reductionBytes);
  reductionBytes = 0;
  opDat5.data = OP_reduct_h + reductionBytes;
  opDat5.data_d = ((char *)OP_reduct_d) + reductionBytes;
  for (i1 = 0; i1 < blocksPerGrid; ++i1) {
    for (i2 = 0; i2 < 1; ++i2) {
      ((float *)opDat5.data)[i2 + i1 * 1] = 0.00000F;
    }
  }
  reductionBytes += ROUND_UP(blocksPerGrid * sizeof(float ) * 1);
  mvReductArraysToDevice(reductionBytes);
  kernelPointer = getKernel("update_kernel");
  errorCode = clSetKernelArg(kernelPointer,0,sizeof(cl_mem ),&opDat1.data_d);
  errorCode = errorCode | clSetKernelArg(kernelPointer,1,sizeof(cl_mem ),&opDat2.data_d);
  errorCode = errorCode | clSetKernelArg(kernelPointer,2,sizeof(cl_mem ),&opDat3.data_d);
  errorCode = errorCode | clSetKernelArg(kernelPointer,3,sizeof(cl_mem ),&opDat4.data_d);
  errorCode = errorCode | clSetKernelArg(kernelPointer,4,sizeof(cl_mem ),&opDat5.data_d);
  errorCode = errorCode | clSetKernelArg(kernelPointer,5,sizeof(int ),&sharedMemoryOffset);
  errorCode = errorCode | clSetKernelArg(kernelPointer,6,sizeof(int ),&set -> size);
  errorCode = errorCode | clSetKernelArg(kernelPointer,7,dynamicSharedMemorySize,NULL);
  assert_m(errorCode == CL_SUCCESS,"Error setting OpenCL kernel arguments");
  errorCode = clEnqueueNDRangeKernel(cqCommandQueue,kernelPointer,1,NULL,&totalThreadNumber,&threadsPerBlock,0,NULL,&event);
  assert_m(errorCode == CL_SUCCESS,"Error executing OpenCL kernel");
  errorCode = clFinish(cqCommandQueue);
  assert_m(errorCode == CL_SUCCESS,"Error completing device command queue");
  mvReductArraysToHost(reductionBytes);
  for (i1 = 0; i1 < blocksPerGrid; ++i1) {
    for (i2 = 0; i2 < 1; ++i2) {
      reductionArrayHost5[i2] += ((float *)opDat5.data)[i2 + i1 * 1];
    }
  }
op_timers(&cpu_t2, &wall_t2);
op_timing_realloc(4);
  OP_kernels[4].name = userSubroutine;
  OP_kernels[4].count = OP_kernels[4].count + 1;
  OP_kernels[4].time = OP_kernels[4].time + (wall_t2 - wall_t1);
  OP_kernels[4].transfer = OP_kernels[4].transfer + ((float )(set -> size)) * opDat1.size * 1.00000F;
  OP_kernels[4].transfer = OP_kernels[4].transfer + ((float )(set -> size)) * opDat2.size * 1.00000F;
  OP_kernels[4].transfer = OP_kernels[4].transfer + ((float )(set -> size)) * opDat3.size * 2.00000F;
  OP_kernels[4].transfer = OP_kernels[4].transfer + ((float )(set -> size)) * opDat4.size * 1.00000F;
}
