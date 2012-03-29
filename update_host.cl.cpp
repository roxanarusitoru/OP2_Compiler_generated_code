
extern float alpha;
extern float cfl;
extern float eps;
extern float gam;
extern float gm1;
extern float mach;
extern float qinf[4UL];
int threadsPerBlockSize_update = 512;

void update_host(const char *userSubroutine,op_set set,op_arg opDat1,op_arg opDat2,op_arg opDat3,op_arg opDat4,op_arg opDat5)
{
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
  opDat5.data_d = (char*) OP_reduct_d + reductionBytes;
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
  //errorCode = errorCode | clSetKernelArg(kernelPointer,7,sizeof(size_t ),&dynamicSharedMemorySize);
  errorCode = errorCode | clSetKernelArg(kernelPointer,7,dynamicSharedMemorySize,NULL);
  assert_m(errorCode == CL_SUCCESS,"Error setting OpenCL kernel arguments update_calc");
  errorCode = clEnqueueNDRangeKernel(cqCommandQueue,kernelPointer,1,NULL,&totalThreadNumber,&threadsPerBlock,0,NULL,&event);
  assert_m(errorCode == CL_SUCCESS,"Error executing OpenCL kernel update_calc");
  errorCode = clFinish(cqCommandQueue);
  assert_m(errorCode == CL_SUCCESS,"Error completing device command queue");
  mvReductArraysToHost(reductionBytes);
  for (i1 = 0; i1 < blocksPerGrid; ++i1) {
    for (i2 = 0; i2 < 1; ++i2) {
      reductionArrayHost5[i2] += ((float *)opDat5.data)[i2 + i1 * 1];
    }
  }
}
