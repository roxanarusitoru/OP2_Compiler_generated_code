extern float alpha;
extern float cfl;
extern float eps;
extern float gam;
extern float gm1;
extern float mach;
extern float qinf[4UL];
int threadsPerBlockSize_save_soln = 512;

void save_soln_host(const char *userSubroutine,op_set set,op_arg opDat1,op_arg opDat2)
{
  size_t blocksPerGrid;
  size_t threadsPerBlock;
  size_t totalThreadNumber;
  size_t dynamicSharedMemorySize;
  cl_int errorCode;
  cl_event event;
  cl_kernel kernelPointer;
  int sharedMemoryOffset;
  blocksPerGrid = 200;
  threadsPerBlock = threadsPerBlockSize_save_soln;
  totalThreadNumber = threadsPerBlock * blocksPerGrid;
  dynamicSharedMemorySize = 0;
  dynamicSharedMemorySize = MAX(dynamicSharedMemorySize,sizeof(float ) * 4);
  dynamicSharedMemorySize = MAX(dynamicSharedMemorySize,sizeof(float ) * 4);
  sharedMemoryOffset = dynamicSharedMemorySize * OP_WARPSIZE;
  dynamicSharedMemorySize = dynamicSharedMemorySize * threadsPerBlock;
  kernelPointer = getKernel("save_soln_kernel");
  errorCode = clSetKernelArg(kernelPointer,0,sizeof(cl_mem ),&opDat1.data_d);
  errorCode = errorCode | clSetKernelArg(kernelPointer,1,sizeof(cl_mem ),&opDat2.data_d);
  errorCode = errorCode | clSetKernelArg(kernelPointer,2,sizeof(int ),&sharedMemoryOffset);
  errorCode = errorCode | clSetKernelArg(kernelPointer,3,sizeof(int ),&set -> size);
  //errorCode = errorCode | clSetKernelArg(kernelPointer,4,sizeof(size_t ),&dynamicSharedMemorySize);
  errorCode = errorCode | clSetKernelArg(kernelPointer,4,dynamicSharedMemorySize,NULL);
  //printf("errorCode after 5: %d\n", errorCode);
  assert_m(errorCode == CL_SUCCESS,"Error setting OpenCL kernel arguments save_soln");
  errorCode = clEnqueueNDRangeKernel(cqCommandQueue,kernelPointer,1,NULL,&totalThreadNumber,&threadsPerBlock,0,NULL,&event);
  assert_m(errorCode == CL_SUCCESS,"Error executing OpenCL kernel save_soln");
  errorCode = clFinish(cqCommandQueue);
  assert_m(errorCode == CL_SUCCESS,"Error completing device command queue");
}
