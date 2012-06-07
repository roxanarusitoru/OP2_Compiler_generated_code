
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
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers(&cpu_t1, &wall_t1);

  blocksPerGrid = 200;
  threadsPerBlock = threadsPerBlockSize_update;
  totalThreadNumber = threadsPerBlock * blocksPerGrid;
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
  dynamicSharedMemorySize = 0;
  dynamicSharedMemorySize = MAX(dynamicSharedMemorySize,sizeof(float ) * 4);
  dynamicSharedMemorySize = MAX(dynamicSharedMemorySize,sizeof(float ) * 4);
  dynamicSharedMemorySize = MAX(dynamicSharedMemorySize,sizeof(float ) * 4);
  dynamicSharedMemorySize = MAX(dynamicSharedMemorySize,sizeof(float ) * 4);
  sharedMemoryOffset = dynamicSharedMemorySize * OP_WARPSIZE;
  //there is no need to use MAX as dynamicSharedMemorySize > reductionSharedMemorySize => 4float > flat
  //dynamicSharedMemorySize = dynamicSharedMemorySize * threadsPerBlock;
  dynamicSharedMemorySize = MAX(dynamicSharedMemorySize*threadsPerBlock, reductionSharedMemorySize*threadsPerBlock);

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
#ifdef PROFILE
    cl_ulong tqueue, tsubmit, tstart, tend, telapsed;
    ciErrNum = clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &tqueue, NULL );
    ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &tsubmit, NULL );
    ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tstart, NULL );
    ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL );
    assert_m( ciErrNum == CL_SUCCESS, "error getting profiling info" );
    OP_kernels[4].queue_time      += (tsubmit - tqueue);
    OP_kernels[4].wait_time       += (tstart - tsubmit);
    OP_kernels[4].execution_time  += (tend - tstart);
#endif

  op_timers(&cpu_t2,&wall_t2);
  op_timing_realloc(4);
  OP_kernels[4].name      = userSubroutine;
  OP_kernels[4].count    += 1;
  OP_kernels[4].time     += wall_t2 - wall_t1;
  OP_kernels[4].transfer += (float)set->size * opDat1.size;
  OP_kernels[4].transfer += (float)set->size * opDat2.size;
  OP_kernels[4].transfer += (float)set->size * opDat3.size*2.0f;
  OP_kernels[4].transfer += (float)set->size * opDat4.size;
  
}
