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
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers(&cpu_t1, &wall_t1);
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

#ifdef PROFILE
  unsigned long tqueue, tsubmit, tstart, tend, telapsed;
  ciErrNum  = clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(tqueue), &tqueue, NULL );
  ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof(tsubmit), &tsubmit, NULL );
  ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_START, sizeof(tstart), &tstart, NULL );
  ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_END, sizeof(tend), &tend, NULL );
  assert_m( ciErrNum == CL_SUCCESS, "error getting profiling info" );
  OP_kernels[0].queue_time      += (tsubmit - tqueue);
  OP_kernels[0].wait_time       += (tstart - tsubmit);
  OP_kernels[0].execution_time  += (tend - tstart);
  //printf("%20lu\n%20lu\n%20lu\n%20lu\n\n", tqueue, tsubmit, tstart, tend);
  //printf("queue: %8.4f\nwait:%8.4f\nexec: %8.4f\n\n", OP_kernels[0].queue_time * 1.0e-9, OP_kernels[0].wait_time * 1.0e-9, OP_kernels[0].execution_time * 1.0e-9 );
#endif

  // update kernel record

  op_timers(&cpu_t2, &wall_t2);
  op_timing_realloc(0);
  OP_kernels[0].name      = userSubroutine;
  OP_kernels[0].count    += 1;
  OP_kernels[0].time     += wall_t2 - wall_t1;
  OP_kernels[0].transfer += (float)set->size * opDat1.size;
  OP_kernels[0].transfer += (float)set->size * opDat2.size;
}
