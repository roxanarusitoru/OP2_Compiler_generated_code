#include <utility>
#include <CL/cl.h>

//#include "op_lib_tuner.h"
#include "op_opencl_rt_support.h"

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


int threadsPerBlockSize_adt_calc = 512;
int setPartitionSize_adt_calc = 512;

void adt_calc_host(const char *userSubroutine,op_set set,op_arg opDat1,op_arg opDat2,op_arg opDat3,op_arg opDat4,op_arg opDat5,op_arg opDat6)
{
  //hack because when setting as kernel arg a var that is __constant, then doing
  //clSetKernelArg(kernel_pointer, some_index, sizeof(var), &var); does not work - ex: for a float. 
  //similar, using (in ex of float): sizeof(float), &gam - where gam is a float, does not work.
  cl_mem gam_d = op_allocate_constant(&gam, sizeof(float));
  cl_mem gm1_d = op_allocate_constant(&gm1, sizeof(float)); 
  cl_mem cfl_d = op_allocate_constant(&cfl, sizeof(float));
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
  indirectionDescriptorArray[2] = 0;
  indirectionDescriptorArray[3] = 0;
  indirectionDescriptorArray[4] = -1;
  indirectionDescriptorArray[5] = -1;
  //printf("ADT_CALC name: %s\n", userSubroutine);
  //printf("ADT_CALC setPartitionSize_adt_calc: %d\n", setPartitionSize_adt_calc);
  planRet = op_plan_get(userSubroutine,set,setPartitionSize_adt_calc,6,opDatArray,1,indirectionDescriptorArray);
  //printf("ADT_CALC_plan_retrieved\n");
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers(&cpu_t1, &wall_t1);
  blockOffset = 0;
  for (i3 = 0; i3 < planRet -> ncolors; ++i3) {
    blocksPerGrid = planRet -> ncolblk[i3];
    threadsPerBlock = threadsPerBlockSize_adt_calc;
    dynamicSharedMemorySize = planRet -> nshared;
    //printf("ADT_CALC shared %d\n", dynamicSharedMemorySize);
    totalThreadNumber = threadsPerBlock * blocksPerGrid;
    kernelPointer = getKernel("adt_calc_kernel");
    //printf("ADT_CALC_kernel_retrieved\n");
    errorCode = clSetKernelArg(kernelPointer,0,sizeof(cl_mem ),&opDat1.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,1,sizeof(cl_mem ),&opDat5.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,2,sizeof(cl_mem ),&opDat6.data_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,3,sizeof(cl_mem ),&planRet -> ind_maps[0]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,4,sizeof(cl_mem ),&planRet -> loc_maps[0]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,5,sizeof(cl_mem ),&planRet -> loc_maps[1]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,6,sizeof(cl_mem ),&planRet -> loc_maps[2]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,7,sizeof(cl_mem ),&planRet -> loc_maps[3]);
    errorCode = errorCode | clSetKernelArg(kernelPointer,8,sizeof(cl_mem ),&planRet -> ind_sizes);
    errorCode = errorCode | clSetKernelArg(kernelPointer,9,sizeof(cl_mem ),&planRet -> ind_offs);
    errorCode = errorCode | clSetKernelArg(kernelPointer,10,sizeof(cl_mem ),&planRet -> blkmap);
    errorCode = errorCode | clSetKernelArg(kernelPointer,11,sizeof(cl_mem ),&planRet -> offset);
    errorCode = errorCode | clSetKernelArg(kernelPointer,12,sizeof(cl_mem ),&planRet -> nelems);
    errorCode = errorCode | clSetKernelArg(kernelPointer,13,sizeof(cl_mem ),&planRet -> nthrcol);
    errorCode = errorCode | clSetKernelArg(kernelPointer,14,sizeof(cl_mem ),&planRet -> thrcol);
    errorCode = errorCode | clSetKernelArg(kernelPointer,15,sizeof(int ),&blockOffset);
    errorCode = errorCode | clSetKernelArg(kernelPointer,16,dynamicSharedMemorySize, NULL);
    errorCode = errorCode | clSetKernelArg(kernelPointer,17,sizeof(cl_mem ),&gam_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,18,sizeof(cl_mem ),&gm1_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,19,sizeof(cl_mem ),&cfl_d);
    assert_m(errorCode == CL_SUCCESS,"Error setting OpenCL kernel arguments adt_calc");
    //printf("ADT_CALC_arguments_set\n");
    errorCode = clEnqueueNDRangeKernel(cqCommandQueue,kernelPointer,1,NULL,&totalThreadNumber,&threadsPerBlock,0,NULL,&event);
    //printf("ADT_CALC_kernel_executed\n");
    assert_m(errorCode == CL_SUCCESS,"Error executing OpenCL kernel adt_calc");
    //printf("ADT_CALC_command_queue %d\n", cqCommandQueue);
    errorCode = clFinish(cqCommandQueue);
    //printf("ADT_CALC_command_queue_completed\n");
    assert_m(errorCode == CL_SUCCESS,"Error completing device command queue");

#ifdef PROFILE
    cl_ulong tqueue, tsubmit, tstart, tend, telapsed;
    ciErrNum = clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &tqueue, NULL );
    ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &tsubmit, NULL );
    ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tstart, NULL );
    ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL );
    assert_m( ciErrNum == CL_SUCCESS, "error getting profiling info" );
    OP_kernels[1].queue_time      += (tsubmit - tqueue);
    OP_kernels[1].wait_time       += (tstart - tsubmit);
    OP_kernels[1].execution_time  += (tend - tstart);
#endif

    blockOffset += blocksPerGrid;

  }
  op_timers(&cpu_t2, &wall_t2);
  op_timing_realloc(1);
  OP_kernels[1].name       = userSubroutine;
  OP_kernels[1].count     += 1;
  OP_kernels[1].time      += wall_t2 - wall_t1;
  OP_kernels[1].transfer  += planRet->transfer;
  OP_kernels[1].transfer2 += planRet->transfer2;
}

int threadsPerBlockSize_res_calc = 512;
int setPartitionSize_res_calc = 512;

void res_calc_host(const char *userSubroutine,op_set set,op_arg opDat1,op_arg opDat2,op_arg opDat3,op_arg opDat4,op_arg opDat5,op_arg opDat6,op_arg opDat7,op_arg opDat8)
{
  cl_mem gm1_d = op_allocate_constant(&gm1, sizeof(float));
  cl_mem eps_d = op_allocate_constant(&eps, sizeof(float));
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
  //printf("RES_CALC_plan_retrieved\n");
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers(&cpu_t1, &wall_t1);
  blockOffset = 0;
  for (i3 = 0; i3 < planRet -> ncolors; ++i3) {
    blocksPerGrid = planRet -> ncolblk[i3];
    dynamicSharedMemorySize = planRet -> nshared;
    threadsPerBlock = threadsPerBlockSize_res_calc;
    totalThreadNumber = threadsPerBlock * blocksPerGrid;
    kernelPointer = getKernel("res_calc_kernel");
    //printf("RES_CALC_kernel_retrieved\n");
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
    //errorCode = errorCode | clSetKernelArg(kernelPointer,24,sizeof(cl_mem ),&dynamicSharedMemorySize);
    errorCode = errorCode | clSetKernelArg(kernelPointer,24,dynamicSharedMemorySize,NULL);
    errorCode = errorCode | clSetKernelArg(kernelPointer,25,sizeof(cl_mem ),&gm1_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,26,sizeof(cl_mem ),&eps_d);
    assert_m(errorCode == CL_SUCCESS,"Error setting OpenCL kernel arguments res_calc");
    //printf("RES_CALC_arguments_set\n");
    errorCode = clEnqueueNDRangeKernel(cqCommandQueue,kernelPointer,1,NULL,&totalThreadNumber,&threadsPerBlock,0,NULL,&event);
    assert_m(errorCode == CL_SUCCESS,"Error executing OpenCL kernel res_calc");
    //printf("RES_CALC_kernel_executed\n");
    errorCode = clFinish(cqCommandQueue);
    assert_m(errorCode == CL_SUCCESS,"Error completing device command queue");
    //printf("RES_CALC_command_queue_completed");
#ifdef PROFILE
    cl_ulong tqueue, tsubmit, tstart, tend, telapsed;
    ciErrNum = clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &tqueue, NULL );
    ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &tsubmit, NULL );
    ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tstart, NULL );
    ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL );
    assert_m( ciErrNum == CL_SUCCESS, "error getting profiling info" );
    OP_kernels[2].queue_time      += (tsubmit - tqueue);
    OP_kernels[2].wait_time       += (tstart - tsubmit);
    OP_kernels[2].execution_time  += (tend - tstart);
#endif

    blockOffset += blocksPerGrid;
  }
  op_timers(&cpu_t2, &wall_t2);
  op_timing_realloc(2);
  OP_kernels[2].name       = userSubroutine;
  OP_kernels[2].count     += 1;
  OP_kernels[2].time      += wall_t2 - wall_t1;
  OP_kernels[2].transfer  += planRet->transfer;
  OP_kernels[2].transfer2 += planRet->transfer2;
}

int threadsPerBlockSize_bres_calc = 256;
int setPartitionSize_bres_calc = 256;

long max(long a, long b) {
  if (a >= b) {
    return a;
  } 
  return b;
}
void bres_calc_host(const char *userSubroutine,op_set set,op_arg opDat1,op_arg opDat2,op_arg opDat3,op_arg opDat4,op_arg opDat5,op_arg opDat6)
{
//  printf("bres_calc_set %d\n", set);
  cl_mem qinf_d = op_allocate_constant(&qinf, 4*sizeof(float));
  cl_mem gm1_d = op_allocate_constant(&gm1, sizeof(float));
  cl_mem eps_d = op_allocate_constant(&eps, sizeof(float));
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
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers(&cpu_t1, &wall_t1);
  blockOffset = 0;
  for (i3 = 0; i3 < planRet -> ncolors; ++i3) {
    blocksPerGrid = planRet -> ncolblk[i3];
    dynamicSharedMemorySize = planRet -> nshared;
    threadsPerBlock = threadsPerBlockSize_bres_calc;
    totalThreadNumber = threadsPerBlock * blocksPerGrid;

//    printf("nshared %d, nblocks %d\n", dynamicSharedMemorySize, blocksPerGrid);
    
    kernelPointer = getKernel("bres_calc_kernel");
    //op_fetch_data(opDat1.dat);
    //printf("BRES_CALC_kernel_retrieved\n");
    //for(long i = 0; i < 2*721801; ++i) {
    //  printf("%f\n", (float) opDat1.dat->data[i]);
    //}
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
    //errorCode = errorCode | clSetKernelArg(kernelPointer,22,sizeof(cl_mem ),&dynamicSharedMemorySize);
    errorCode = errorCode | clSetKernelArg(kernelPointer,22,dynamicSharedMemorySize,NULL);
    errorCode = errorCode | clSetKernelArg(kernelPointer,23,sizeof(cl_mem ),&gm1_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,24,sizeof(cl_mem ),&qinf_d);
    errorCode = errorCode | clSetKernelArg(kernelPointer,25,sizeof(cl_mem ),&eps_d);
    assert_m(errorCode == CL_SUCCESS,"Error setting OpenCL kernel arguments bres_calc");
    //printf("BRES_CALC_arguments_set\m");
    errorCode = clEnqueueNDRangeKernel(cqCommandQueue,kernelPointer,1,NULL,&totalThreadNumber,&threadsPerBlock,0,NULL,&event);
    assert_m(errorCode == CL_SUCCESS,"Error executing OpenCL kernel bres_calc");
    //printf("BRES_CALC_kernel_executed\n");
    errorCode = clFinish(cqCommandQueue);
    assert_m(errorCode == CL_SUCCESS,"Error completing device command queue");
    //printf("BRES_CALC_command_queue_completed\n");
#ifdef PROFILE
    cl_ulong tqueue, tsubmit, tstart, tend, telapsed;
    ciErrNum = clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &tqueue, NULL );
    ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &tsubmit, NULL );
    ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tstart, NULL );
    ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL );
    assert_m( ciErrNum == CL_SUCCESS, "error getting profiling info" );
    OP_kernels[3].queue_time      += (tsubmit - tqueue);
    OP_kernels[3].wait_time       += (tstart - tsubmit);
    OP_kernels[3].execution_time  += (tend - tstart);
#endif
    blockOffset += blocksPerGrid;
  }

  op_timers(&cpu_t2, &wall_t2);
  op_timing_realloc(3);
  OP_kernels[3].name       = userSubroutine;
  OP_kernels[3].count     += 1;
  OP_kernels[3].time      += wall_t2 - wall_t1;
  OP_kernels[3].transfer  += planRet->transfer;
  OP_kernels[3].transfer2 += planRet->transfer2;
}

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
