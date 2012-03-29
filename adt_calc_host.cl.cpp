/*
extern float alpha;
extern float cfl;
extern float eps;
extern float gam;
extern float gm1;
extern float mach;
extern float qinf[4UL];*/
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
    blockOffset += blocksPerGrid;
  }
}

