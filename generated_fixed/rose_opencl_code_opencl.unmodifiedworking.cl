#define ROUND_UP(bytes) (((bytes) + 15) & ~15)
#define MIN(a,b) ((a<b) ? (a) : (b))
#define ZERO_float 0.0f
#pragma OPENCL EXTENSION cl_intel_printf : enable
__kernel void ReductionFloat4(__global float *volatile reductionResult,__private float inputValue,__private int reductionOperation,__local float *sharedFloat4)
//inline void ReductionFloat4(__global float volatile *reductionResult, float inputValue, int reductionOperation, __local float *sharedFloat4)
{
  __local float *volatile volatileSharedFloat4;
  int i1;
  int threadID;
  threadID = get_local_id(0);
  i1 = get_local_size(0) >> 1;
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedFloat4[threadID] = inputValue;
  for (; i1 > OP_WARPSIZE; i1 >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (threadID < i1) {
      switch(reductionOperation){
        case 0:
{
          sharedFloat4[threadID] = sharedFloat4[threadID] + sharedFloat4[threadID + i1];
          break; 
        }
        case 1:
{
          if (sharedFloat4[threadID + i1] < sharedFloat4[threadID]) {
            sharedFloat4[threadID] = sharedFloat4[threadID + i1];
          }
          break; 
        }
        case 2:
{
          if (sharedFloat4[threadID + i1] > sharedFloat4[threadID]) {
            sharedFloat4[threadID] = sharedFloat4[threadID + i1];
          }
          break; 
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  volatileSharedFloat4 = sharedFloat4;
  if (threadID < OP_WARPSIZE) {
    for (; i1 > 0; i1 >>= 1) {
      if (threadID < i1) {
        switch(reductionOperation){
          case 0:
{
            volatileSharedFloat4[threadID] = volatileSharedFloat4[threadID] + volatileSharedFloat4[threadID + i1];
            break; 
          }
          case 1:
{
            if (volatileSharedFloat4[threadID + i1] < volatileSharedFloat4[threadID]) {
              volatileSharedFloat4[threadID] = volatileSharedFloat4[threadID + i1];
            }
            break; 
          }
          case 2:
{
            if (volatileSharedFloat4[threadID + i1] > volatileSharedFloat4[threadID]) {
              volatileSharedFloat4[threadID] = volatileSharedFloat4[threadID + i1];
            }
            break; 
          }
        }
      }
    }
  }
  if (threadID == 0) {
    switch(reductionOperation){
      case 0:
{
         *reductionResult =  *reductionResult + volatileSharedFloat4[0];
        break; 
      }
      case 1:
{
        if (sharedFloat4[0] <  *reductionResult) {
           *reductionResult = volatileSharedFloat4[0];
        }
        break; 
      }
      case 2:
{
        if (sharedFloat4[0] >  *reductionResult) {
           *reductionResult = volatileSharedFloat4[0];
        }
        break; 
      }
    }
  }
}

inline void adt_calc_modified(__local float *x1,__local float *x2,__local float *x3,__local float *x4,__global float *q,__global float *adt, __constant float* gam, __constant float* gm1, __constant float* cfl)
{
  //printf("EXECUTING ADT_CALC INLINE\n");
  float dx;
  float dy;
  float ri;
  float u;
  float v;
  float c;
  ri = (1.0f / q[0]);
  u = (ri * q[1]);
  v = (ri * q[2]);
  c = (sqrt(((*gam * *gm1) * ((ri * q[3]) - (0.5f * ((u * u) + (v * v)))))));
  dx = (x2[0] - x1[0]);
  dy = (x2[1] - x1[1]);
   *adt = (fabs(((u * dy) - (v * dx))) + (c * sqrt(((dx * dx) + (dy * dy)))));
  dx = (x3[0] - x2[0]);
  dy = (x3[1] - x2[1]);
   *adt += (fabs(((u * dy) - (v * dx))) + (c * sqrt(((dx * dx) + (dy * dy)))));
  dx = (x4[0] - x3[0]);
  dy = (x4[1] - x3[1]);
   *adt += (fabs(((u * dy) - (v * dx))) + (c * sqrt(((dx * dx) + (dy * dy)))));
  dx = (x1[0] - x4[0]);
  dy = (x1[1] - x4[1]);
   *adt += (fabs(((u * dy) - (v * dx))) + (c * sqrt(((dx * dx) + (dy * dy)))));
   *adt = ( *adt / *cfl);
  //printf("FINISHED EXECUTING ADT_CALC INLINE\n");
}

__kernel void adt_calc_kernel(__global float *opDat1,__global float *opDat5,__global float *opDat6,__global int *ind_maps1,__global short *mappingArray1,__global short *mappingArray2,__global short *mappingArray3,__global short *mappingArray4,__global int *pindSizes,__global int *pindOffs,__global int *pblkMap,__global int *poffset,__global int *pnelems,__global int *pnthrcol,__global int *pthrcol,__private int blockOffset,__local float *shared_adt_calc,__constant float *gam,__constant float *gm1,__constant float *cfl)
{
  //printf("EXECUTING ADT_CALC\n");
  __local int sharedMemoryOffset;
  __local int numberOfActiveThreads;
  int nbytes;
  int blockID;
  int i1;
  __global int *opDat1IndirectionMap;
  __local int opDat1SharedIndirectionSize;
  __local float *opDat1SharedIndirection;
  if (get_local_id(0) == 0) {
    //blockID = pblkMap[get_global_size(0) + blockOffset];
    blockID = pblkMap[get_group_id(0) + blockOffset];
    numberOfActiveThreads = pnelems[blockID];
    sharedMemoryOffset = poffset[blockID];
    opDat1SharedIndirectionSize = pindSizes[0 + blockID * 1];
    opDat1IndirectionMap = ind_maps1 + pindOffs[0 + blockID * 1];
    nbytes = 0;
    opDat1SharedIndirection = (shared_adt_calc + nbytes);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  //printf("ADT_CALC BARRIER 1 REACHED\n");
  for (i1 = get_local_id(0); i1 < opDat1SharedIndirectionSize * 2; i1 += get_local_size(0)) {
    opDat1SharedIndirection[i1] = opDat1[i1 % 2 + opDat1IndirectionMap[i1 / 2] * 2];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  //printf("ADT_CALC BARRIER 2 REACHED\n");
  for (i1 = get_local_id(0); i1 < numberOfActiveThreads; i1 += get_local_size(0)) {
    adt_calc_modified(opDat1SharedIndirection + mappingArray1[i1 + sharedMemoryOffset] * 2,opDat1SharedIndirection + mappingArray2[i1 + sharedMemoryOffset] * 2,opDat1SharedIndirection + mappingArray3[i1 + sharedMemoryOffset] * 2,opDat1SharedIndirection + mappingArray4[i1 + sharedMemoryOffset] * 2,opDat5 + (i1 + sharedMemoryOffset) * 4,opDat6 + (i1 + sharedMemoryOffset) * 1,gam,gm1,cfl);
  }
  //printf("FINISHED EXECUTING ADT_CALC\n");
}

inline void bres_calc_modified(__local float *x1,__local float *x2,__local float *q1,__local float *adt1,float *res1,__global int *bound,__constant float* gm1,__constant float *qinf,__constant float* eps)
{
  //printf("%f\n%f\n", x1[0], x1[1]);
  //printf("%f\n%f\n", x2[0], x2[1]);
  //printf("%f\n%f\n%f\n%f\n",q1[0], q1[1], q1[2], q1[3]);
  //printf("%f\n", *adt1);
  //printf("%f\n%f\n%f\n%f\n", res1[0], res1[1], res1[2], res1[3]);
  //printf("%d\n", *bound );
  //printf("%f\n%f\n%f\n%f\n%f\n%f\n", *gm1, qinf[0], qinf[1], qinf[2], qinf[3], *eps);
  
  float dx;
  float dy;
  float mu;
  float ri;
  float p1;
  float vol1;
  float p2;
  float vol2;
  float f;
  dx = (x1[0] - x2[0]);
  dy = (x1[1] - x2[1]);
  ri = (1.0f / q1[0]);
  p1 = (*gm1 * (q1[3] - ((0.5f * ri) * ((q1[1] * q1[1]) + (q1[2] * q1[2])))));
  if ( *bound == 1) {
    res1[1] += (+p1 * dy);
    res1[2] += (-p1 * dx);
  }
  else {
    vol1 = (ri * ((q1[1] * dy) - (q1[2] * dx)));
    ri = (1.0f / qinf[0]);
    p2 = (*gm1 * (qinf[3] - ((0.5f * ri) * ((qinf[1] * qinf[1]) + (qinf[2] * qinf[2])))));
    vol2 = (ri * ((qinf[1] * dy) - (qinf[2] * dx)));
    mu = ( *adt1 * *eps);
    f = ((0.5f * ((vol1 * q1[0]) + (vol2 * qinf[0]))) + (mu * (q1[0] - qinf[0])));
    res1[0] += f;
    f = ((0.5f * ((((vol1 * q1[1]) + (p1 * dy)) + (vol2 * qinf[1])) + (p2 * dy))) + (mu * (q1[1] - qinf[1])));
    res1[1] += f;
    f = ((0.5f * ((((vol1 * q1[2]) - (p1 * dx)) + (vol2 * qinf[2])) - (p2 * dx))) + (mu * (q1[2] - qinf[2])));
    res1[2] += f;
    f = ((0.5f * ((vol1 * (q1[3] + p1)) + (vol2 * (qinf[3] + p2)))) + (mu * (q1[3] - qinf[3])));
    res1[3] += f;
  }
}
/*
__kernel void bres_calc_kernel(__global float *opDat1,__global float *opDat3,__global float *opDat4,__global float *opDat5,__global int *opDat6,__global int *ind_maps1,__global int *ind_maps3,__global int *ind_maps4,__global int *ind_maps5,__global short *mappingArray1,__global short *mappingArray2,__global short *mappingArray3,__global short *mappingArray4,__global short *mappingArray5,__global int *pindSizes,__global int *pindOffs,__global int *pblkMap,__global int *poffset,__global int *pnelems,__global int *pnthrcol,__global int *pthrcol,__private int blockOffset,__local float *shared_bres_calc,__constant float *gm1,__constant float *qinf,__constant float *eps)
{
  //printf("EXECUTING BRES_CALC\n");
  float opDat5Local[4];
  __local int sharedMemoryOffset;
  __local int numberOfActiveThreads;
  int nbytes;
  int blockID;
  int i1;
  __global int *opDat1IndirectionMap;
  __global int *opDat3IndirectionMap;
  __global int *opDat4IndirectionMap;
  __global int *opDat5IndirectionMap;
  __local int opDat1SharedIndirectionSize;
  __local int opDat3SharedIndirectionSize;
  __local int opDat4SharedIndirectionSize;
  __local int opDat5SharedIndirectionSize;
  __local float *opDat1SharedIndirection;
  __local float *opDat3SharedIndirection;
  __local float *opDat4SharedIndirection;
  __local float *opDat5SharedIndirection;
  __local int numOfColours;
  __local int numberOfActiveThreadsCeiling;
  int colour1;
  int colour2;
  int i2;
  if (get_local_id(0) == 0) {
    blockID = pblkMap[get_group_id(0) + blockOffset];
    //printf("%d\n", blockID); - the same 
    numberOfActiveThreads = pnelems[blockID];
    //printf("%d\n", numberOfActiveThreads); - same
    numberOfActiveThreadsCeiling = get_local_size(0) * (1 + (numberOfActiveThreads - 1) / get_local_size(0));
    //printf("%d\n", numberOfActiveThreadsCeiling); - same
    numOfColours = pnthrcol[blockID];
    //printf("%d\n", numOfColours); -same
    sharedMemoryOffset = poffset[blockID];
    //printf("%d\n", sharedMemoryOffset); -same
    opDat1SharedIndirectionSize = pindSizes[0 + blockID * 4];
    opDat3SharedIndirectionSize = pindSizes[1 + blockID * 4];
    opDat4SharedIndirectionSize = pindSizes[2 + blockID * 4];
    opDat5SharedIndirectionSize = pindSizes[3 + blockID * 4];
    //printf("%d\n%d\n%d\n%d\n", opDat1SharedIndirectionSize, opDat3SharedIndirectionSize, opDat4SharedIndirectionSize, opDat5SharedIndirectionSize); - same
    opDat1IndirectionMap = ind_maps1 + pindOffs[0 + blockID * 4];
    opDat3IndirectionMap = ind_maps3 + pindOffs[1 + blockID * 4];
    opDat4IndirectionMap = ind_maps4 + pindOffs[2 + blockID * 4];
    opDat5IndirectionMap = ind_maps5 + pindOffs[3 + blockID * 4];
    nbytes = 0;
    opDat1SharedIndirection = ((&shared_bres_calc[nbytes]));
    nbytes += ROUND_UP(opDat1SharedIndirectionSize * (2));
    opDat3SharedIndirection = ((&shared_bres_calc[nbytes]));
    nbytes += ROUND_UP(opDat3SharedIndirectionSize * (4));
    opDat4SharedIndirection = ((&shared_bres_calc[nbytes]));
    nbytes += ROUND_UP(opDat4SharedIndirectionSize * (1));
    opDat5SharedIndirection = ((&shared_bres_calc[nbytes]));
    //printf("%d\n", nbytes); - same
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (i1 = get_local_id(0); i1 < opDat1SharedIndirectionSize * 2; i1 += get_local_size(0)) {
    opDat1SharedIndirection[i1] = opDat1[i1 % 2 + opDat1IndirectionMap[i1 / 2] * 2];
    //printf("%d\n", opDat1IndirectionMap[i1/2]);
    //printf("%f\n",opDat1SharedIndirection[i1]);
  }
  for (i1 = get_local_id(0); i1 < opDat3SharedIndirectionSize * 4; i1 += get_local_size(0)) {
    opDat3SharedIndirection[i1] = opDat3[i1 % 4 + opDat3IndirectionMap[i1 / 4] * 4];
  }
  for (i1 = get_local_id(0); i1 < opDat4SharedIndirectionSize * 1; i1 += get_local_size(0)) {
    opDat4SharedIndirection[i1] = opDat4[i1 % 1 + opDat4IndirectionMap[i1 / 1] * 1];
  }
  for (i1 = get_local_id(0); i1 < opDat5SharedIndirectionSize * 4; i1 += get_local_size(0)) {
    opDat5SharedIndirection[i1] = 0.00000F;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (i1 = get_local_id(0); i1 < numberOfActiveThreadsCeiling; i1 += get_local_size(0)) {
    colour2 = -1;
    if (i1 < numberOfActiveThreads) {
      for (i2 = 0; i2 < 4; ++i2) {
        opDat5Local[i2] = 0.00000F;
      }
      bres_calc_modified(opDat1SharedIndirection + mappingArray1[i1 + sharedMemoryOffset] * 2,opDat1SharedIndirection + mappingArray2[i1 + sharedMemoryOffset] * 2,opDat3SharedIndirection + mappingArray3[i1 + sharedMemoryOffset] * 4,opDat4SharedIndirection + mappingArray4[i1 + sharedMemoryOffset] * 1,opDat5Local,opDat6 + (i1 + sharedMemoryOffset) * 1,gm1,qinf,eps);
      colour2 = pthrcol[i1 + sharedMemoryOffset];
    }
    for (colour1 = 0; colour1 < numOfColours; ++colour1) {
      if (colour2 == colour1) {
//        printf("local[0] = %f, local[1] = %f, local[2] = %f, local[3] = %f\n", opDat5Local[0], opDat5Local[1], opDat5Local[2], opDat5Local[3]);
        for (i2 = 0; i2 < 4; ++i2) {
          opDat5SharedIndirection[i2 + mappingArray5[i1 + sharedMemoryOffset] * 4] += opDat5Local[i2];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  for (i1 = get_local_id(0); i1 < opDat5SharedIndirectionSize * 4; i1 += get_local_size(0)) {
    opDat5[i1 % 4 + opDat5IndirectionMap[i1 / 4] * 4] += opDat5SharedIndirection[i1];
  }
  //printf("FINISHED EXECUTING BRES_CALC\n");
}
*/

__kernel void bres_calc_kernel(
  __global float *opDat1,
  __global float *opDat3,
  __global float *opDat4,
  __global float *opDat5,
  __global int *opDat6,
  __global int *ind_maps1,
  __global int *ind_maps3,
  __global int *ind_maps4,
  __global int *ind_maps5,
  __global short *mappingArray1,
  __global short *mappingArray2,
  __global short *mappingArray3,
  __global short *mappingArray4,
  __global short *mappingArray5,
  __global int   *pindSizes,
  __global int   *pindOffs,
  __global int   *pblkMap,
  __global int   *poffset,
  __global int   *pnelems,
  __global int   *pnthrcol,
  __global int   *pthrcol,
  __private  int    blockOffset,
  __local  float  *shared_bres_calc, 
  __constant float *gm1,
  __constant float  *qinf,
  __constant float *eps) {

  float opDat5Local[4];

  __global int   * __local opDat1IndirectionMap, * __local opDat3IndirectionMap, * __local opDat4IndirectionMap, * __local opDat5IndirectionMap;
//    __global int *opDat1IndirectionMap, *opDat3IndirectionMap, *opDat4IndirectionMap, *opDat5IndirectionMap; --WRONG!
  __local int opDat1SharedIndirectionSize, opDat3SharedIndirectionSize, opDat4SharedIndirectionSize, opDat5SharedIndirectionSize;
  __local float * __local opDat1SharedIndirection;
  __local float * __local opDat3SharedIndirection;
  __local float * __local opDat4SharedIndirection;
  __local float * __local opDat5SharedIndirection;
  __local int    numberOfActiveThreadsCeiling, numOfColours;
  __local int    numberOfActiveThreads, sharedMemoryOffset;
  int colour1, colour2;
  int i1, i2;
  int blockID, nbytes;

  if (get_local_id(0)==0) {

    // get sizes and shift pointers and direct-mapped data
  
    blockID = pblkMap[get_group_id(0) + blockOffset];
    //printf("%d\n",blockID);
    numberOfActiveThreads    = pnelems[blockID];
    //printf("%d\n", numberOfActiveThreads);
    sharedMemoryOffset = poffset[blockID];
    //printf("%d\n", sharedMemoryOffset);
    numberOfActiveThreadsCeiling  = get_local_size(0)*(1+(numberOfActiveThreads-1)/get_local_size(0));
    //printf("%d\n", numberOfActiveThreadsCeiling);
    numOfColours   = pnthrcol[blockID];
    //printf("%d\n", numOfColours); 
 
    opDat1SharedIndirectionSize = pindSizes[0+blockID*4];
    opDat3SharedIndirectionSize = pindSizes[1+blockID*4];
    opDat4SharedIndirectionSize = pindSizes[2+blockID*4];
    opDat5SharedIndirectionSize = pindSizes[3+blockID*4];
    //printf("%d\n%d\n%d\n%d\n", opDat1SharedIndirectionSize, opDat3SharedIndirectionSize, opDat4SharedIndirectionSize, opDat5SharedIndirectionSize);

    opDat1IndirectionMap = ind_maps1 + pindOffs[0+blockID*4];
    opDat3IndirectionMap = ind_maps3 + pindOffs[1+blockID*4];
    opDat4IndirectionMap = ind_maps4 + pindOffs[2+blockID*4];
    opDat5IndirectionMap = ind_maps5 + pindOffs[3+blockID*4];

    // set shared memory pointers
    nbytes = 0;
    opDat1SharedIndirection = &shared_bres_calc[nbytes];
    nbytes    += ROUND_UP(opDat1SharedIndirectionSize*2);
    opDat3SharedIndirection = &shared_bres_calc[nbytes];
    nbytes    += ROUND_UP(opDat3SharedIndirectionSize*4);
    opDat4SharedIndirection = &shared_bres_calc[nbytes];
    nbytes    += ROUND_UP(opDat4SharedIndirectionSize*1);
    opDat5SharedIndirection = &shared_bres_calc[nbytes];
    //printf("%d\n", nbytes); 
  }

  barrier( CLK_LOCAL_MEM_FENCE ); 

  // copy indirect datasets into shared memory or zero increment
  for (i1=get_local_id(0); i1<opDat1SharedIndirectionSize*2; i1+=get_local_size(0)) {
    opDat1SharedIndirection[i1] = opDat1[i1%2+opDat1IndirectionMap[i1/2]*2];
    //printf("%d\n", opDat1IndirectionMap[i1/2]);
    //printf("%f\n", opDat1SharedIndirection[i1]);
  }

  for (i1=get_local_id(0); i1<opDat3SharedIndirectionSize*4; i1+=get_local_size(0))
    opDat3SharedIndirection[i1] = opDat3[i1%4+opDat3IndirectionMap[i1/4]*4];

  for (i1=get_local_id(0); i1<opDat4SharedIndirectionSize*1; i1+=get_local_size(0))
    opDat4SharedIndirection[i1] = opDat4[i1%1+opDat4IndirectionMap[i1/1]*1];

  for (i1=get_local_id(0); i1<opDat5SharedIndirectionSize*4; i1+=get_local_size(0))
    opDat5SharedIndirection[i1] = 0.00000F;

  barrier( CLK_LOCAL_MEM_FENCE );

  // process set elements
  for (i1=get_local_id(0); i1<numberOfActiveThreadsCeiling; i1+=get_local_size(0)) {
    colour2 = -1;

    if (i1<numberOfActiveThreads) {

      // initialise local variables
      for (i2=0; i2<4; ++i2)
        opDat5Local[i2] = 0.00000F;

      // user-supplied kernel call
      bres_calc_modified( opDat1SharedIndirection+mappingArray1[i1 + sharedMemoryOffset]*2,
                 opDat1SharedIndirection+mappingArray2[i1 + sharedMemoryOffset]*2,
                 opDat3SharedIndirection+mappingArray3[i1 + sharedMemoryOffset]*4,
                 opDat4SharedIndirection+mappingArray4[i1 + sharedMemoryOffset]*1,
                 opDat5Local,
                 opDat6+(i1 + sharedMemoryOffset)*1, gm1, qinf, eps);

      colour2 = pthrcol[i1 + sharedMemoryOffset];
    }

    // store local variables
    for (colour1=0; colour1<numOfColours; colour1++) {
      if (colour2==colour1) {
        //printf("local[0] = %f, local[1] = %f, local[2] = %f, local[3] = %f\n",  opDat5Local[0], opDat5Local[1], opDat5Local[2], opDat5Local[3]);
        for (i2=0; i2<4; ++i2)
          opDat5SharedIndirection[i2+mappingArray5[i1+sharedMemoryOffset]*4] += opDat5Local[i2];
      }
      barrier( CLK_LOCAL_MEM_FENCE );
    }
  }

  // apply pointered write/increment
  for (int i1=get_local_id(0); i1<opDat5SharedIndirectionSize*4; i1+=get_local_size(0))
    opDat5[i1%4+opDat5IndirectionMap[i1/4]*4] += opDat5SharedIndirection[i1];
}





inline void res_calc_modified(__local float *x1,__local float *x2,__local float *q1,__local float *q2,__local float *adt1,__local float *adt2,float *res1,float *res2, __constant float* gm1, __constant float* eps)
{
  float dx;
  float dy;
  float mu;
  float ri;
  float p1;
  float vol1;
  float p2;
  float vol2;
  float f;
  dx = (x1[0] - x2[0]);
  dy = (x1[1] - x2[1]);
  ri = (1.0f / q1[0]);
  p1 = (*gm1 * (q1[3] - ((0.5f * ri) * ((q1[1] * q1[1]) + (q1[2] * q1[2])))));
  vol1 = (ri * ((q1[1] * dy) - (q1[2] * dx)));
  ri = (1.0f / q2[0]);
  p2 = (*gm1 * (q2[3] - ((0.5f * ri) * ((q2[1] * q2[1]) + (q2[2] * q2[2])))));
  vol2 = (ri * ((q2[1] * dy) - (q2[2] * dx)));
  mu = ((0.5f * ( *adt1 +  *adt2)) * *eps);
  f = ((0.5f * ((vol1 * q1[0]) + (vol2 * q2[0]))) + (mu * (q1[0] - q2[0])));
  res1[0] += f;
  res2[0] -= f;
  f = ((0.5f * ((((vol1 * q1[1]) + (p1 * dy)) + (vol2 * q2[1])) + (p2 * dy))) + (mu * (q1[1] - q2[1])));
  res1[1] += f;
  res2[1] -= f;
  f = ((0.5f * ((((vol1 * q1[2]) - (p1 * dx)) + (vol2 * q2[2])) - (p2 * dx))) + (mu * (q1[2] - q2[2])));
  res1[2] += f;
  res2[2] -= f;
  f = ((0.5f * ((vol1 * (q1[3] + p1)) + (vol2 * (q2[3] + p2)))) + (mu * (q1[3] - q2[3])));
  res1[3] += f;
  res2[3] -= f;
}

__kernel void res_calc_kernel(__global float *opDat1,__global float *opDat3,__global float *opDat5,__global float *opDat7,__global int *ind_maps1,__global int *ind_maps3,__global int *ind_maps5,__global int *ind_maps7,__global short *mappingArray1,__global short *mappingArray2,__global short *mappingArray3,__global short *mappingArray4,__global short *mappingArray5,__global short *mappingArray6,__global short *mappingArray7,__global short *mappingArray8,__global int *pindSizes,__global int *pindOffs,__global int *pblkMap,__global int *poffset,__global int *pnelems,__global int *pnthrcol,__global int *pthrcol,__private int blockOffset,__local float *shared_res_calc,__constant float *gm1,__constant float *eps)
{
  //printf("EXECUTING RES_CALC\n");
  float opDat7Local[4];
  float opDat8Local[4];
  int sharedMemoryOffset;
  int numberOfActiveThreads;
  int nbytes;
  int blockID;
  int i1;
  __global int *opDat1IndirectionMap;
  __global int *opDat3IndirectionMap;
  __global int *opDat5IndirectionMap;
  __global int *opDat7IndirectionMap;
  __local int opDat1SharedIndirectionSize;
  __local int opDat3SharedIndirectionSize;
  __local int opDat5SharedIndirectionSize;
  __local int opDat7SharedIndirectionSize;
  __local float *opDat1SharedIndirection;
  __local float *opDat3SharedIndirection;
  __local float *opDat5SharedIndirection;
  __local float *opDat7SharedIndirection;
  int numOfColours;
  int numberOfActiveThreadsCeiling;
  int colour1;
  int colour2;
  int i2;
  if (get_local_id(0) == 0) {
    blockID = pblkMap[get_group_id(0) + blockOffset];
    numberOfActiveThreads = pnelems[blockID];
    numberOfActiveThreadsCeiling = get_local_size(0) * (1 + (numberOfActiveThreads - 1) / get_local_size(0));
    numOfColours = pnthrcol[blockID];
    sharedMemoryOffset = poffset[blockID];
    opDat1SharedIndirectionSize = pindSizes[0 + blockID * 4];
    opDat3SharedIndirectionSize = pindSizes[1 + blockID * 4];
    opDat5SharedIndirectionSize = pindSizes[2 + blockID * 4];
    opDat7SharedIndirectionSize = pindSizes[3 + blockID * 4];
    opDat1IndirectionMap = ind_maps1 + pindOffs[0 + blockID * 4];
    opDat3IndirectionMap = ind_maps3 + pindOffs[1 + blockID * 4];
    opDat5IndirectionMap = ind_maps5 + pindOffs[2 + blockID * 4];
    opDat7IndirectionMap = ind_maps7 + pindOffs[3 + blockID * 4];
    nbytes = 0;
    opDat1SharedIndirection = ((&shared_res_calc[nbytes/sizeof(float)]));
    nbytes += ROUND_UP(opDat1SharedIndirectionSize * (sizeof(float ) * 2));
    opDat3SharedIndirection = ((&shared_res_calc[nbytes/sizeof(float)]));
    nbytes += ROUND_UP(opDat3SharedIndirectionSize * (sizeof(float ) * 4));
    opDat5SharedIndirection = ((&shared_res_calc[nbytes/sizeof(float)]));
    nbytes += ROUND_UP(opDat5SharedIndirectionSize * (sizeof(float ) * 1));
    opDat7SharedIndirection = ((&shared_res_calc[nbytes/sizeof(float)]));
  }
  //printf("RES_CALC BARRIER 1\n");
  barrier(CLK_LOCAL_MEM_FENCE);
  for (i1 = get_local_id(0); i1 < opDat1SharedIndirectionSize * 2; i1 += get_local_size(0)) {
    opDat1SharedIndirection[i1] = opDat1[i1 % 2 + opDat1IndirectionMap[i1 / 2] * 2];
  }
  for (i1 = get_local_id(0); i1 < opDat3SharedIndirectionSize * 4; i1 += get_local_size(0)) {
    opDat3SharedIndirection[i1] = opDat3[i1 % 4 + opDat3IndirectionMap[i1 / 4] * 4];
  }
  for (i1 = get_local_id(0); i1 < opDat5SharedIndirectionSize * 1; i1 += get_local_size(0)) {
    opDat5SharedIndirection[i1] = opDat5[i1 % 1 + opDat5IndirectionMap[i1 / 1] * 1];
  }
  for (i1 = get_local_id(0); i1 < opDat7SharedIndirectionSize * 4; i1 += get_local_size(0)) {
    opDat7SharedIndirection[i1] = 0.00000F;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (i1 = get_local_id(0); i1 < numberOfActiveThreadsCeiling; i1 += get_local_size(0)) {
    colour2 = -1;
    if (i1 < numberOfActiveThreads) {
      for (i2 = 0; i2 < 4; ++i2) {
        opDat7Local[i2] = 0.00000F;
      }
      for (i2 = 0; i2 < 4; ++i2) {
        opDat8Local[i2] = 0.00000F;
      }
      res_calc_modified(opDat1SharedIndirection + mappingArray1[i1 + sharedMemoryOffset] * 2,opDat1SharedIndirection + mappingArray2[i1 + sharedMemoryOffset] * 2,opDat3SharedIndirection + mappingArray3[i1 + sharedMemoryOffset] * 4,opDat3SharedIndirection + mappingArray4[i1 + sharedMemoryOffset] * 4,opDat5SharedIndirection + mappingArray5[i1 + sharedMemoryOffset] * 1,opDat5SharedIndirection + mappingArray6[i1 + sharedMemoryOffset] * 1,opDat7Local,opDat8Local,gm1,eps);
      colour2 = pthrcol[i1 + sharedMemoryOffset];
    }
    for (colour1 = 0; colour1 < numOfColours; ++colour1) {
      if (colour2 == colour1) {
        for (i2 = 0; i2 < 4; ++i2) {
          opDat7SharedIndirection[i2 + mappingArray7[i1 + sharedMemoryOffset] * 4] += opDat7Local[i2];
        }
        for (i2 = 0; i2 < 4; ++i2) {
          opDat7SharedIndirection[i2 + mappingArray8[i1 + sharedMemoryOffset] * 4] += opDat8Local[i2];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  for (i1 = get_local_id(0); i1 < opDat7SharedIndirectionSize * 4; i1 += get_local_size(0)) {
    opDat7[i1 % 4 + opDat7IndirectionMap[i1 / 4] * 4] += opDat7SharedIndirection[i1];
  }
  //printf("FINISHED EXECUTING RES_CALC\n");
}

inline void save_soln_modified(float *q,float *qold)
{
  for (int n = 0; n < 4; n++) 
    qold[n] = q[n];
}

__kernel void save_soln_kernel(__global float *opDat1,__global float *opDat2,int sharedMemoryOffset,int setSize,__local char *shared_save_soln)
{
  float opDat1Local[4];
  float opDat2Local[4];
  __local char *sharedPointer_save_soln;
  int i1;
  int i2;
  int localOffset;
  int numberOfActiveThreads;
  int threadID;
  threadID = get_local_id(0) % OP_WARPSIZE;
  sharedPointer_save_soln = shared_save_soln + sharedMemoryOffset * (get_local_id(0) / OP_WARPSIZE);
  for (i1 = get_global_id(0); i1 < setSize; i1 += get_global_size(0)) {
    localOffset = i1 - threadID;
    numberOfActiveThreads = MIN(OP_WARPSIZE,setSize - localOffset);
    for (i2 = 0; i2 < 4; ++i2) {
      ((float *)sharedPointer_save_soln)[threadID + i2 * numberOfActiveThreads] = opDat1[threadID + i2 * numberOfActiveThreads + localOffset * 4];
    }
    for (i2 = 0; i2 < 4; ++i2) {
      opDat1Local[i2] = ((float *)sharedPointer_save_soln)[i2 + threadID * 4];
    }
    save_soln_modified(opDat1Local,opDat2Local);
    for (i2 = 0; i2 < 4; ++i2) {
      ((float *)sharedPointer_save_soln)[i2 + threadID * 4] = opDat2Local[i2];
    }
    for (i2 = 0; i2 < 4; ++i2) {
      opDat2[threadID + i2 * numberOfActiveThreads + localOffset * 4] = ((float *)sharedPointer_save_soln)[threadID + i2 * numberOfActiveThreads];
    }
  }
}

inline void update_modified(float *qold,float *q,float *res,__global float *adt,float *rms)
{
  float del;
  float adti;
  adti = (1.0f /  *adt);
  for (int n = 0; n < 4; n++) {
    del = (adti * res[n]);
    q[n] = (qold[n] - del);
    res[n] = 0.0f;
     *rms += (del * del);
  }
}

__kernel void update_kernel(__global float *opDat1,__global float *opDat2,__global float *opDat3,__global float *opDat4,__global float *reductionArrayDevice5,int sharedMemoryOffset,int setSize,__local float *shared_update)
{
  float opDat1Local[4];
  float opDat2Local[4];
  float opDat3Local[4];
  __local float *sharedPointer_update;
  float opDat5Local[1];
  __local float reductionTemporaryArray5[2048];
  int i1;
  int i2;
  int localOffset;
  int numberOfActiveThreads;
  int threadID;
  threadID = get_local_id(0) % OP_WARPSIZE;
  for (i1 = 0; i1 < 1; ++i1) {
    opDat5Local[i1] = 0.00000F;
  }
  sharedPointer_update = shared_update + sharedMemoryOffset * (get_local_id(0) / OP_WARPSIZE)/sizeof(float);
  for (i1 = get_global_id(0); i1 < setSize; i1 += get_global_size(0)) {
    localOffset = i1 - threadID;
    numberOfActiveThreads = MIN(OP_WARPSIZE,setSize - localOffset);
    for (i2 = 0; i2 < 4; ++i2) {
      (sharedPointer_update)[threadID + i2 * numberOfActiveThreads] = opDat1[threadID + i2 * numberOfActiveThreads + localOffset * 4];
    }
    for (i2 = 0; i2 < 4; ++i2) {
      opDat1Local[i2] = (sharedPointer_update)[i2 + threadID * 4];
    }
    for (i2 = 0; i2 < 4; ++i2) {
      (sharedPointer_update)[threadID + i2 * numberOfActiveThreads] = opDat3[threadID + i2 * numberOfActiveThreads + localOffset * 4];
    }
    for (i2 = 0; i2 < 4; ++i2) {
      opDat3Local[i2] = (sharedPointer_update)[i2 + threadID * 4];
    }
    update_modified(opDat1Local,opDat2Local,opDat3Local,opDat4 + i1,opDat5Local);
    for (i2 = 0; i2 < 4; ++i2) {
      (sharedPointer_update)[i2 + threadID * 4] = opDat2Local[i2];
    }
    for (i2 = 0; i2 < 4; ++i2) {
      opDat2[threadID + i2 * numberOfActiveThreads + localOffset * 4] = (sharedPointer_update)[threadID + i2 * numberOfActiveThreads];
    }
    for (i2 = 0; i2 < 4; ++i2) {
      (sharedPointer_update)[i2 + threadID * 4] = opDat3Local[i2];
    }
    for (i2 = 0; i2 < 4; ++i2) {
      opDat3[threadID + i2 * numberOfActiveThreads + localOffset * 4] = (sharedPointer_update)[threadID + i2 * numberOfActiveThreads];
    }
  }
  for (i1 = 0; i1 < 1; ++i1) {
    ReductionFloat4(&reductionArrayDevice5[i1 + get_group_id(0) * 1],opDat5Local[i1],0,reductionTemporaryArray5);
  }
}

/*
__kernel void update_kernel(
  __global float *opDat1,
  __global float *opDat2,
  __global float *opDat3,
  __global float *opDat4,
  __global float *reductionArrayDevice5,
  int   sharedMemoryOffset,
  int   setSize,
  __local  float *shared_update ) {

  float opDat1Local[4];
  float opDat2Local[4];
  float opDat3Local[4];
  float opDat5Local[1];
  int i1;
  int i2;

  for (i1 = 0; i1 < 1; ++i1) 
    opDat5Local[i1] = 0.00000F;


  int   tid = get_local_id(0)%OP_WARPSIZE;

  __local float *sharedPointer_update =  shared_update+ sharedMemoryOffset *(get_local_id(0)/OP_WARPSIZE)/sizeof(float);

  // process set elements
  for (int n=get_global_id(0); n<setSize; n+=get_global_size(0)) {

    int offset = n - tid;
    int nelems = MIN(OP_WARPSIZE,setSize-offset);

    // copy data into shared memory, then into local
    for (int m=0; m<4; m++)
      sharedPointer_update[tid+m*nelems] = opDat1[tid+m*nelems+offset*4];

    for (int m=0; m<4; m++)
      opDat1Local[m] = sharedPointer_update[m+tid*4];

    for (int m=0; m<4; m++)
      sharedPointer_update[tid+m*nelems] = opDat3[tid+m*nelems+offset*4];

    for (int m=0; m<4; m++)
      opDat3Local[m] = sharedPointer_update[m+tid*4];


    // user-supplied kernel call
    update_modified( opDat1Local,
            opDat2Local,
            opDat3Local,
            opDat4+n,
            opDat5Local );
    // copy back into shared memory, then to device
    for (int m=0; m<4; m++)
      sharedPointer_update[m+tid*4] = opDat2Local[m];

    for (int m=0; m<4; m++)
      opDat2[tid+m*nelems+offset*4] = sharedPointer_update[tid+m*nelems];

    for (int m=0; m<4; m++)
      sharedPointer_update[m+tid*4] = opDat3Local[m];

    for (int m=0; m<4; m++)
      opDat3[tid+m*nelems+offset*4] = sharedPointer_update[tid+m*nelems];


  }

  for(int d=0; d<1; d++)
    ReductionFloat4(&reductionArrayDevice5[d+get_group_id(0)*1],opDat5Local[d],0, shared_update);
}*/
