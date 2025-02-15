// 
// auto-generated by op2.m on 30-May-2011 22:03:11 
//

/*
  Open source copyright declaration based on BSD open source template:
  http://www.opensource.org/licenses/bsd-license.php

* Copyright (c) 2009-2011, Mike Giles
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * The name of Mike Giles may not be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//
//     Nonlinear airfoil lift calculation
//
//     Written by Mike Giles, 2010-2011, based on FORTRAN code
//     by Devendra Ghate and Mike Giles, 2005
//

//
// standard headers
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sstream>

//#define cl_intel_printf 1
//#pragma OPENCL EXTENSION cl_amd_printf : enable
//#pragma OPENCL EXTENSION cl_intel_printf : enable
#include <utility>
#define __NO_STD_VECTOR //use cl::vector
#include <CL/cl.h>
// global constants

//float gam, gm1, cfl, eps, mach, alpha, qinf[4];

struct global_constants {
  float gam;
  float gm1;
  float cfl;
  float eps;
  float mach;
  float alpha;
  float qinf[4];
};

struct global_constants g_const;
cl_mem g_const_d;

#include "op_lib_tuner.h"
#include "op_opencl_rt_support.h"

//#define DIAGNOSTIC 1

//
// OP header file
//


//
// op_par_loop declarations
//

void op_par_loop_save_soln(char const *, op_set,
  op_arg,
  op_arg,
  op_tuner*);

void op_par_loop_adt_calc(char const *, op_set,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_tuner*);

void op_par_loop_res_calc(char const *, op_set,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_tuner*);

void op_par_loop_bres_calc(char const *, op_set,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_tuner*);

void op_par_loop_update(char const *, op_set,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_tuner*);

// kernel routines for parallel loops
//

/*
#include "save_soln.h"
#include "adt_calc.h"
#include "res_calc.h"
#include "bres_calc.h"
#include "update.h"
*/

// main program

void print_array( float *array, int len, const char *file ) {
  FILE *flog;
  flog = fopen( file, "w" );
  for( int i=0; i<len; ++i ) {
    fprintf( flog, "%f\n", array[i] );
  }
  fclose( flog );
}

void dump_array( op_dat dat, const char *file ) {
  printf("Dumping array %s\n", file);
  op_fetch_data( dat );
  printf("Data fetched\n");
  printf("%s with size %d\n", file, dat->set->size*dat->dim);
  print_array( ( float *) dat->data, dat->set->size*dat->dim, file );
  printf("Array printed\n");
}

int main(int argc, char **argv){

  int    *becell, *ecell,  *bound, *bedge, *edge, *cell;
  float  *x, *q, *qold, *adt, *res;

  int    nnode,ncell,nedge,nbedge,niter;
  float  rms;

  // read in grid

  printf("reading in grid \n");

  FILE *fp;
  if ( (fp = fopen("new_grid.large.dat","r")) == NULL) {
    printf("can't open file new_grid.dat\n"); exit(-1);
  }

  if (fscanf(fp,"%d %d %d %d \n",&nnode, &ncell, &nedge, &nbedge) != 4) {
    printf("error reading from new_grid.dat\n"); exit(-1);
  }

  cell   = (int *) malloc(4*ncell*sizeof(int));
  edge   = (int *) malloc(2*nedge*sizeof(int));
  ecell  = (int *) malloc(2*nedge*sizeof(int));
  bedge  = (int *) malloc(2*nbedge*sizeof(int));
  becell = (int *) malloc(  nbedge*sizeof(int));
  bound  = (int *) malloc(  nbedge*sizeof(int));

  x      = (float *) malloc(2*nnode*sizeof(float));
  q      = (float *) malloc(4*ncell*sizeof(float));
  qold   = (float *) malloc(4*ncell*sizeof(float));
  res    = (float *) malloc(4*ncell*sizeof(float));
  adt    = (float *) malloc(  ncell*sizeof(float));

  for (int n=0; n<nnode; n++) {
    if (fscanf(fp,"%f %f \n",&x[2*n], &x[2*n+1]) != 2) {
      printf("error reading from new_grid.dat\n"); exit(-1);
    }
  }

  for (int n=0; n<ncell; n++) {
    if (fscanf(fp,"%d %d %d %d \n",&cell[4*n  ], &cell[4*n+1],
                                   &cell[4*n+2], &cell[4*n+3]) != 4) {
      printf("error reading from new_grid.dat\n"); exit(-1);
    }
  }

  for (int n=0; n<nedge; n++) {
    if (fscanf(fp,"%d %d %d %d \n",&edge[2*n], &edge[2*n+1],
                                   &ecell[2*n],&ecell[2*n+1]) != 4) {
      printf("error reading from new_grid.dat\n"); exit(-1);
    }
  }

  for (int n=0; n<nbedge; n++) {
    if (fscanf(fp,"%d %d %d %d \n",&bedge[2*n],&bedge[2*n+1],
                                   &becell[n], &bound[n]) != 4) {
      printf("error reading from new_grid.dat\n"); exit(-1);
    }
  }

  fclose(fp);

#ifdef DIAGNOSTIC
  print_array((float *) x, 2*nnode, "initial_nodes");
  print_array((float *) cell, 4*ncell, "initial_cells");
 
  FILE *flog;
  flog = fopen( "initial_cells_cellarray", "w" );
  for( int i=0; i< ncell; ++i ) {
    fprintf( flog, "%d %d %d %d\n", cell[4*i], cell[4*i+1], cell[4*i+2], cell[4*i+3] );
  }
  fclose( flog );


  print_array((float *) edge, 2*nedge, "initial_edges");
  print_array((float *) ecell, 2*nedge, "initiall_edges_for_cell");
  print_array((float *) bedge, 2*nbedge, "initial_border_edges");
  print_array((float *) becell, nbedge, "initial_becell");
  print_array((float *) bound, nbedge, "initial bound");
#endif

  // set constants and initialise flow field and residual

  printf("initialising flow field \n");

  g_const.gam = 1.4f;
  g_const.gm1 = g_const.gam - 1.0f;
  g_const.cfl = 0.9f;
  g_const.eps = 0.05f;

  g_const.mach  = 0.4f;
  g_const.alpha = 3.0f*atan(1.0f)/45.0f;  
  float p     = 1.0f;
  float r     = 1.0f;
  float u     = sqrt(g_const.gam*p/r)*g_const.mach;
  float e     = p/(r*g_const.gm1) + 0.5f*u*u;

  g_const.qinf[0] = r;
  g_const.qinf[1] = r*u;
  g_const.qinf[2] = 0.0f;
  g_const.qinf[3] = r*e;

  for (int n=0; n<ncell; n++) {
    for (int m=0; m<4; m++) {
        q[4*n+m] = g_const.qinf[m];
      res[4*n+m] = 0.0f;
        qold[4*n+m] = 0.0f;
    }
  }

  // OP initialisation

  printf("OP initialisation\n");

  op_tuner* global_tuner = op_create_global_tuner();  
  global_tuner->op_warpsize = 1;
  global_tuner->block_size = 64;
  global_tuner->part_size = 128;
  global_tuner->cache_line_size = 128;

  op_init(argc,argv,2);
  g_const_d = op_allocate_constant( &g_const, sizeof( struct global_constants ) );

  // declare sets, pointers, datasets and global constants

  op_set nodes  = op_decl_set(nnode,  "nodes");
  op_set edges  = op_decl_set(nedge,  "edges");
  op_set bedges = op_decl_set(nbedge, "bedges");
  op_set cells  = op_decl_set(ncell,  "cells");

  op_map pedge   = op_decl_map(edges, nodes,2,edge,  "pedge");
  op_map pecell  = op_decl_map(edges, cells,2,ecell, "pecell");
  op_map pbedge  = op_decl_map(bedges,nodes,2,bedge, "pbedge");
  op_map pbecell = op_decl_map(bedges,cells,1,becell,"pbecell");
  op_map pcell   = op_decl_map(cells, nodes,4,cell,  "pcell");

  op_dat p_bound = op_decl_dat(bedges,1,"int"  ,bound,"p_bound");
  op_dat p_x     = op_decl_dat(nodes ,2,"float",x    ,"p_x");
  op_dat p_q     = op_decl_dat(cells ,4,"float",q    ,"p_q");
  op_dat p_qold  = op_decl_dat(cells ,4,"float",qold ,"p_qold");
  op_dat p_adt   = op_decl_dat(cells ,1,"float",adt  ,"p_adt");
  op_dat p_res   = op_decl_dat(cells ,4,"float",res  ,"p_res");

  op_decl_const2("gam",1,"float",&g_const.gam  );
  op_decl_const2("gm1",1,"float",&g_const.gm1  );
  op_decl_const2("cfl",1,"float",&g_const.cfl  );
  op_decl_const2("eps",1,"float",&g_const.eps  );
  op_decl_const2("mach",1,"float",&g_const.mach );
  op_decl_const2("alpha",1,"float",&g_const.alpha);
  op_decl_const2("qinf",4,"float",g_const.qinf  );

  op_tuner* save_soln_tuner = op_create_tuner("save_soln");
  save_soln_tuner->part_size = 64;
  save_soln_tuner->block_size = 4;

  op_tuner* adt_calc_tuner = op_create_tuner("adt_calc");
  adt_calc_tuner->part_size = 512;
  adt_calc_tuner->block_size = 512;

  op_tuner* res_calc_tuner = op_create_tuner("res_calc");
  res_calc_tuner->part_size = 512;
  res_calc_tuner->block_size = 512;

//on part_size = 512, block_size = 512 - this one seg faults - don't know why - don't have time to debug it.
  op_tuner* bres_calc_tuner = op_create_tuner("bres_calc");
  bres_calc_tuner->part_size = 256;
  bres_calc_tuner->block_size = 256;
 
  op_tuner* update_tuner = op_create_tuner("update");
  update_tuner->part_size = 64;
  update_tuner->block_size = 4;

  op_diagnostic_output();

#ifdef DIAGNOSTIC
  dump_array(p_bound, "initial_dat_p_bound");
  dump_array(p_x, "initial_dat_p_x");
  dump_array(p_q, "initial_dat_p_q");
  dump_array(p_qold, "initial_dat_p_qold");
  dump_array(p_adt, "initial_dat_p_adt");
  dump_array(p_res, "initial_dat_res");
#endif

// main time-marching loop
  niter = 1000;
//  niter = 2;
//  niter = 1;
  for(int iter=1; iter<=niter; iter++) {
//  save old flow solution

#ifdef DIAGNOSTIC
   dump_array(p_q, "p_q_iter_before");
   dump_array(p_qold, "p_q_old_iter_before");
#endif
    op_par_loop_save_soln("save_soln", cells,
                op_arg_dat(p_q,   -1,OP_ID, 4,"float",OP_READ ),
                op_arg_dat(p_qold,-1,OP_ID, 4,"float",OP_WRITE),
                save_soln_tuner);
  
#ifdef DIAGNOSTIC
    dump_array(p_q, "p_q_iter_after");
    dump_array(p_qold, "p_q_old_iter_after");
#endif

/*    if ( iter == 1 ) {
      dump_array( p_qold, "p_qold" );
    }
    */

#ifdef DIAGNOSTIC
    if (iter==1) {
      dump_array( p_qold, "p_qold" );
    }
#endif
    //dump_array( p_qold, "p_qold" );
    //op_fetch_data( p_qold );
    //print_array( ( float *) p_qold->data, 4*p_qold->set->size, "p_qold" );
//    print_array( p_q, "p_qold2" );
//    print_array( p_qold, "p_qold" );

    //assert( p_q->data[0] != 0.0f );

//  predictor/corrector update loop

  //  dump_array(p_adt, "p_adt_before");
    for(int k=0; k<2; k++) {
//    for(int k = 0; k < 1; k++) {
//    calculate area/timstep
      #ifdef DIAGNOSTIC
/*      if(k == 0 && iter == 1) {
        printf("Dumping adt before adt_calc execution array");
         op_fetch_data( p_adt );
	 float* array = (float *) p_adt->data;
         long size = p_adt->set->size;
         for(long elem = 0; elem < size; ++elem) {
           printf("%lf",array[elem]);
         }
      }*/
      #endif    

      op_par_loop_adt_calc("adt_calc",cells,
                  op_arg_dat(p_x,   0,pcell, 2,"float",OP_READ ),
                  op_arg_dat(p_x,   1,pcell, 2,"float",OP_READ ),
                  op_arg_dat(p_x,   2,pcell, 2,"float",OP_READ ),
                  op_arg_dat(p_x,   3,pcell, 2,"float",OP_READ ),
                  op_arg_dat(p_q,  -1,OP_ID, 4,"float",OP_READ ),
                  op_arg_dat(p_adt,-1,OP_ID, 1,"float",OP_WRITE),
                  adt_calc_tuner);
     #ifdef DIAGNOSTIC 
      /* if(k == 0 && iter == 1) {
        printf("Dumping adt after 1x adt_calc execution array");
         op_fetch_data( p_adt );
         float* array = (float *) p_adt->data;
         long size = p_adt->set->size;
         for(long elem = 0; elem < size; ++elem) {
           printf("%lf",array[elem]);
         }
      }*/
      #endif


#ifdef DIAGNOSTIC
    if (iter==1 && k==0) {
      dump_array( p_adt, "p_adt0" );
      dump_array( p_q, "p_q_after_adt_calc0");
    }
    if (iter==1 && k==1) {
      dump_array( p_adt, "p_adt1" );
      dump_array( p_q, "p_q_after_adt_calc1");
    }
#endif
  //  dump_array(p_adt, "p_adt_after");
//    calculate flux residual

#ifdef DIAGNOSTIC /*
      if(k == 0 && iter == 1) {
        printf("Dumping p_res before res_calc execution array");
         op_fetch_data( p_res );
         float* array = (float *) p_res->data;
         long size = p_res->set->size;
         for(long elem = 0; elem < size; ++elem) {
           printf("%lf",array[elem]);
         }
      }*/
#endif
      op_par_loop_res_calc("res_calc",edges,
                  op_arg_dat(p_x,    0,pedge, 2,"float",OP_READ),
                  op_arg_dat(p_x,    1,pedge, 2,"float",OP_READ),
                  op_arg_dat(p_q,    0,pecell,4,"float",OP_READ),
                  op_arg_dat(p_q,    1,pecell,4,"float",OP_READ),
                  op_arg_dat(p_adt,  0,pecell,1,"float",OP_READ),
                  op_arg_dat(p_adt,  1,pecell,1,"float",OP_READ),
                  op_arg_dat(p_res,  0,pecell,4,"float",OP_INC ),
                  op_arg_dat(p_res,  1,pecell,4,"float",OP_INC ),
                  res_calc_tuner);
#ifdef DIAGNOSTIC
/*if(k == 0 && iter == 1) {
        printf("Dumping p_res after res_calc execution array");
         op_fetch_data( p_res );
         float* array = (float *) p_res->data;
         long size = p_res->set->size;
         for(long elem = 0; elem < size; ++elem) {
           printf("%lf",array[elem]);
         }
      }*/
#endif

#ifdef DIAGNOSTIC
    if (iter==1 && k==0) {
      dump_array( p_res, "p_res0" );
      dump_array( p_q, "p_q_after_res_calc0");
    }
    if (iter==1 && k==1) {
      dump_array( p_res, "p_res1" );
      dump_array( p_q, "p_q_after_res_calc1");
    }
#endif/*
if(k == 0 && iter == 1) {
        printf("Dumping p_res before bres_calc execution array");
         op_fetch_data( p_res );
         float* array = (float *) p_res->data;
         long size = p_res->set->size;
         for(long elem = 0; elem < size; ++elem) {
           printf("%lf",array[elem]);
         }
      }
*/
#ifdef DIAGNOSTIC
  if (iter == 1 && k == 0) {
    dump_array(p_x, "p_x_after_res_calc0");
    dump_array(p_adt, "p_adt_after_res_calc0");
    dump_array(p_bound, "p_bound_after_res_calc0");
  }
  if (iter == 1 && k == 0) {
    dump_array(p_x, "p_x_after_res_calc1");
    dump_array(p_adt, "p_adt_after_res_calc1");
    dump_array(p_bound, "p_bound_after_res_calc1");
  }
#endif
      op_par_loop_bres_calc("bres_calc",bedges,
                  op_arg_dat(p_x,     0,pbedge, 2,"float",OP_READ),
                  op_arg_dat(p_x,     1,pbedge, 2,"float",OP_READ),
                  op_arg_dat(p_q,     0,pbecell,4,"float",OP_READ),
                  op_arg_dat(p_adt,   0,pbecell,1,"float",OP_READ),
                  op_arg_dat(p_res,   0,pbecell,4,"float",OP_INC ),
                  op_arg_dat(p_bound,-1,OP_ID  ,1,"int",  OP_READ),
                  bres_calc_tuner);
/*
if(k == 0 && iter == 1) {
        printf("Dumping p_res after bres_calc execution array");
         op_fetch_data( p_res );
         float* array = (float *) p_res->data;
         long size = p_res->set->size;
         for(long elem = 0; elem < size; ++elem) {
           printf("%lf",array[elem]);
         }
      }
*/

#ifdef DIAGNOSTIC
    if (iter==1 && k==0) {
      dump_array( p_res, "p_res_a0" );
    }
    if (iter==1 && k==1) {
      dump_array( p_res, "p_res_a1" );
    }
#endif
//    update flow field

      rms = 0.0;

      op_par_loop_update("update",cells,
                  op_arg_dat(p_qold,-1,OP_ID, 4,"double",OP_READ ),
                  op_arg_dat(p_q, -1,OP_ID, 4,"double",OP_WRITE),
                  op_arg_dat(p_res, -1,OP_ID, 4,"double",OP_RW ),
                  op_arg_dat(p_adt, -1,OP_ID, 1,"double",OP_READ ),
                  op_arg_gbl(&rms,1,"float",OP_INC),
                  update_tuner);

#ifdef DIAGNOSTIC
      if (iter==1 && k==0) {
        dump_array(p_res, "p_res_update_0");
        dump_array(p_q,"p_q_update_0");
        printf("rms_1_0 %f\n", rms);
      }
      if (iter==1 && k==1) {
        dump_array(p_res, "p_res_update_1");
        dump_array(p_q,"p_q_update_1");
        printf("rms_1_1 %f\n", rms);
      }
#endif

    }

#ifdef DIAGNOSTIC
    if (iter==1) {
      dump_array( p_q, "p_q1" );
    }
#endif

//  print iteration history
//    printf("before rms %f, ncell %d\n", rms, ncell);
//    printf("before %d %10.5e \n", iter, rms);
    rms = sqrt(rms/(float) ncell);

    if (iter%100 == 0)
      printf("after %d  %10.5e \n",iter,rms);



  }

  op_timing_output();

//#ifdef DIAGNOSTIC
  dump_array( p_q, "p_q" );
//#endif



}

