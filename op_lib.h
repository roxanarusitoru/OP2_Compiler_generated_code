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
// include core definitions
//

#include "op_lib_core.h"

#ifdef _OPENMP
#include <omp.h>
#endif


#include <utility>
#define __NO_STD_VECTOR //use cl::vector
#include <CL/cl.h>

//
// run-time type-checking routines
//

inline int type_error(const double *,const char *type){return strcmp(type,"double");}
inline int type_error(const float  *,const char *type){return strcmp(type,"float" );}
inline int type_error(const int    *,const char *type){return strcmp(type,"int"   );}
inline int type_error(const uint   *,const char *type){return strcmp(type,"uint"  );}
inline int type_error(const ll     *,const char *type){return strcmp(type,"ll"    );}
inline int type_error(const ull    *,const char *type){return strcmp(type,"ull"   );}
inline int type_error(const bool   *,const char *type){return strcmp(type,"bool"  );}

//
// add in user's datatypes
//

#ifdef OP_USER_DATATYPES
#include <OP_USER_DATATYPES>
#endif

//
// zero constants
//

#define ZERO_double  0.0;
#define ZERO_float   0.0f;
#define ZERO_int     0;
#define ZERO_uint    0;
#define ZERO_ll      0;
#define ZERO_ull     0;
#define ZERO_bool    0;


// identity mapping and global identifier

#define OP_ID  (op_map) NULL
#define OP_GBL (op_map) NULL

//
// external variables declared in op_lib_core.cpp
//

extern int OP_diags, OP_part_size, OP_block_size;

extern int OP_set_index,  OP_set_max,
           OP_map_index,  OP_map_max,
           OP_dat_index,  OP_dat_max,
           OP_plan_index, OP_plan_max,
                          OP_kern_max;

extern op_set    *OP_set_list;
extern op_map    *OP_map_list;
extern op_dat    *OP_dat_list;
extern op_plan   *OP_plans;
extern op_kernel *OP_kernels;

//
// OP function prototypes
//
cl_mem op_allocate_constant( void*, size_t );
cl_kernel compileKernel( const char, const char * );

extern "C"
void op_init_core(int, char **, int);
void op_init(int, char **, int);

extern "C"
op_set op_decl_set(int, char const *);

extern "C"
op_map op_decl_map(op_set, op_set, int, int *, char const *);

extern "C"
op_dat op_decl_dat_core(op_set, int, char const *, int, char *, char const *);
op_dat op_decl_dat_char(op_set, int, char const *, int, char *, char const *);

void op_decl_const_char(int, char const *, int, char *, char const *);

extern "C"
void op_arg_check(op_set, int, op_arg, int *, char const *);

extern "C"
op_arg op_arg_dat(op_dat, int, op_map, int, char const *, op_access);

extern "C"
op_arg op_arg_gbl_core(char *, int, char const *, op_access);

void op_fetch_data(op_dat);

extern "C"
void op_diagnostic_output();

extern "C"
void op_timing_output();

extern "C"
op_plan * op_plan_old_core(char const *, op_set, int, int, op_dat *,
    int *, op_map *, int *, char const **, op_access *, int, int *);

op_plan * plan(char const *, op_set, int, int, op_dat *,
    int *, op_map *, int *, char const **, op_access *, int, int *);

extern "C"
op_plan * op_plan_core(char const *, op_set, int, int, op_arg *, int, int *);

op_plan * op_plan_get(char const *, op_set, int, int, op_arg *, int, int *);

extern "C"
void op_timers(double *, double *);

extern "C"
void op_timing_realloc(int);

extern "C"
void op_exit_core();
void op_exit();

//
// templates for handling datasets and constants
//

template < class T >
op_dat op_decl_dat(op_set set, int dim, char const *type,
                               T *data, char const *name){
  if (type_error(data,type)) {
    printf("incorrect type specified for dataset \"%s\" \n",name); exit(1);
  }
  return op_decl_dat_char(set, dim, type, sizeof(T), (char *)data, name);
}

template < class T >
void op_decl_const2(char const *name, int dim, char const *type, T *data){
  if (type_error(data,type)) {
    printf("incorrect type specified for constant \"%s\" \n",name); exit(1);
  }
  op_decl_const_char(dim, type, sizeof(T), (char *)data, name);
}

template < class T >
void op_decl_const(int dim, char const *type, T *data){
  if (type_error(data,type)) {
    printf("incorrect type specified for constant in op_decl_const"); exit(1);
  }
}

template < class T >
op_arg op_arg_gbl(T *data, int dim, char const *type, op_access acc){
  if (type_error(data,type))
    return op_arg_gbl_core((char *)data, dim, "error", acc);
  else
    return op_arg_gbl_core((char *)data, dim, type, acc);
}

#define LOG_FATAL    (1)
#define LOG_ERR      (2)
#define LOG_WARN     (3)
#define LOG_INFO     (4)
#define LOG_DBG      (5)

#define DEBUG_LEVEL LOG_WARN

#ifdef DEBUG_LEVEL
#define LOG(level, ...) do {  \
                              if (level <= DEBUG_LEVEL ) { \
                                                                fprintf(stderr,"%s:%d:", __FILE__, __LINE__); \
                                                                fprintf(stderr, __VA_ARGS__); \
                                                                fprintf(stderr, "\n"); \
                                                                fflush(stderr); \
                                                            } \
                          } while (0)
#else
#define LOG(level, ...)  do { } while(0)
#endif
