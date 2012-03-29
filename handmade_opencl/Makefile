#
# set paths for header files and libraries
#
OP_BLOCK_SIZE_1 := 64

ifdef AMDAPPSDKROOT
  OPENCL		:=$(AMDAPPSDKROOT)
endif
export OPENCL
OP2_INC		= -I$(OP2_INSTALL_PATH)/c/include
OP2_LIB 	= -L$(OP2_INSTALL_PATH)/c/lib

OCLINC		:= -I$(OPENCL)/include
LIB		:= -L$(OPENCL)/lib64 
OCLLIB	:= -L/usr/lib64 -L$(OPENCL)/lib/x86_64 -lOpenCL

CPP	=	g++
CPPFLAGS 	=	-g -fPIC -DUNIX -Wall

OP2_H		:= $(OP2_INC)/op_lib_core.h $(OP2_INC)/op_lib_cpp.h $(OP2_INC)/op_opencl_rt_support.h $(OP2_INC)/op_lib_tuner.h

# 
# OpenCL version
#

airfoil_opencl: airfoil_op.cpp airfoil_kernels_ocl.o Makefile
		$(CPP) $(CPPFLAGS) -I$(INC) $(OCLINC) $(OP2_INC) -o airfoil_opencl airfoil_op.cpp airfoil_kernels_ocl.o $(OP2_LIB) $(OCLLIB) -lop2_tuner -lop2_opencl

airfoil_kernels_ocl.o:	airfoil_kernels.cl.cpp      \
		save_soln_kernel.cl.cpp \
		adt_calc_kernel.cl.cpp  \
		res_calc_kernel.cl.cpp  \
		bres_calc_kernel.cl.cpp \
		update_kernel.cl.cpp    \
                Makefile 
		$(CPP) $(CFLAGS) $(OCLINC) $(OP2_INC) -c -o airfoil_kernels_ocl.o airfoil_kernels.cl.cpp

single: airfoil_opencl
	-ln -fs kernels_single.cl kernels.cl

double: airfoil_opencl
	-ln -fs kernels_double.cl kernels.cl

vector: airfoil_opencl
	-ln -fs kernels_vector.cl kernels.cl

single_fixediter: airfoil_opencl
	-ln -fs kernels_single.fixediterations.cl kernels.cl

single_coalescing: airfoil_opencl
	-ln -fs kernels_single.working_coalescing.cl kernels.cl

single_noncoalescing: airfoil_opencl
	-ln -fs kernels_single.working_noncoalescing.cl kernels.cl

#
# cleanup
#

clean:
	-rm -f airfoil_opencl *.o
