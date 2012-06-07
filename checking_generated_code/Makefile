
OP2_INC		:= -I$(OP2_INSTALL_PATH)/c/include
OP2_LIB		:= -L$(OP2_INSTALL_PATH)/c/lib

ifdef AMDAPPSDKROOT
  OPENCL    :=$(AMDAPPSDKROOT)
endif
ifdef CUDA_HOME
  OPENCL    :=$(CUDA_HOME)
endif
export OPENCL

#OCLINC	:= -I$(OPENCL)/include 
#OCLLIB			:= -L/usr/lib64 -L$(OPENCL)/lib/x86_64 -lOpenCL
OCLINC 	:= -I/usr/include
OCLLIB	:= -L/usr/lib64/OpenCL/vendors/intel -lOpenCL

EXEC 			:= airfoil_opencl

CPP				:= g++
CPPFLAGS 	:= -g -fPIC -DUNIX -Wall

all: airfoil_opencl


airfoil_opencl: rose_airfoil_opencl.cpp rose_airfoil_opencl_hosts.o Makefile
	$(CPP) $(CPPFLAGS) $(OCLINC) $(OP2_INC) -o airfoil_opencl rose_airfoil_opencl.cpp rose_airfoil_opencl_hosts.o $(OP2_LIB) $(OCLLIB) -lop2_tuner -lop2_opencl

rose_airfoil_opencl_hosts.o: rose_opencl_hosts.cpp Makefile 
	$(CPP) $(CPPFLAGS) $(OCLINC) $(OP2_INC) -c -o rose_airfoil_opencl_hosts.o rose_opencl_hosts.cpp 

single: airfoil_opencl
	ln -fs rose_opencl_code_opencl.cl kernels.cl

clean: 
	rm -f airfoil_opencl *.o 
