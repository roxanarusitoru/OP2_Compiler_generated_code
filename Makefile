
OP2_INC		:= -I$(OP2_INSTALL_PATH)/c/include
OP2_LIB		:= -L$(OP2_INSTALL_PATH)/c/lib

ifdef AMDAPPSDKROOT
  OPENCL    :=$(AMDAPPSDKROOT)
endif
ifdef CUDA_HOME
  OPENCL    :=$(CUDA_HOME)
endif
export OPENCL

BLOCK_SIZE := 64
PART_SIZE := 64

ifndef BLOCK_SIZE
	BLOCK_SIZE := 256
endif

ifndef OP_WARPSIZE
	OP_WARPSIZE := 1
endif

ifndef PART_SIZE
	PART_SIZE := 256
endif

#OCLINC	:= -I$(OPENCL)/include 
#OCLLIB			:= -L/usr/lib64 -L$(OPENCL)/lib/x86_64 -lOpenCL
OCLINC 	:= -I/usr/include
OCLLIB	:= -L/usr/lib64/OpenCL/vendors/intel -lOpenCL

EXEC 			:= airfoil_opencl

CPP				:= g++
CPPFLAGS 	:= -g -fPIC -DUNIX -Wall -DOP_BLOCK_SIZE_0=$(BLOCK_SIZE) -DOP_WARPSIZE_0=$(OP_WARPSIZE) -DOP_PART_SIZE_0=$(PART_SIZE) -DOP_BLOCK_SIZE_1=$(BLOCK_SIZE) -DOP_PART_SIZE_1=$(PART_SIZE) -DOP_BLOCK_SIZE_2=$(BLOCK_SIZE) -DOP_PART_SIZE_2=$(PART_SIZE) -DOP_BLOCK_SIZE_3=$(BLOCK_SIZE) -DOP_PART_SIZE_3=$(PART_SIZE) -DOP_BLOCK_SIZE_4=$(BLOCK_SIZE) -DOP_PART_SIZE_4=$(PART_SIZE) -DOP_BLOCK_SIZE_5=$(BLOCK_SIZE) -DOP_PART_SIZE_5=$(PART_SIZE) -DOP_BLOCK_SIZE_6=$(BLOCK_SIZE) -DOP_PART_SIZE_6=$(PART_SIZE) -DOP_BLOCK_SIZE_7=$(BLOCK_SIZE) -DOP_PART_SIZE_7=$(PART_SIZE) -DOP_BLOCK_SIZE_8=$(BLOCK_SIZE) -DOP_PART_SIZE_8=$(PART_SIZE) -DOP_BLOCK_SIZE_9=$(BLOCK_SIZE) -DOP_PART_SIZE_9=$(PART_SIZE) -DOP_BLOCK_SIZE_10=$(BLOCK_SIZE) -DOP_PART_SIZE_10=$(PART_SIZE) -DOP_BLOCK_SIZE_11=$(BLOCK_SIZE) -DOP_PART_SIZE_11=$(PART_SIZE) -DOP_BLOCK_SIZE_12=$(BLOCK_SIZE) -DOP_PART_SIZE_12=$(PART_SIZE) -DOP_BLOCK_SIZE_13=$(BLOCK_SIZE) -DOP_PART_SIZE_13=$(PART_SIZE)

all: airfoil_opencl

airfoil_opencl: rose_airfoil_opencl.cpp rose_airfoil_opencl_hosts.o Makefile
	$(CPP) $(CPPFLAGS) $(OCLINC) $(OP2_INC) -o airfoil_opencl rose_airfoil_opencl.cpp rose_airfoil_opencl_hosts.o $(OP2_LIB) $(OCLLIB) -lop2_opencl

rose_airfoil_opencl_hosts.o: rose_opencl_hosts.cpp Makefile 
	$(CPP) $(CPPFLAGS) $(OCLINC) $(OP2_INC) -c -o rose_airfoil_opencl_hosts.o rose_opencl_hosts.cpp 

single: airfoil_opencl
	ln -fs rose_opencl_code_opencl.cl kernels.cl

clean: 
	rm -f airfoil_opencl *.o 
