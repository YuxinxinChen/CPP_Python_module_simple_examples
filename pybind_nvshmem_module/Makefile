# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See COPYRIGHT.txt for license information


NVSHMEM_HOME ?= /ccs/home/yuxinc/pkg/nvshmem_2.2.1_build
CUDA_HOME ?= /sw/summit/cuda/11.4.2
MPI_HOME ?= /sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-11.1.0/spectrum-mpi-10.4.0.3-20210112-6kg6anupjriji6pnvijebfn7ha5vsqp2 
MPISUPPORT ?= 1
DEBUG ?= 0
VERBOSE ?= 1

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

CUDA_INC ?= $(CUDA_HOME)/include
NVCC ?= $(CUDA_HOME)/bin/nvcc

# Better define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
NVCC_GENCODE ?= -gencode=arch=compute_70,code=sm_70
NVCC_CG_GENCODE ?= -gencode=arch=compute_70,code=sm_70

TESTCUFLAGS  := -ccbin $(CXX) 

# Use addprefix so that we can specify more than one path
TESTLDFLAGS :=

TESTINC := -I$(CUDA_INC) -I$(NVSHMEM_HOME)/include

ifeq ($(MPISUPPORT), 1)
MPI_LIB = -lmpi
TESTINC += -I$(MPI_HOME)/include -DENABLE_MPI_SUPPORT
endif

ifeq ($(DEBUG), 0)
TESTCUFLAGS  += -O3 
else
TESTCUFLAGS  += -O0 -g -G -lineinfo
endif

ifneq ($(VERBOSE), 0)
TESTCUFLAGS  += -lineinfo -Xptxas -v --expt-extended-lambda -Xcompiler -fopenmp -ccbin g++
endif


TESTLDFLAGS += -L$(NVSHMEM_HOME)/lib -lnvshmem -lnvidia-ml -lcuda -lcudart -L$(CUDA_HOME)/lib64 -lcudart -lgomp
ifeq ($(MPISUPPORT), 1)
TESTLDFLAGS += -L$(MPI_HOME)/lib $(MPI_LIB)
endif
 
.PHONY : default 
default : examples

EXAMPLECUSRCFILES := nvshmem_test.cu 

CUPERFBIN   := $(patsubst %.cu, %, $(filter %.cu, $(EXAMPLECUSRCFILES)))
CXXPERFBIN  := $(patsubst %.cpp, %, $(filter %.cpp, $(PERFCXXSRCFILES)))
TESTBIN1 := $(patsubst %.cu, %, $(filter %.cu, $(EXAMPLECUSRCFILES))) 

$(info $(CUPERFBIN))
$(info $(CXXPERFBIN))
$(info $(TESTBIN1))

examples : $(TESTBIN1) 

% : %.cu
	@printf "Compiling %-25s > %-25s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCC_GENCODE) $(TESTCUFLAGS) $(TESTINC) -rdc=true $< -o $@ $(TESTLDFLAGS)

clean : 
	rm -rf $(TESTBIN1)
