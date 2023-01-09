# Copyright (c) 2017-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# FindTorch
# -------
#
# Finds the Torch library
#
# This will define the following variables:
#
#   TORCH_FOUND    	       			- True if the system has the Torch library
#   TORCH_INCLUDE_DIRECTORIES  	- The include directories for torch
#   TORCH_LIBRARIES 						- Libraries to link to
#
# and the following imported targets::
#
#   Torch

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/FindCUDA")

execute_process(COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)"
		RESULT_VARIABLE CMD_ERROR
		OUTPUT_VARIABLE torch_cmake_path)

if(DEFINED torch_cmake_path)
  string(LENGTH ${torch_cmake_path} path_length)
  math(EXPR torch_path_length "${path_length}-13")
  string(SUBSTRING ${torch_cmake_path} 0 ${torch_path_length} torch_path)  
endif(DEFINED torch_cmake_path)

FIND_PATH(PYTORCH_DIR include/torch/csrc/api/include/torch/torch.h 
  DOC "Path to pytorch include directory"
  HINTS
    ENV ${PYTORCH_HOME}
    ENV ${PYTORCH_DIRECTORY}
    ENV ${PYTORCH_DIR}
    ENV ${TORCH_HOME}
    "${torch_path}"
  )
IF(MSVC)
  SET(TORCH_LIBDIR "${PYTORCH_DIR}/torch/lib/")
ELSE(MSVC)
  SET(TORCH_LIBDIR "${PYTORCH_DIR}/lib")
ENDIF(MSVC)

FIND_LIBRARY(TORCH_LIBRARY torch
  PATHS "${TORCH_LIBDIR}"
    NO_DEFAULT_PATH
  )

FIND_LIBRARY(TORCH_CPU_LIBRARY torch_cpu
  PATHS "${TORCH_LIBDIR}"
    NO_DEFAULT_PATH
  )

FIND_LIBRARY(CAFFE2_LIB caffe2
  PATHS "${TORCH_LIBDIR}"
    NO_DEFAULT_PATH
  )

FIND_LIBRARY(C10_LIB c10
  PATHS "${TORCH_LIBDIR}"
    NO_DEFAULT_PATH
  )

SET(TORCH_INCLUDE_DIRECTORIES
  "${PYTORCH_DIR}/include"
  "${PYTORCH_DIR}/include/torch/csrc/api/include"
  "${PYTORCH_DIR}/include/ATen"
  "${PYTORCH_DIR}/include/c10"
  "${PYTORCH_DIR}/include/caffe2"
  "${PYTORCH_DIR}/include/pybind11"
  )

#INCLUDE(FindPackageHandleStandardArgs)
INCLUDE(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

### Optionally, link CUDA
FIND_PACKAGE(CUDA)
IF(CUDA_FOUND)
  ADD_DEFINITIONS(-DCUDA_FOUND)
  FIND_LIBRARY(CAFFE2_CUDA_LIB caffe2_nvrtc
    PATHS "${TORCH_LIBDIR}"
    NO_DEFAULT_PATH
  )
  FIND_LIBRARY(C10_CUDA_LIB c10_cuda
    PATHS "${TORCH_LIBDIR}"
    NO_DEFAULT_PATH
  )
  LIST(APPEND TORCH_INCLUDE_DIRECTORIES ${CUDA_TOOLKIT_INCLUDE})
  IF(MSVC)
	  LIST(APPEND CAFFE2_CUDA_LIB 
		${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib
		${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib 
		${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvrtc.lib)
  ELSE(MSVC)
	  LIST(APPEND CAFFE2_CUDA_LIB -L"${CUDA_TOOLKIT_ROOT_DIR}/lib" cuda cudart nvrtc nvToolsExt)
  ENDIF(MSVC)
ENDIF(CUDA_FOUND)

#message("TORCH_INCLUDE_DIRECTORIES:")
#foreach(ele ${TORCH_INCLUDE_DIRECTORIES})
#  message("${ele}")
#endforeach()

SET(TORCH_LIBRARIES "${TORCH_LIBRARY}")

if(TORCH_CPU_LIBRARY)
  LIST(APPEND TORCH_LIBRARIES ${TORCH_CPU_LIBRARY})
endif(TORCH_CPU_LIBRARY)

if(CAFFE2_LIB)
  LIST(APPEND TORCH_LIBRARIES ${CAFFE2_LIB})
endif(CAFFE2_LIB)

if(CAFFE2_CUDA_LIB)
  LIST(APPEND TORCH_LIBRARIES ${CAFFE2_CUDA_LIB})
endif(CAFFE2_CUDA_LIB)
if(C10_LIB)
  LIST(APPEND TORCH_LIBRARIES ${C10_LIB})
endif(C10_LIB)
if(C10_CUDA_LIB)
  LIST(APPEND TORCH_LIBRARIES ${C10_CUDA_LIB})
endif(C10_CUDA_LIB)

#message("TORCH_LIBRARIES:")
#foreach(ele ${TORCH_LIBRARIES})
#  message("${ele}")
#endforeach()

ADD_LIBRARY(Torch SHARED IMPORTED)
IF(MSVC)
  SET_TARGET_PROPERTIES(Torch PROPERTIES IMPORTED_IMPLIB "${TORCH_LIBRARY}")
ELSE(MSVC)
  SET_TARGET_PROPERTIES(Torch PROPERTIES IMPORTED_LOCATION "${TORCH_LIBRARY}")
ENDIF(MSVC)

SET_TARGET_PROPERTIES(Torch PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRECTORIES}"
  INTERFACE_LINK_LIBRARIES "${TORCH_LIBRARIES}"
)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(TORCH DEFAULT_MSG TORCH_LIBRARIES TORCH_INCLUDE_DIRECTORIES)
