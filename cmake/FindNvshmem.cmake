# FindNvshmem
# -------
#
# Finds the NVSHMEM library
#
# This will define the following variables:
#
#   NVSHMEM_FOUND    	       			- True if the system has the Torch library
#   NVSHMEM_INCLUDE_DIRECTORIES  	- The include directories for torch
#   NVSHMEM_LIBRARIES 						- Libraries to link to
#
# and the following imported targets::
#
#   Nvshmem

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/FindCUDA")

STRING(REPLACE "\"" "" NVSHMEM_HOME_WITHOUT_QUOTES $ENV{NVSHMEM_HOME})

FIND_PATH(NVSHMEM_DIR include/nvshmem.h 
  DOC "Path to nvshmem include directory"
  HINTS
    ENV ${NVSHMEMHOME}
    ENV ${NVSHMEM_HOME}
    ${NVSHMEM_HOME_WITHOUT_QUOTES}
  )

SET(NVSHMEM_LIBDIR "${NVSHMEM_DIR}/lib")

FIND_LIBRARY(NVSHMEM_LIBRARY nvshmem
  PATHS "${NVSHMEM_LIBDIR}"
    NO_DEFAULT_PATH
  )

SET(NVSHMEM_INCLUDE_DIRECTORIES
  "${NVSHMEM_DIR}/include"
  )

SET(NVSHMEM_LIBRARIES "${NVSHMEM_LIBRARY}")
#INCLUDE(FindPackageHandleStandardArgs)
INCLUDE(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

FIND_PACKAGE(CUDA)
IF(CUDA_FOUND)
  ADD_DEFINITIONS(-DCUDA_FOUND)
  LIST(APPEND NVSHMEM_INCLUDE_DIRECTORIES ${CUDA_TOOLKIT_INCLUDE})
	LIST(APPEND NVSHMEM_LIBRARIES -L"${CUDA_TOOLKIT_ROOT_DIR}/lib" nvidia-ml cuda cudart)
ELSE()
  message(ERROR "CUDA is required")
ENDIF(CUDA_FOUND)

IF(MPISUPPORT)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc|ppc)64")
    message("Detected PowerPC 64-bit architecture")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
      message("Detected little-endian PowerPC 64-bit architecture (POWER8 or later)")
      execute_process(
        COMMAND grep -q "POWER9" /proc/cpuinfo
        RESULT_VARIABLE IS_POWER9
      )
      if(IS_POWER9 EQUAL 0)
        message("Detected IBM POWER9 architecture, use libmpi_ibm instead")
        set(IBM_POWER9_FOUND TRUE)
      endif()
    endif()
  endif()

  if(IBM_POWER9_FOUND)
    set(MPI_LIB_NAME "mpi_ibm")
  else()
    set(MPI_LIB_NAME "mpi")
  endif()

  FIND_PATH(MPI_DIR include/mpi.h
  HINTS
    ENV ${MPI_HOME}
  )
  IF(NOT MPI_DIR)
    execute_process(COMMAND which mpirun
    RESULT_VARIABLE CMD_ERROR
    OUTPUT_VARIABLE mpirun_path)
    if(DEFINED mpirun_path)
      string(LENGTH ${mpirun_path} path_length)
      math(EXPR mpi_path_length "${path_length}-11")
      string(SUBSTRING ${mpirun_path} 0 ${mpi_path_length} MPI_DIR)
    endif(DEFINED mpirun_path)
  ENDIF()

  SET(MPI_INCLUDE "${MPI_DIR}/include")
  LIST(APPEND NVSHMEM_INCLUDE_DIRECTORIES ${MPI_INCLUDE})

  SET(MPI_LIBDIR "${MPI_DIR}/lib")
  FIND_LIBRARY(MPI_LIBRARY ${MPI_LIB_NAME}
  PATHS "${MPI_LIBDIR}"
    NO_DEFAULT_PATH
  )
  LIST(APPEND NVSHMEM_LIBRARIES "${MPI_LIBRARY}")
ENDIF(MPISUPPORT)

#Message("NVSHMEM_INCLUDE_DIRECTORIES:")
#Foreach(ele ${NVSHMEM_INCLUDE_DIRECTORIES})
#  message("${ele}")
#Endforeach()
#
#Message("NVSHMEM_LIBRARIES:")
#Foreach(ele ${NVSHMEM_LIBRARIES})
#  message("${ele}")
#Endforeach()

ADD_LIBRARY(Nvshmem SHARED IMPORTED)
IF(MSVC)
  SET_TARGET_PROPERTIES(Nvshmem PROPERTIES IMPORTED_IMPLIB "${NVSHMEM_LIBRARY}")
ELSE(MSVC)
  SET_TARGET_PROPERTIES(Nvshmem PROPERTIES IMPORTED_LOCATION "${NVSHMEM_LIBRARY}")
ENDIF(MSVC)

SET_TARGET_PROPERTIES(Nvshmem PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIRECTORIES}"
  INTERFACE_LINK_LIBRARIES "${NVSHMEM_LIBRARIES}"
)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Nvshmem DEFAULT_MSG NVSHMEM_LIBRARIES NVSHMEM_INCLUDE_DIRECTORIES)

