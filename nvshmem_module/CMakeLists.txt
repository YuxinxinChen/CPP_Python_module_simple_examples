cmake_minimum_required (VERSION 3.9)

project(nvshmem_example LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Development)
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")
find_package(Torch REQUIRED)
SET(MPISUPPORT 1)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

find_package(Nvshmem REQUIRED)

add_library(nvshmem_example SHARED
  nvshmem_init.cu
)
target_compile_features(nvshmem_example PRIVATE cxx_std_11)
target_link_libraries(nvshmem_example PRIVATE Torch Python::Python)
target_link_libraries(nvshmem_example PRIVATE Nvshmem)

#target_include_directories(nvshmem_example PRIVATE
#  ${NVSHMEM_INCLUDE_DIRECTORIES}
#)
#target_link_libraries(nvshmem_example PRIVATE
#  ${NVSHMEM_LIBRARIES}
#)

# Use if the default GCC version gives issues
target_compile_options(nvshmem_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ccbin g++-9>)

set_target_properties( nvshmem_example
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Use a variant of this if you're on an earlier cmake than 3.18
# target_compile_options(nvshmem_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)
