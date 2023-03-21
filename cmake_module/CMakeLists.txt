cmake_minimum_required(VERSION 3.10)
project(mymath)

find_package(PythonInterp REQUIRED)
# This goes after, since it uses PythonInterp as hint
find_package(PythonLibs REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# This comes to hand if we also need to use the NumPy C API
exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"import numpy; print(numpy.get_include())\""
             OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
             RETURN_VALUE NUMPY_NOT_FOUND
            )
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()

add_library(mymath SHARED mymath.c)
target_link_libraries(mymath Python::Python Python::Module)

set_target_properties(
    mymath
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "mymath"
        LINKER_LANGUAGE C
    )
