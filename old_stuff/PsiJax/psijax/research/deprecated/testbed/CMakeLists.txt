cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(libint_interface)

# CMake will sometimes find the conda python version
# (rather the the python inside the sss environment)
set(PYTHON_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python3)

# Find PyBind11
set(PYBIND11_CPP_STANDARD -std=c++11)
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11: ${pybind11_CONFIG}")


pybind11_add_module(libint_interface libint_interface.cc)
set(CMAKE_CXX_FLAGS "-fopenmp")

#find_package(TargetLAPACK REQUIRED)
#target_link_libraries(core PRIVATE tgt::lapack)

find_package(Libint2 REQUIRED)

PYBIND11_ADD_MODULE(libint_interface libint_interface.cc)
TARGET_LINK_LIBRARIES(pyface dlib ${OpenCV_LIBS})


add_subdirectory(/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11)
