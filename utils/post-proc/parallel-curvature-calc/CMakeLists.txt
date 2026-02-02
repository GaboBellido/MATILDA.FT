cmake_minimum_required(VERSION 3.10)
project(prl-curvature-calc)

set(VTK_DIR /home/gabriel/Install/vtk/build)
set(Eigen3_DIR /usr/share/eigen3/cmake)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(VTK REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(prl-curvature-calc prl-curvature-calc.cpp)

target_link_libraries(prl-curvature-calc PRIVATE ${VTK_LIBRARIES} Eigen3::Eigen OpenMP::OpenMP_CXX)
