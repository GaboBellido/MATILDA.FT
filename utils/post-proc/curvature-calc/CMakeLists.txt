cmake_minimum_required(VERSION 3.10)
project(curvature-calc)

set(VTK_DIR /home/gabriel/Install/vtk/build)
set(Eigen3_DIR /usr/share/eigen3/cmake)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(VTK REQUIRED)


add_executable(curvature-calc curvature-calc.cpp)

target_link_libraries(curvature-calc PRIVATE ${VTK_LIBRARIES} Eigen3::Eigen)