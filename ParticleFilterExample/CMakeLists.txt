cmake_minimum_required(VERSION 3.16.2)
project(Uebungen)
set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(main
src/main.cpp
include/helpers.h)
# target_include_directories(base PRIVATE)
# # add_executable(main main.cpp)
# set(SOURCES main.cpp help_functions.h)
# add_executable(base ${SOURCES})

# ## How to add header files to multiple targets
# # set(HEADER_FILES help_functions.h)

# # # add_library(mylib src.cpp ${HEADER_FILES})
# # # target_include_directories(mylib PRIVATE ${YOUR_DIRECTORY})
# # add_executable(main main.cpp ${HEADER_FILES})
# # target_include_directories(main PRIVATE ${YOUR_DIRECTORY})