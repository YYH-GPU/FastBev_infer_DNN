cmake_minimum_required(VERSION 3.10)
project(fastbev_cpp_infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn videoio)
find_package(nlohmann_json QUIET)

# ---- OpenMP ----
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP enabled")
else()
    message(STATUS "OpenMP not found — Feature→BEV will run single-threaded")
endif()

# ---- CUDA ----
find_package(CUDA QUIET)
if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2")
    message(STATUS "CUDA found — GPU backprojection enabled")
else()
    message(STATUS "CUDA not found — GPU backprojection disabled")
endif()

# ---- Common library sources (shared by all executables) ----
set(LIB_SOURCES
    infer/fastbev_infer.cpp
    loader/nuscenes_loader.cpp
    backproject/backproject.cpp
    postprocess/postprocess.cpp
    visualize/visualize.cpp
    optimize/bev_pool.cpp
)

# ---- Single-inference executable ----
add_executable(fastbev_infer main_single.cpp ${LIB_SOURCES})

# ---- Video-inference executable ----
add_executable(fastbev_video main_video.cpp ${LIB_SOURCES})

if(CUDA_FOUND)
    # Enable OpenMP in host code within .cu files
    if(OpenMP_CXX_FOUND)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")
    endif()
    # Compile CUDA kernel with WITH_CUDA define so the dispatcher can find it
    cuda_add_library(bev_pool_cuda optimize/bev_pool_cuda.cu)
    target_include_directories(bev_pool_cuda PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/optimize
    )
    target_compile_definitions(bev_pool_cuda PRIVATE WITH_CUDA)
    target_compile_definitions(fastbev_infer PRIVATE WITH_CUDA)
    target_compile_definitions(fastbev_video PRIVATE WITH_CUDA)
    target_link_libraries(fastbev_infer PRIVATE bev_pool_cuda ${CUDA_LIBRARIES})
    target_link_libraries(fastbev_video PRIVATE bev_pool_cuda ${CUDA_LIBRARIES})
endif()

# ---- Includes (both targets) ----
foreach(TGT fastbev_infer fastbev_video)
    target_include_directories(${TGT} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/infer
        ${CMAKE_CURRENT_SOURCE_DIR}/loader
        ${CMAKE_CURRENT_SOURCE_DIR}/backproject
        ${CMAKE_CURRENT_SOURCE_DIR}/postprocess
        ${CMAKE_CURRENT_SOURCE_DIR}/visualize
        ${CMAKE_CURRENT_SOURCE_DIR}/optimize
    )

    # ---- Link ----
    if(nlohmann_json_FOUND)
        target_link_libraries(${TGT} PRIVATE
            ${OpenCV_LIBS}
            nlohmann_json::nlohmann_json
        )
    else()
        target_link_libraries(${TGT} PRIVATE ${OpenCV_LIBS})
        message(STATUS "nlohmann_json CMake package not found.")
        message(STATUS "Install: sudo apt install nlohmann-json3-dev")
    endif()
endforeach()
