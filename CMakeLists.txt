cmake_minimum_required(VERSION 3.10)

# set the project name
project(simple_palyer)

#set(CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -DLINUX -DMFX_VA -DLIBVA_SUPPORT -DLIBVA_DRM_SUPPORT -DLIBVA_X11_SUPPORT")


# add the executable
add_executable(simple_player simple_payer/simple_player.cpp
                            simple_payer/opencl_filter_va.cpp
                            simple_payer/opencl_filter.cpp
                            sample_common/src/avc_bitstream.cpp
                            sample_common/src/avc_nal_spl.cpp
                            sample_common/src/avc_spl.cpp
                            sample_common/src/base_allocator.cpp
                            sample_common/src/decode_render.cpp
                            sample_common/src/general_allocator.cpp
                            sample_common/src/sysmem_allocator.cpp
                            sample_common/src/sample_utils.cpp
                            sample_common/src/vaapi_allocator.cpp
                            sample_common/src/vaapi_device.cpp
                            sample_common/src/vaapi_utils.cpp
                            sample_common/src/vaapi_utils_x11.cpp
                            sample_common/src/vaapi_utils_drm.cpp
                            sample_common/src/vm/time_linux.cpp
                            sample_common/src/vm/atomic_linux.cpp
                            )

target_include_directories(simple_player PUBLIC
    "/usr/include/libdrm"
    "/opt/intel/openvino_2021/inference_engine/include/"
    "/opt/intel/mediasdk/include"
    "/opt/intel/mediasdk/include/mfx/"
    "simple_payer"
    "sample_common/include"
    "sample_common/include/vm")

#find_package(InferenceEngine 2.1 REQUIRED)
#find_library(INPUT_LIBRARYS NAMES inference_engine dl ngraph mfx OpenCL PATHS /opt/intel/openvino_2021/inference_engine/lib/intel64/
#/opt/intel/openvino_2021/inference_engine/external/tbb/lib/
#/opt/intel/openvino_2021/deployment_tools/ngraph/lib/
#/opt/intel/mediasdk/lib64/
#)

#find_library(INPUT_LIBRARYS inference_engine PATH /opt/intel/openvino_2021/inference_engine/lib/intel64)
#find_library(INPUT_LIBRARYS ngraph PATH /opt/intel/openvino_2021/deployment_tools/ngraph/lib)
#find_library(INPUT_LIBRARYS mfx PATH /opt/intel/mediasdk/lib64)
#find_library(INPUT_LIBRARYS OpenCL PATH /usr/lib/x86_64-linux-gnu)

target_link_libraries(simple_player PUBLIC  ${INPUT_LIBRARYS} ${CMAKE_DL_LIBS}
"/opt/intel/openvino_2021/inference_engine/lib/intel64/libinference_engine.so"
"/opt/intel/openvino_2021/inference_engine/external/tbb/lib/libtbb.so"
"/opt/intel/openvino_2021/deployment_tools/ngraph/lib/libngraph.so"
"/usr/lib/x86_64-linux-gnu/libOpenCL.so"
"/opt/intel/mediasdk/lib64/libmfx.so"
)
#target_link_libraries(Test PUBLIC
#    "/opt/intel/openvino_2021/inference_engine/lib/intel64/libinference_engine.so"
#    "/opt/intel/openvino_2021/inference_engine/lib/intel64/libinference_engine.so"
#    "/opt/intel/openvino_2021/inference_engine/external/tbb/lib/libtbb.so"
#    "/opt/intel/openvino_2021/deployment_tools/ngraph/lib/libngraph.so" )

