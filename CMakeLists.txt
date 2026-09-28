cmake_minimum_required(VERSION 3.16)
project(lingmo-camera VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

include(GNUInstallDirs)

find_package(Qt6 REQUIRED COMPONENTS Core DBus)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio objdetect dnn)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SYSTEMD REQUIRED IMPORTED_TARGET libsystemd)

add_executable(lingmo-camera
    src/main.cpp
    src/camera.cpp
    src/pipeline.cpp
    src/framer.cpp
    src/loopback.cpp
    src/background.cpp
)

set(MODEL_DIR ${CMAKE_INSTALL_FULL_DATADIR}/lingmo-camera)
target_compile_definitions(lingmo-camera PRIVATE
    MODEL_PATH="${MODEL_DIR}/face_detection_yunet_2023mar.onnx"
    SEGMENTATION_MODEL_PATH="${MODEL_DIR}/selfie_segmenter_landscape.tflite"
    MATTING_MODEL_PATH="${MODEL_DIR}/rvm_mobilenetv3_640x360.onnx")
target_include_directories(lingmo-camera PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lingmo-camera PRIVATE Qt6::Core Qt6::DBus ${OpenCV_LIBS})

install(TARGETS lingmo-camera RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Face login: enrolls and checks the user's face, run by Settings and by PAM (pam_exec)
add_executable(lingmo-faceauth
    src/faceauth/main.cpp
    src/faceauth/v4l2camera.cpp
    src/faceauth/faceengine.cpp
    src/faceauth/enrollment.cpp
)
set_target_properties(lingmo-faceauth PROPERTIES AUTOMOC OFF)
target_compile_definitions(lingmo-faceauth PRIVATE FACEAUTH_MODEL_DIR="${MODEL_DIR}")
target_include_directories(lingmo-faceauth PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lingmo-faceauth PRIVATE opencv_core opencv_imgproc opencv_imgcodecs opencv_objdetect opencv_dnn
                      PkgConfig::SYSTEMD)
install(TARGETS lingmo-faceauth RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# YuNet face detector (MIT, from opencv_zoo)
install(FILES data/face_detection_yunet_2023mar.onnx DESTINATION ${MODEL_DIR})
# SFace face recognition for lingmo-faceauth (Apache-2.0, from opencv_zoo)
install(FILES data/face_recognition_sface_2021dec.onnx DESTINATION ${MODEL_DIR})
# Background blur: MediaPipe's selfie segmenter, landscape (Apache-2.0, from Google)
install(FILES data/selfie_segmenter_landscape.tflite DESTINATION ${MODEL_DIR})
# and Robust Video Matting, MobileNetV3, fixed to 640x360 for OpenCV (GPL-3.0; tools/rvm-for-opencv.py)
install(FILES data/rvm_mobilenetv3_640x360.onnx DESTINATION ${MODEL_DIR})
# Session daemon
install(FILES data/lingmo-camera.desktop DESTINATION ${CMAKE_INSTALL_FULL_SYSCONFDIR}/xdg/autostart)
# The virtual camera: loaded at boot, with a second device left for OBS
install(FILES data/modprobe.conf DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/modprobe.d RENAME lingmo-camera.conf)
install(FILES data/modules-load.conf DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/modules-load.d RENAME lingmo-camera.conf)
