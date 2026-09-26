cmake_minimum_required(VERSION 3.16)
project(lingmo-camera VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

include(GNUInstallDirs)

find_package(Qt6 REQUIRED COMPONENTS Core DBus)
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio objdetect dnn)

add_executable(lingmo-camera
    src/main.cpp
    src/camera.cpp
    src/pipeline.cpp
    src/framer.cpp
    src/loopback.cpp
)

set(MODEL_DIR ${CMAKE_INSTALL_FULL_DATADIR}/lingmo-camera)
target_compile_definitions(lingmo-camera PRIVATE
    MODEL_PATH="${MODEL_DIR}/face_detection_yunet_2023mar.onnx")
target_include_directories(lingmo-camera PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lingmo-camera PRIVATE Qt6::Core Qt6::DBus ${OpenCV_LIBS})

install(TARGETS lingmo-camera RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# YuNet face detector (MIT, from opencv_zoo)
install(FILES data/face_detection_yunet_2023mar.onnx DESTINATION ${MODEL_DIR})
# Session daemon
install(FILES data/lingmo-camera.desktop DESTINATION ${CMAKE_INSTALL_FULL_SYSCONFDIR}/xdg/autostart)
# The virtual camera: loaded at boot, with a second device left for OBS
install(FILES data/modprobe.conf DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/modprobe.d RENAME lingmo-camera.conf)
install(FILES data/modules-load.conf DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/modules-load.d RENAME lingmo-camera.conf)
