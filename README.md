# lingmo-camera

Auto framing for video calls on the Lingmo desktop. The camera follows your face and
keeps you centred, in every app: lingmo-camera reads the real webcam, finds the faces
with OpenCV's YuNet detector, moves a smooth 16:9 crop over the picture and writes the
result to a virtual camera, **Lingmo Camera** (v4l2loopback).

- The real camera is only opened while an app streams from Lingmo Camera.
- Options live in Settings > Camera (framing on/off, zoom, which camera), stored in
  `~/.config/lingmoos/camera.conf` and exposed on the session bus as `com.lingmo.Camera`.
- The v4l2loopback module is set up by `/usr/lib/modprobe.d/lingmo-camera.conf`; a
  second loopback device is left free for OBS's virtual camera.

Build: `cmake -B build -DCMAKE_INSTALL_PREFIX=/usr && cmake --build build`

The face model `data/face_detection_yunet_2023mar.onnx` comes from
[opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT).
