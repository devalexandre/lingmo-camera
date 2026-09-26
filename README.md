# lingmo-camera

Auto framing for video calls on the Lingmo desktop. The camera follows your face and
keeps you centred, in every app: lingmo-camera reads the real webcam, finds the faces
with OpenCV's YuNet detector, moves a smooth 16:9 crop over the picture and writes the
result to a virtual camera, **Lingmo Camera** (v4l2loopback).

- The real camera is only opened while an app streams from Lingmo Camera.
- Background blur (light or strong), or an image behind you: the people are found with
  the PP-HumanSeg segmentation model on a separate thread, so the frame rate holds.
- Options live in Settings > Camera (framing on/off, zoom, which camera, background), stored in
  `~/.config/lingmoos/camera.conf` and exposed on the session bus as `com.lingmo.Camera`.
- The v4l2loopback module is set up by `/usr/lib/modprobe.d/lingmo-camera.conf`; a
  second loopback device is left free for OBS's virtual camera.

Build: `cmake -B build -DCMAKE_INSTALL_PREFIX=/usr && cmake --build build`

The face model `data/face_detection_yunet_2023mar.onnx` comes from
[opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT).

The person segmentation model `data/human_segmentation_pphumanseg_2023mar.onnx` comes from
[opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg)
(Apache-2.0, ported from PaddleSeg's PP-HumanSeg).
