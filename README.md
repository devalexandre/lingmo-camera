# lingmo-camera

Auto framing for video calls on the Lingmo desktop. The camera follows your face and
keeps you centred, in every app: lingmo-camera reads the real webcam, finds the faces
with OpenCV's YuNet detector, moves a smooth 16:9 crop over the picture and writes the
result to a virtual camera, **Lingmo Camera** (v4l2loopback).

- The real camera is only opened while an app streams from Lingmo Camera.
- Background blur (light or strong), or an image behind you. Two small models look at a
  640x360 copy of the picture on a separate thread, so the frame rate holds: Robust Video
  Matting draws a soft outline that keeps the hair and the headset, and MediaPipe's selfie
  segmenter fills the inside (dark clothes on a dark chair). The mask is smoothed over time,
  the room is blurred with the person taken out (no glow), and with an image the old room's
  colour is taken out of the soft edge. It costs about 5 ms per frame on the camera thread;
  the models take ~40-50 ms per run on two threads (masks at ~20 fps on a desktop CPU).
- Options live in Settings > Camera (framing on/off, zoom, which camera, background), stored in
  `~/.config/lingmoos/camera.conf` and exposed on the session bus as `com.lingmo.Camera`.
- The v4l2loopback module is set up by `/usr/lib/modprobe.d/lingmo-camera.conf`; a
  second loopback device is left free for OBS's virtual camera.

Build: `cmake -B build -DCMAKE_INSTALL_PREFIX=/usr && cmake --build build`

The face model `data/face_detection_yunet_2023mar.onnx` comes from
[opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT).

The background models:

- `data/selfie_segmenter_landscape.tflite` is MediaPipe's selfie segmenter, landscape
  (256x144), from Google
  ([download](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite),
  [model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf),
  Apache-2.0). OpenCV reads the .tflite as it is.
- `data/rvm_mobilenetv3_640x360.onnx` is
  [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) (MobileNetV3,
  `rvm_mobilenetv3_fp32.onnx` from its v1.0.0 release, GPL-3.0), made runnable by OpenCV with
  `tools/rvm-for-opencv.py`: fixed 640x360 input and downsample ratio 0.25, simplified, and
  two OpenCV quirks worked around (ceil-mode average pooling, Tanh on large inputs). The
  output matches onnxruntime's.

## Face login: lingmo-faceauth

`lingmo-faceauth` lets the login screen, the lock screen and (optionally) sudo and polkit
recognize the user's face with the webcam. Settings > User registers the face and turns it on;
it is used through `pam_exec`, and the password always keeps working.

- `lingmo-faceauth enroll` (run by Settings as the user) looks through the real camera (the
  first V4L2 capture device that isn't a loopback, like the daemon), finds the face with YuNet,
  aligns it and keeps 5 different views as SFace features: numbers, no picture. They are saved
  in `~/.local/share/lingmoos/face/<user>.dat` (mode 0600). The login screen reads that file as
  root, the lock screen as the user.
- `lingmo-faceauth verify` looks for up to 4 seconds and exits 0 when two frames match an
  enrolled view (cosine similarity ≥ 0.40; SFace's own threshold is 0.363) and the head moved a
  little; 1 when it doesn't match, 2 without an enrolled face, 3 when there is no camera or
  another program uses it (a video call through Lingmo Camera: it fails at once instead of
  waiting), 4 in a remote (ssh) session. Its short messages ("Look at the camera…") reach the
  login screen, the polkit dialog and sudo through `pam_exec … stdout`.
- For sudo and polkit (`verify --system`) only root's copy in `/var/lib/lingmo-face` counts,
  made by Settings' face-pam helper after an administrator password, so a program running as
  the user can't swap the face that unlocks administrator rights.
- With root privileges it opens the camera and the files, then continues as `nobody`.
- Liveness is basic: the nose must move out of the plane of the eyes and the mouth (a small turn
  or nod), which stops a still photo or a paused video but not a photo moved in front of the
  camera. An ordinary webcam has no infrared or depth sensor; Settings warns about it and keeps
  the sudo/polkit switch off by default.

The face recognition model `data/face_recognition_sface_2021dec.onnx` comes from
[opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
(Apache-2.0).
