/*
 * Copyright (C) 2026 LingmoOS Team.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef FACEAUTH_V4L2CAMERA_H
#define FACEAUTH_V4L2CAMERA_H

#include <opencv2/core.hpp>

#include <string>
#include <vector>

// A small V4L2 capture (mmap, YUYV or MJPG at 640x480), written by hand instead of
// cv::VideoCapture so that a camera held by another program (a video call through
// lingmo-camera) is reported at once as busy instead of after a long timeout.
class V4l2Camera
{
public:
    enum class Status { Ok, NoCamera, Busy, Error };

    V4l2Camera() = default;
    ~V4l2Camera();
    V4l2Camera(const V4l2Camera &) = delete;
    V4l2Camera &operator=(const V4l2Camera &) = delete;

    // Real cameras only: the first video capture nodes that are not loopback
    // devices (the virtual Lingmo Camera, OBS's), in /dev/videoN order, like
    // lingmo-camera's Pipeline::cameras().
    static std::vector<std::string> cameras();

    // Opens and starts streaming; an empty path picks the first camera.
    Status open(const std::string &path = std::string());
    void close();
    bool isOpen() const { return m_fd >= 0; }
    const std::string &path() const { return m_path; }

    // Waits up to timeoutMs for the next frame, as BGR.
    bool read(cv::Mat &bgr, int timeoutMs = 1500);

private:
    struct Buffer {
        void *start = nullptr;
        size_t length = 0;
    };

    Status fail(Status status);

    int m_fd = -1;
    std::string m_path;
    unsigned m_format = 0;
    int m_width = 0;
    int m_height = 0;
    int m_stride = 0;
    std::vector<Buffer> m_buffers;
    bool m_streaming = false;
};

#endif
