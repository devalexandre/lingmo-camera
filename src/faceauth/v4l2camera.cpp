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

#include "v4l2camera.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>

#include <dirent.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr int Width = 640;
constexpr int Height = 480;
constexpr unsigned BufferCount = 4;

int xioctl(int fd, unsigned long request, void *arg)
{
    int r;
    do {
        r = ::ioctl(fd, request, arg);
    } while (r < 0 && errno == EINTR);
    return r;
}

// A capture node of a real camera?
bool isCamera(int fd)
{
    v4l2_capability cap = {};
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) < 0)
        return false;
    const unsigned caps = (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps : cap.capabilities;
    return (caps & V4L2_CAP_VIDEO_CAPTURE) && (caps & V4L2_CAP_STREAMING)
        && std::strcmp(reinterpret_cast<const char *>(cap.driver), "v4l2 loopback") != 0;
}

int nodeNumber(const std::string &name)
{
    return std::atoi(name.c_str() + 5); // "video12"
}
}

V4l2Camera::~V4l2Camera()
{
    close();
}

std::vector<std::string> V4l2Camera::cameras()
{
    std::vector<std::string> nodes;
    if (DIR *dir = ::opendir("/sys/class/video4linux")) {
        while (dirent *entry = ::readdir(dir)) {
            if (std::strncmp(entry->d_name, "video", 5) == 0)
                nodes.emplace_back(entry->d_name);
        }
        ::closedir(dir);
    }
    std::sort(nodes.begin(), nodes.end(), [](const std::string &a, const std::string &b) {
        return nodeNumber(a) < nodeNumber(b);
    });

    std::vector<std::string> list;
    for (const std::string &node : nodes) {
        const std::string path = "/dev/" + node;
        const int fd = ::open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0)
            continue;
        if (isCamera(fd))
            list.push_back(path);
        ::close(fd);
    }
    return list;
}

V4l2Camera::Status V4l2Camera::fail(Status status)
{
    close();
    return status;
}

V4l2Camera::Status V4l2Camera::open(const std::string &wanted)
{
    close();

    std::string path = wanted;
    if (path.empty()) {
        const auto list = cameras();
        if (list.empty())
            return Status::NoCamera;
        path = list.front();
    }

    m_fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW);
    if (m_fd < 0)
        return fail(errno == ENOENT || errno == ENODEV || errno == ENXIO ? Status::NoCamera : Status::Error);
    m_path = path;

    struct stat st;
    if (::fstat(m_fd, &st) < 0 || !S_ISCHR(st.st_mode) || !isCamera(m_fd))
        return fail(Status::NoCamera);

    // YUYV needs no decoding; MJPG for cameras that only offer that
    v4l2_format fmt = {};
    bool formatSet = false;
    for (unsigned pixelformat : {unsigned(V4L2_PIX_FMT_YUYV), unsigned(V4L2_PIX_FMT_MJPEG)}) {
        fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = Width;
        fmt.fmt.pix.height = Height;
        fmt.fmt.pix.pixelformat = pixelformat;
        fmt.fmt.pix.field = V4L2_FIELD_ANY;
        if (xioctl(m_fd, VIDIOC_S_FMT, &fmt) < 0) {
            // Another program streams from it
            if (errno == EBUSY)
                return fail(Status::Busy);
            continue;
        }
        if (fmt.fmt.pix.pixelformat == pixelformat) {
            formatSet = true;
            break;
        }
    }
    if (!formatSet)
        return fail(Status::Error);

    m_format = fmt.fmt.pix.pixelformat;
    m_width = int(fmt.fmt.pix.width);
    m_height = int(fmt.fmt.pix.height);
    m_stride = int(fmt.fmt.pix.bytesperline);
    if (m_stride < m_width * 2)
        m_stride = m_width * 2;

    v4l2_streamparm parm = {};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = 30;
    xioctl(m_fd, VIDIOC_S_PARM, &parm); // best effort

    v4l2_requestbuffers req = {};
    req.count = BufferCount;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(m_fd, VIDIOC_REQBUFS, &req) < 0)
        return fail(errno == EBUSY ? Status::Busy : Status::Error);
    if (req.count < 2)
        return fail(Status::Error);

    for (unsigned i = 0; i < req.count; ++i) {
        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(m_fd, VIDIOC_QUERYBUF, &buf) < 0)
            return fail(Status::Error);
        void *start = ::mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, buf.m.offset);
        if (start == MAP_FAILED)
            return fail(Status::Error);
        m_buffers.push_back({start, buf.length});
        if (xioctl(m_fd, VIDIOC_QBUF, &buf) < 0)
            return fail(Status::Error);
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(m_fd, VIDIOC_STREAMON, &type) < 0)
        return fail(errno == EBUSY || errno == ENOSPC ? Status::Busy : Status::Error);
    m_streaming = true;
    return Status::Ok;
}

void V4l2Camera::close()
{
    if (m_fd < 0)
        return;
    if (m_streaming) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(m_fd, VIDIOC_STREAMOFF, &type);
        m_streaming = false;
    }
    for (const Buffer &b : m_buffers)
        ::munmap(b.start, b.length);
    m_buffers.clear();
    ::close(m_fd);
    m_fd = -1;
    m_path.clear();
}

bool V4l2Camera::read(cv::Mat &bgr, int timeoutMs)
{
    if (!m_streaming)
        return false;

    // A few broken frames in a row are skipped, more mean the camera is gone
    for (int attempt = 0; attempt < 10; ++attempt) {
        pollfd pfd = {m_fd, POLLIN, 0};
        const int r = ::poll(&pfd, 1, timeoutMs);
        if (r < 0 && errno == EINTR) {
            --attempt;
            continue;
        }
        if (r <= 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)))
            return false;

        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (xioctl(m_fd, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN)
                continue;
            return false;
        }

        bool ok = false;
        if (buf.index < m_buffers.size() && !(buf.flags & V4L2_BUF_FLAG_ERROR)) {
            const Buffer &b = m_buffers[buf.index];
            const size_t used = std::min<size_t>(buf.bytesused, b.length);
            if (m_format == V4L2_PIX_FMT_YUYV) {
                if (used >= size_t(m_stride) * m_height) {
                    const cv::Mat yuyv(m_height, m_width, CV_8UC2, b.start, m_stride);
                    cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUYV);
                    ok = true;
                }
            } else if (used > 0) {
                const cv::Mat data(1, int(used), CV_8UC1, b.start);
                try {
                    bgr = cv::imdecode(data, cv::IMREAD_COLOR);
                    ok = !bgr.empty();
                } catch (const cv::Exception &) {
                    ok = false;
                }
            }
        }

        if (xioctl(m_fd, VIDIOC_QBUF, &buf) < 0)
            return false;
        if (ok)
            return true;
        // A broken frame: wait for the next one
    }
    return false;
}
