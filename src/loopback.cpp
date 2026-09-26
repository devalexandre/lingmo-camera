/*
 * Copyright (C) 2026 LingmoOS Team.
 *
 * Author:     devalexandre <alexandre@dev2learn.com>
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

#include "loopback.h"

#include <QDebug>
#include <QDir>
#include <QFile>

#include <opencv2/imgproc.hpp>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>

// Not in the kernel headers: v4l2loopback's own event, sent when the number of apps
// streaming from the device changes (see v4l2loopback.c)
#define V4L2_EVENT_PRI_CLIENT_USAGE (V4L2_EVENT_PRIVATE_START + 0x08E00000 + 1)

Loopback::~Loopback()
{
    close();
}

QString Loopback::findDevice(const QString &label)
{
    const QDir sys("/sys/class/video4linux");
    for (const QString &node : sys.entryList({"video*"}, QDir::Dirs | QDir::System)) {
        QFile name(sys.filePath(node + "/name"));
        // Stray quotes stay in the name when the module options were quoted per label
        if (name.open(QIODevice::ReadOnly)
            && QString::fromUtf8(name.readAll()).trimmed().remove('"') == label)
            return "/dev/" + node;
    }
    return {};
}

bool Loopback::open(const QString &label)
{
    close();

    const QString device = findDevice(label);
    if (device.isEmpty())
        return false;

    int fd = ::open(QFile::encodeName(device).constData(), O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        qWarning() << "cannot open" << device;
        return false;
    }

    v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    fmt.fmt.pix.width = Width;
    fmt.fmt.pix.height = Height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    fmt.fmt.pix.bytesperline = Width;
    fmt.fmt.pix.sizeimage = Width * Height * 3 / 2;
    fmt.fmt.pix.colorspace = V4L2_COLORSPACE_SRGB;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        // Another program (OBS, say) is already writing to it
        qWarning() << "cannot set the format of" << device;
        ::close(fd);
        return false;
    }

    v4l2_streamparm parm = {};
    parm.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    parm.parm.output.timeperframe = {1, 30};
    ioctl(fd, VIDIOC_S_PARM, &parm);

    v4l2_event_subscription sub = {};
    sub.type = V4L2_EVENT_PRI_CLIENT_USAGE;
    sub.flags = V4L2_EVENT_SUB_FL_SEND_INITIAL;
    if (ioctl(fd, VIDIOC_SUBSCRIBE_EVENT, &sub) < 0)
        qWarning() << "v4l2loopback too old: can't tell when the camera is in use";

    m_fd = fd;
    m_device = device;
    m_consumers = false;
    qInfo() << "writing to" << device;
    return true;
}

void Loopback::close()
{
    if (m_fd >= 0)
        ::close(m_fd);
    m_fd = -1;
    m_device.clear();
    m_consumers = false;
}

bool Loopback::write(const cv::Mat &bgr)
{
    if (m_fd < 0)
        return false;

    cv::cvtColor(bgr, m_yuv, cv::COLOR_BGR2YUV_I420);
    const size_t size = m_yuv.total() * m_yuv.elemSize();
    const ssize_t written = ::write(m_fd, m_yuv.data, size);
    if (written < 0 && errno != EAGAIN && errno != EINTR) {
        // The module was unloaded under us
        qWarning() << "lost" << m_device;
        close();
        return false;
    }
    return true;
}

bool Loopback::hasConsumers()
{
    if (m_fd < 0)
        return false;

    pollfd pfd = {m_fd, POLLPRI, 0};
    while (poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLPRI)) {
        v4l2_event ev = {};
        if (ioctl(m_fd, VIDIOC_DQEVENT, &ev) < 0)
            break;
        if (ev.type == V4L2_EVENT_PRI_CLIENT_USAGE) {
            quint32 count;
            memcpy(&count, ev.u.data, sizeof(count));
            m_consumers = count > 0;
        }
    }
    return m_consumers;
}
