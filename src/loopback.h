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

#ifndef LOOPBACK_H
#define LOOPBACK_H

#include <QString>
#include <opencv2/core.hpp>

// The writing end of the v4l2loopback device that apps see as "Lingmo Camera".
// Frames go out as YUV 4:2:0, which every browser and meeting app accepts.
class Loopback
{
public:
    static constexpr int Width = 1280;
    static constexpr int Height = 720;

    ~Loopback();

    // Finds the device by its card label; false while the module isn't loaded
    bool open(const QString &label);
    void close();
    bool isOpen() const { return m_fd >= 0; }
    QString device() const { return m_device; }

    // A BGR frame of Width x Height
    bool write(const cv::Mat &bgr);

    // Whether some app is streaming from the camera. v4l2loopback reports it with an
    // event, so this is cheap to call every frame.
    bool hasConsumers();

    static QString findDevice(const QString &label);

private:
    int m_fd = -1;
    QString m_device;
    bool m_consumers = false;
    cv::Mat m_yuv;
};

#endif // LOOPBACK_H
