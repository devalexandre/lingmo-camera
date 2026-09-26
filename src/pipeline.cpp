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

#include "pipeline.h"
#include "loopback.h"

#include <QCollator>
#include <QDebug>
#include <QDir>
#include <QFile>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>

#ifndef MODEL_PATH
#define MODEL_PATH "/usr/share/lingmo-camera/face_detection_yunet_2023mar.onnx"
#endif
#ifndef SEGMENTATION_MODEL_PATH
#define SEGMENTATION_MODEL_PATH "/usr/share/lingmo-camera/human_segmentation_pphumanseg_2023mar.onnx"
#endif

Pipeline::Pipeline(QObject *parent)
    : QThread(parent)
{
}

void Pipeline::setSource(const QString &device)
{
    QMutexLocker lock(&m_mutex);
    if (m_source != device) {
        m_source = device;
        m_sourceChanged = true;
    }
}

void Pipeline::setBackgroundImage(const QString &path)
{
    QMutexLocker lock(&m_mutex);
    if (m_image != path) {
        m_image = path;
        m_imageChanged = true;
    }
}

QVector<QPair<QString, QString>> Pipeline::cameras()
{
    QVector<QPair<QString, QString>> list;
    QStringList nodes = QDir("/sys/class/video4linux").entryList({"video*"}, QDir::Dirs | QDir::System);
    QCollator collator;
    collator.setNumericMode(true);
    std::sort(nodes.begin(), nodes.end(), collator);

    for (const QString &node : nodes) {
        const QByteArray path = QFile::encodeName("/dev/" + node);
        const int fd = ::open(path.constData(), O_RDONLY | O_NONBLOCK);
        if (fd < 0)
            continue;
        v4l2_capability cap = {};
        const bool ok = ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0;
        ::close(fd);
        if (!ok)
            continue;
        const quint32 caps = (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps : cap.capabilities;
        // Skip metadata nodes and every loopback device (ours, OBS's)
        if (!(caps & V4L2_CAP_VIDEO_CAPTURE)
            || qstrcmp(reinterpret_cast<const char *>(cap.driver), "v4l2 loopback") == 0)
            continue;
        list.append({QString::fromLatin1(path), QString::fromUtf8(reinterpret_cast<const char *>(cap.card))});
    }
    return list;
}

QString Pipeline::pickSource() const
{
    QString wanted;
    {
        QMutexLocker lock(&m_mutex);
        wanted = m_source;
    }
    const auto list = cameras();
    for (const auto &camera : list) {
        if (camera.first == wanted)
            return wanted;
    }
    return list.isEmpty() ? QString() : list.first().first;
}

void Pipeline::run()
{
    Loopback out;
    Framer framer(MODEL_PATH);
    if (!framer.isValid())
        qWarning() << "face model not loaded from" << MODEL_PATH << "- showing the whole picture";
    Background background(SEGMENTATION_MODEL_PATH);
    if (!background.isValid())
        qWarning() << "segmentation model not loaded from" << SEGMENTATION_MODEL_PATH << "- no background blur";
    bool blurring = false;

    cv::VideoCapture camera;
    QString cameraDevice;
    cv::Mat frame, scaled;
    const cv::Size outSize(Loopback::Width, Loopback::Height);
    const cv::Mat idle(outSize, CV_8UC3, cv::Scalar(24, 24, 24));
    bool available = false, active = false;

    auto closeCamera = [&] {
        if (camera.isOpened()) {
            camera.release();
            qInfo() << "camera released";
        }
        cameraDevice.clear();
    };

    while (!m_stop) {
        if (!out.isOpen() && !out.open(Label)) {
            if (available) {
                available = false;
                emit availableChanged(false);
            }
            msleep(2000);
            continue;
        }
        if (!available) {
            available = true;
            emit availableChanged(true);
        }

        const bool consumers = out.hasConsumers();
        if (consumers != active) {
            active = consumers;
            emit activeChanged(active);
        }

        if (!consumers) {
            closeCamera();
            // Keep a picture on the device, or apps don't list it at all
            out.write(idle);
            msleep(200);
            continue;
        }

        if (m_sourceChanged.exchange(false))
            closeCamera();

        if (!camera.isOpened()) {
            cameraDevice = pickSource();
            if (cameraDevice.isEmpty() || !camera.open(cameraDevice.toStdString(), cv::CAP_V4L2)) {
                // No camera, or another app holds it
                out.write(idle);
                msleep(1000);
                continue;
            }
            camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            camera.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
            camera.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
            camera.set(cv::CAP_PROP_FPS, 30);
            camera.set(cv::CAP_PROP_BUFFERSIZE, 1);
            framer.reset();
            background.reset();
            for (const auto &c : cameras()) {
                if (c.first == cameraDevice)
                    emit sourceNameChanged(c.second);
            }
            qInfo() << "camera" << cameraDevice << camera.get(cv::CAP_PROP_FRAME_WIDTH)
                    << "x" << camera.get(cv::CAP_PROP_FRAME_HEIGHT);
        }

        if (!camera.read(frame) || frame.empty()) {
            // Unplugged: look for it again
            closeCamera();
            msleep(500);
            continue;
        }

        cv::Rect2f crop = m_framing ? framer.frame(frame, m_zoom) : Framer::whole(frame.size());
        if (!m_framing)
            framer.reset();
        const cv::Rect roi = cv::Rect(cvRound(crop.x), cvRound(crop.y),
                                      cvRound(crop.width), cvRound(crop.height))
                             & cv::Rect(0, 0, frame.cols, frame.rows);
        cv::resize(frame(roi), scaled, outSize, 0, 0,
                   roi.width > outSize.width ? cv::INTER_AREA : cv::INTER_LINEAR);

        // After the crop, so the mask matches what the apps get
        if (m_imageChanged.exchange(false)) {
            QMutexLocker lock(&m_mutex);
            background.setImage(QFile::encodeName(m_image).toStdString());
        }
        if (m_blur) {
            background.apply(scaled, m_blurStrength);
        } else if (blurring) {
            background.reset();
        }
        blurring = m_blur;
        out.write(scaled);
    }

    closeCamera();
}
