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

#include "camera.h"
#include "pipeline.h"

#include <QStandardPaths>

namespace {
const QStringList Zooms = {"wide", "medium", "close"};
}

Camera::Camera(QObject *parent)
    : QObject(parent)
    , m_settings(QStandardPaths::writableLocation(QStandardPaths::GenericConfigLocation)
                     + "/lingmoos/camera.conf",
                 QSettings::IniFormat)
    , m_pipeline(new Pipeline(this))
{
    connect(m_pipeline, &Pipeline::availableChanged, this, [this](bool available) {
        m_available = available;
        emit Changed();
    });
    connect(m_pipeline, &Pipeline::activeChanged, this, [this](bool active) {
        m_active = active;
        emit Changed();
    });
    connect(m_pipeline, &Pipeline::sourceNameChanged, this, [this](const QString &name) {
        m_sourceName = name;
        emit Changed();
    });

    apply();
    m_pipeline->start();
}

Camera::~Camera()
{
    m_pipeline->stop();
    m_pipeline->wait();
}

void Camera::apply()
{
    m_pipeline->setFraming(framing());
    m_pipeline->setZoom(Framer::Zoom(std::max<qsizetype>(0, Zooms.indexOf(zoom()))));
    m_pipeline->setSource(source());
}

bool Camera::framing() const
{
    return m_settings.value("Framing", true).toBool();
}

void Camera::setFraming(bool on)
{
    m_settings.setValue("Framing", on);
    apply();
    emit Changed();
}

QString Camera::zoom() const
{
    const QString value = m_settings.value("Zoom", "medium").toString();
    return Zooms.contains(value) ? value : "medium";
}

void Camera::setZoom(const QString &zoom)
{
    if (!Zooms.contains(zoom))
        return;
    m_settings.setValue("Zoom", zoom);
    apply();
    emit Changed();
}

QString Camera::source() const
{
    return m_settings.value("Source").toString();
}

void Camera::setSource(const QString &device)
{
    m_settings.setValue("Source", device);
    apply();
    emit Changed();
}

QStringList Camera::Cameras() const
{
    QStringList list;
    for (const auto &camera : Pipeline::cameras())
        list << camera.first + "|" + camera.second;
    return list;
}
