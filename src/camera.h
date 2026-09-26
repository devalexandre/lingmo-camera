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

#ifndef CAMERA_H
#define CAMERA_H

#include <QObject>
#include <QSettings>
#include <QStringList>

class Pipeline;

// com.lingmo.Camera on the session bus: the settings page reads and changes the
// camera options through it. Options live in ~/.config/lingmoos/camera.conf.
class Camera : public QObject
{
    Q_OBJECT
    Q_CLASSINFO("D-Bus Interface", "com.lingmo.Camera")
    Q_PROPERTY(bool Framing READ framing WRITE setFraming)
    Q_PROPERTY(QString Zoom READ zoom WRITE setZoom)
    Q_PROPERTY(QString Source READ source WRITE setSource)
    Q_PROPERTY(bool Available READ available)
    Q_PROPERTY(bool Active READ active)
    Q_PROPERTY(QString SourceName READ sourceName)

public:
    explicit Camera(QObject *parent = nullptr);
    ~Camera() override;

    bool framing() const;
    void setFraming(bool on);
    // "wide", "medium" or "close"
    QString zoom() const;
    void setZoom(const QString &zoom);
    // Device of the real camera, "" for the first one
    QString source() const;
    void setSource(const QString &device);

    // The virtual camera exists (the v4l2loopback module is loaded)
    bool available() const { return m_available; }
    // Some app is using it right now
    bool active() const { return m_active; }
    QString sourceName() const { return m_sourceName; }

public slots:
    // "device|name" for each real camera
    QStringList Cameras() const;

signals:
    // Any property changed
    void Changed();

private:
    void apply();

    QSettings m_settings;
    Pipeline *m_pipeline;
    bool m_available = false;
    bool m_active = false;
    QString m_sourceName;
};

#endif // CAMERA_H
