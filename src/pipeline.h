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

#ifndef PIPELINE_H
#define PIPELINE_H

#include "framer.h"

#include <QMutex>
#include <QPair>
#include <QThread>
#include <QVector>

#include <atomic>

// Camera -> framing -> "Lingmo Camera", on its own thread. The real camera is only
// opened while some app is streaming from the virtual one, so its light stays off the
// rest of the time.
class Pipeline : public QThread
{
    Q_OBJECT

public:
    static constexpr const char *Label = "Lingmo Camera";

    explicit Pipeline(QObject *parent = nullptr);

    void setFraming(bool on) { m_framing = on; }
    void setZoom(Framer::Zoom zoom) { m_zoom = zoom; }
    // "" picks the first real camera
    void setSource(const QString &device);

    void stop() { m_stop = true; }

    // Real cameras: (device, name)
    static QVector<QPair<QString, QString>> cameras();

signals:
    void availableChanged(bool available);
    void activeChanged(bool active);
    void sourceNameChanged(const QString &name);

protected:
    void run() override;

private:
    QString pickSource() const;

    std::atomic<bool> m_framing = true;
    std::atomic<Framer::Zoom> m_zoom = Framer::Medium;
    std::atomic<bool> m_stop = false;
    std::atomic<bool> m_sourceChanged = false;
    mutable QMutex m_mutex;
    QString m_source;
};

#endif // PIPELINE_H
