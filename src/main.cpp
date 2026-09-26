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

#include <QCoreApplication>
#include <QDBusConnection>
#include <QDebug>

#include <csignal>

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    app.setApplicationName("lingmo-camera");

    // One per session
    QDBusConnection bus = QDBusConnection::sessionBus();
    if (!bus.registerService("com.lingmo.Camera")) {
        qWarning() << "lingmo-camera is already running";
        return 0;
    }

    // Quit cleanly on logout so the real camera is released
    for (int sig : {SIGTERM, SIGINT, SIGHUP})
        std::signal(sig, [](int) { QCoreApplication::quit(); });

    Camera camera;
    bus.registerObject("/Camera", &camera,
                       QDBusConnection::ExportAllProperties | QDBusConnection::ExportAllSlots
                           | QDBusConnection::ExportAllSignals);

    return app.exec();
}
