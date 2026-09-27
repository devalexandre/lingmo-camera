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

#ifndef FACEAUTH_ENROLLMENT_H
#define FACEAUTH_ENROLLMENT_H

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <sys/types.h>
#include <vector>

// The enrolled face of one user: a few SFace features (numbers, no picture).
//
// Stored in two places:
//  - ~/.local/share/lingmoos/face/<user>.dat (mode 0600, owned by the user): written by
//    "lingmo-faceauth enroll" as the user, read by the lock screen (PAM as the user)
//    and by the login screen (sddm-helper runs PAM as root, and root can read it);
//  - /var/lib/lingmo-face/<user>.dat (0600 root, directory 0711): a copy made by root
//    ("install-system", run by lingmo-settings' face-pam helper through pkexec) and
//    the only one sudo and polkit use ("verify --system"). A program running as the
//    user can rewrite the first file, but must know an administrator password to
//    change what unlocks administrator rights.
struct Enrollment {
    static constexpr int Dims = 128;
    static constexpr int MaxFeatures = 10;

    std::vector<cv::Mat> features; // 1x128 CV_32F, L2-normalized
    int64_t created = 0;

    enum class LoadResult { Ok, Missing, Invalid };

    // Reads path without following a symlink at the end; the file must be a small
    // regular file owned by owner and not writable by anybody else.
    LoadResult load(const std::string &path, uid_t owner);
    // Atomic write (temporary file + rename), mode 0600, parent directories created
    // with dirMode.
    bool save(const std::string &path, mode_t dirMode, std::string *error = nullptr) const;
};

// ~/.local/share/lingmoos/face/<user>.dat of that user (empty if unknown).
std::string userStorePath(const std::string &user);
std::string systemStorePath(const std::string &systemDir, const std::string &user);

#endif
