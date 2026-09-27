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

#include "enrollment.h"

#include <cerrno>
#include <cmath>
#include <cstring>

#include <fcntl.h>
#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
// File layout (little endian, as written on x86/arm):
//   8 bytes  magic "LMFACE\0\1"
//   u32      dimensions (128)
//   u32      number of features (1..10)
//   i64      creation time (seconds since the epoch)
//   f32[]    the features, one after the other
const char Magic[8] = {'L', 'M', 'F', 'A', 'C', 'E', 0, 1};
constexpr size_t HeaderSize = 8 + 4 + 4 + 8;

bool readAll(int fd, char *data, size_t size)
{
    size_t done = 0;
    while (done < size) {
        const ssize_t r = ::read(fd, data + done, size - done);
        if (r < 0 && errno == EINTR)
            continue;
        if (r <= 0)
            return false;
        done += size_t(r);
    }
    return true;
}

bool writeAll(int fd, const char *data, size_t size)
{
    size_t done = 0;
    while (done < size) {
        const ssize_t r = ::write(fd, data + done, size - done);
        if (r < 0 && errno == EINTR)
            continue;
        if (r <= 0)
            return false;
        done += size_t(r);
    }
    return true;
}

// mkdir -p, the directories we create get mode
bool makePath(const std::string &dir, mode_t mode)
{
    if (dir.empty())
        return false;
    struct stat st;
    if (::stat(dir.c_str(), &st) == 0)
        return S_ISDIR(st.st_mode);
    const size_t slash = dir.find_last_of('/');
    if (slash != std::string::npos && slash > 0 && !makePath(dir.substr(0, slash), mode))
        return false;
    if (::mkdir(dir.c_str(), mode) < 0 && errno != EEXIST)
        return false;
    ::chmod(dir.c_str(), mode); // umask
    return true;
}
}

Enrollment::LoadResult Enrollment::load(const std::string &path, uid_t owner)
{
    features.clear();

    const int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | O_NOCTTY);
    if (fd < 0)
        return errno == ENOENT || errno == ENOTDIR ? LoadResult::Missing : LoadResult::Invalid;

    struct stat st;
    const size_t maxSize = HeaderSize + size_t(MaxFeatures) * Dims * sizeof(float);
    if (::fstat(fd, &st) < 0 || !S_ISREG(st.st_mode) || st.st_uid != owner
        || (st.st_mode & (S_IWGRP | S_IWOTH)) || st.st_size < off_t(HeaderSize) || size_t(st.st_size) > maxSize) {
        ::close(fd);
        return LoadResult::Invalid;
    }

    std::vector<char> data(size_t(st.st_size));
    const bool ok = readAll(fd, data.data(), data.size());
    ::close(fd);
    if (!ok || std::memcmp(data.data(), Magic, sizeof(Magic)) != 0)
        return LoadResult::Invalid;

    uint32_t dims = 0, count = 0;
    std::memcpy(&dims, data.data() + 8, 4);
    std::memcpy(&count, data.data() + 12, 4);
    std::memcpy(&created, data.data() + 16, 8);
    if (dims != Dims || count < 1 || count > MaxFeatures
        || data.size() != HeaderSize + size_t(count) * Dims * sizeof(float))
        return LoadResult::Invalid;

    for (uint32_t i = 0; i < count; ++i) {
        cv::Mat f(1, Dims, CV_32F);
        std::memcpy(f.data, data.data() + HeaderSize + size_t(i) * Dims * sizeof(float), Dims * sizeof(float));
        const double norm = cv::norm(f);
        if (!std::isfinite(norm) || std::abs(norm - 1.0) > 0.01) {
            features.clear();
            return LoadResult::Invalid;
        }
        features.push_back(f);
    }
    return LoadResult::Ok;
}

bool Enrollment::save(const std::string &path, mode_t dirMode, std::string *error) const
{
    auto fail = [error](const std::string &message) {
        if (error)
            *error = message;
        return false;
    };

    if (features.empty() || features.size() > size_t(MaxFeatures))
        return fail("nothing to save");

    const size_t slash = path.find_last_of('/');
    const std::string dir = slash == std::string::npos ? std::string(".") : path.substr(0, slash);
    if (!makePath(dir, dirMode))
        return fail("can't create " + dir + ": " + std::strerror(errno));

    std::string data(Magic, sizeof(Magic));
    const uint32_t dims = Dims, count = uint32_t(features.size());
    data.append(reinterpret_cast<const char *>(&dims), 4);
    data.append(reinterpret_cast<const char *>(&count), 4);
    data.append(reinterpret_cast<const char *>(&created), 8);
    for (const cv::Mat &f : features) {
        cv::Mat c;
        f.convertTo(c, CV_32F);
        c = c.reshape(1, 1);
        if (c.cols != Dims || !c.isContinuous())
            return fail("bad feature");
        data.append(reinterpret_cast<const char *>(c.data), Dims * sizeof(float));
    }

    std::string tmp = dir + "/.face-XXXXXX";
    std::vector<char> name(tmp.begin(), tmp.end());
    name.push_back('\0');
    const int fd = ::mkostemp(name.data(), O_CLOEXEC);
    if (fd < 0)
        return fail("can't write in " + dir + ": " + std::strerror(errno));
    tmp = name.data();

    const bool ok = ::fchmod(fd, 0600) == 0 && writeAll(fd, data.data(), data.size()) && ::fsync(fd) == 0;
    ::close(fd);
    if (!ok || ::rename(tmp.c_str(), path.c_str()) < 0) {
        const std::string reason = std::strerror(errno);
        ::unlink(tmp.c_str());
        return fail("can't write " + path + ": " + reason);
    }
    return true;
}

std::string userStorePath(const std::string &user)
{
    const passwd *pw = ::getpwnam(user.c_str());
    if (!pw || !pw->pw_dir || pw->pw_dir[0] != '/')
        return std::string();
    return std::string(pw->pw_dir) + "/.local/share/lingmoos/face/" + user + ".dat";
}

std::string systemStorePath(const std::string &systemDir, const std::string &user)
{
    return systemDir + "/" + user + ".dat";
}
