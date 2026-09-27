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

// lingmo-faceauth: face login with the webcam.
//
//   enroll  [--user NAME] [--out FILE] [--preview]      register the face (as the user)
//   verify  [--user NAME] [--timeout S] [--system]      check the face (PAM, through pam_exec)
//   install-system --user NAME | remove-system --user NAME|--all
//                                                       root: the copy used by sudo and polkit
//   cameras                                             list the real cameras
//
// verify exits 0 when the face matches, 1 when it doesn't, 2 without an enrolled
// face, 3 when the camera is missing or busy, 4 when refused (remote session) and
// 5 on other errors. Under pam_exec (PAM_USER set) the user comes from PAM, the
// environment is PAM's, and the short messages printed on stdout reach the login
// screen, the polkit dialog or the sudo prompt with the "stdout" option.

#include "enrollment.h"
#include "faceengine.h"
#include "v4l2camera.h"

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <dirent.h>
#include <grp.h>
#include <poll.h>
#include <pwd.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <systemd/sd-login.h>
#include <unistd.h>

#ifndef FACEAUTH_MODEL_DIR
#define FACEAUTH_MODEL_DIR "/usr/share/lingmo-camera"
#endif

namespace {

const char DetectorFile[] = "face_detection_yunet_2023mar.onnx";
const char RecognizerFile[] = "face_recognition_sface_2021dec.onnx";
const char SystemDir[] = "/var/lib/lingmo-face";

enum Exit { Match = 0, NoMatch = 1, NotEnrolled = 2, CameraUnavailable = 3, Refused = 4, Failure = 5 };

// SFace's cosine threshold is 0.363 (LFW); a bit stricter here, and two frames
// must agree.
constexpr float MatchThreshold = 0.40f;
constexpr int RequiredMatches = 2;
// Liveness, a cheap one: the nose must leave the plane of the eyes and the mouth
// by this much (in eye distances, FaceEngine::parallax) between the first
// matching frames and later ones, i.e. the head must turn or nod a little. The
// landmarks are averaged over Smoothing frames, since YuNet's jitter alone
// reaches 0.05-0.10 in a dim picture. It stops a still photo or a paused video;
// a photo waved in front of the camera can still pass (see the Settings warning).
constexpr float MinParallax = 0.10f;
constexpr int Smoothing = 3;
constexpr int DefaultTimeout = 4;

constexpr int EnrollFeatures = 5;
constexpr int DefaultEnrollSeconds = 20;
// A new sample must differ this much from the ones kept (a different angle or light)
constexpr float EnrollMaxSimilarity = 0.93f;
// ...and still look like the first one: the same person
constexpr float EnrollMinSimilarity = 0.45f;

using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point t)
{
    return std::chrono::duration<double>(Clock::now() - t).count();
}

// Short messages, in Portuguese or English: PAM's environment rarely has the
// session's locale, so /etc/locale.conf is read too.
bool portuguese()
{
    for (const char *var : {"LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"}) {
        const char *value = std::getenv(var);
        if (value && *value)
            return std::strncmp(value, "pt", 2) == 0;
    }
    std::ifstream conf("/etc/locale.conf");
    std::string line;
    while (std::getline(conf, line)) {
        if (line.rfind("LANG=", 0) == 0) {
            std::string value = line.substr(5);
            value.erase(std::remove(value.begin(), value.end(), '"'), value.end());
            return value.rfind("pt", 0) == 0;
        }
    }
    return false;
}

const char *text(const char *key)
{
    static const bool pt = portuguese();
    static const std::map<std::string, std::pair<const char *, const char *>> texts = {
        {"look", {"Look at the camera…", "Olhe para a câmera…"}},
        {"nomatch", {"Face not recognized. Type your password.", "Rosto não reconhecido. Digite a senha."}},
        {"liveness", {"Face not confirmed: move your head a little next time. Type your password.",
                      "Rosto não confirmado: mexa um pouco a cabeça da próxima vez. Digite a senha."}},
        {"busy", {"The camera is in use. Type your password.", "A câmera está em uso. Digite a senha."}},
        {"nocamera", {"No camera found. Type your password.", "Nenhuma câmera encontrada. Digite a senha."}},
        {"match", {"Face recognized", "Rosto reconhecido"}},
    };
    const auto it = texts.find(key);
    if (it == texts.end())
        return key;
    return pt ? it->second.second : it->second.first;
}

struct Options {
    std::string command;
    std::string user;
    std::string out;
    std::string store;
    std::string systemDir = SystemDir;
    std::string models = FACEAUTH_MODEL_DIR;
    std::string device;
    std::vector<std::string> images;
    int timeout = DefaultTimeout;
    int duration = DefaultEnrollSeconds;
    bool system = false;
    bool machine = false;
    bool skipIfPassword = false;
    bool preview = false;
    bool verbose = false;
    bool all = false;
    bool warp = false;
    float minParallax = MinParallax;
    float threshold = MatchThreshold;
};

// Running with privileges: sudo (setuid), sddm-helper or polkit's helper (root)
bool privileged()
{
    return ::geteuid() == 0 || ::getuid() != ::geteuid() || ::getgid() != ::getegid();
}

void usage()
{
    std::fprintf(stderr,
                 "Usage: lingmo-faceauth enroll [--user NAME] [--out FILE] [--preview] [--duration S]\n"
                 "       lingmo-faceauth verify [--user NAME] [--timeout S] [--system] [--machine]\n"
                 "                              [--skip-if-password] [--verbose]\n"
                 "       lingmo-faceauth install-system --user NAME\n"
                 "       lingmo-faceauth remove-system --user NAME | --all\n"
                 "       lingmo-faceauth cameras\n"
                 "Test options (refused with privileges): --store FILE --system-dir DIR --models DIR\n"
                 "       --device /dev/videoN --image FILE... [--warp] --threshold T --min-parallax P\n");
}

bool validUserName(const std::string &name)
{
    if (name.empty() || name.size() > 32 || name[0] == '-' || name[0] == '.')
        return false;
    for (char c : name) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.' || c == '$'))
            return false;
    }
    return true;
}

std::string currentUser()
{
    const passwd *pw = ::getpwuid(::getuid());
    return pw ? pw->pw_name : std::string();
}

// Machine-readable lines for the lock screen ("verify --machine"), human ones otherwise
void say(const Options &o, const char *machine, const char *human)
{
    if (o.machine) {
        std::printf("face:%s\n", machine);
    } else if (human) {
        std::printf("%s\n", human);
    }
    std::fflush(stdout);
}

// pam_fprintd refuses remote sessions too: a face in front of this computer
// must not authorize sudo typed over ssh.
bool remoteCaller()
{
    const char *rhost = std::getenv("PAM_RHOST");
    if (rhost && *rhost && std::strcmp(rhost, "localhost") != 0)
        return true;

    char *session = nullptr;
    if (sd_pid_get_session(::getppid(), &session) >= 0 && session) {
        const int remote = sd_session_is_remote(session);
        std::free(session);
        if (remote > 0)
            return true;
    }
    return false;
}

// With expose_authtok (the login screen), pam_exec writes the typed password to
// stdin. When there is one, the password check comes next: no need to look.
bool passwordTyped()
{
    if (::isatty(STDIN_FILENO))
        return false;
    pollfd pfd = {STDIN_FILENO, POLLIN, 0};
    if (::poll(&pfd, 1, 500) <= 0)
        return false;
    char c = 0;
    const ssize_t r = ::read(STDIN_FILENO, &c, 1);
    const bool typed = r == 1 && c != '\0';
    c = 0;
    // Drop the rest without looking at it
    char buf[256];
    while (::read(STDIN_FILENO, buf, sizeof(buf)) > 0) {
    }
    std::memset(buf, 0, sizeof(buf));
    return typed;
}

// Frames from the camera, or from pictures for tests (--image, optionally moved
// around with --warp like a photo held in front of the camera).
class Source
{
public:
    explicit Source(const Options &o)
        : m_images(o.images)
        , m_warp(o.warp)
        , m_device(o.device)
    {
    }

    V4l2Camera::Status open()
    {
        if (!m_images.empty()) {
            for (const auto &path : m_images) {
                cv::Mat m = cv::imread(path, cv::IMREAD_COLOR);
                if (m.empty())
                    return V4l2Camera::Status::NoCamera;
                m_loaded.push_back(m);
            }
            return V4l2Camera::Status::Ok;
        }
        return m_camera.open(m_device);
    }

    bool read(cv::Mat &frame)
    {
        if (m_loaded.empty())
            return m_camera.read(frame);

        ::usleep(33000);
        const cv::Mat &src = m_loaded[m_index++ % m_loaded.size()];
        if (!m_warp) {
            frame = src.clone();
            return true;
        }
        // A picture held in front of the camera: turned, tilted, moved, closer and further
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        const cv::Point2f c(src.cols / 2.f, src.rows / 2.f);
        const float s = 1.f + 0.12f * u(m_rng);
        const cv::Point2f src4[4] = {{0, 0}, {float(src.cols), 0}, {float(src.cols), float(src.rows)}, {0, float(src.rows)}};
        cv::Point2f dst4[4];
        const float yaw = 0.10f * u(m_rng), pitch = 0.10f * u(m_rng), roll = 0.15f * u(m_rng);
        const cv::Point2f shift(20 * u(m_rng), 20 * u(m_rng));
        for (int i = 0; i < 4; ++i) {
            cv::Point2f p = (src4[i] - c) * s;
            const float x = p.x * std::cos(roll) - p.y * std::sin(roll);
            const float y = p.x * std::sin(roll) + p.y * std::cos(roll);
            // perspective of a sheet turned around the vertical and horizontal axes
            const float w = 1.f + yaw * x / src.cols + pitch * y / src.rows;
            dst4[i] = cv::Point2f(x / w, y / w) + c + shift;
        }
        cv::warpPerspective(src, frame, FaceEngine::homography(src4, dst4), src.size(),
                            cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        return true;
    }

    std::string name() const { return m_loaded.empty() ? m_camera.path() : std::string("images"); }

private:
    std::vector<std::string> m_images;
    bool m_warp = false;
    std::string m_device;
    std::vector<cv::Mat> m_loaded;
    size_t m_index = 0;
    std::mt19937 m_rng{12345};
    V4l2Camera m_camera;
};

// Everything that needs root is done: go on as nobody (the camera stays open).
bool dropPrivileges()
{
    if (::geteuid() != 0 && ::getuid() == ::geteuid())
        return true;
    if (::geteuid() != 0)
        return false;
    const gid_t nogroup = 65534;
    const uid_t nobody = 65534;
    return ::setgroups(0, nullptr) == 0 && ::setresgid(nogroup, nogroup, nogroup) == 0
        && ::setresuid(nobody, nobody, nobody) == 0 && ::setuid(0) != 0;
}

// Leave with the one who started us (ccheckpass killed by the lock screen, a
// cancelled sudo), so the camera goes off at once.
void followParent(pid_t parent)
{
    ::prctl(PR_SET_PDEATHSIG, SIGTERM);
    if (::getppid() != parent)
        std::_Exit(Failure);
}

// The landmarks of Smoothing faces from first on, averaged
Face averaged(const std::vector<Face> &faces, size_t first)
{
    Face mean;
    for (int p = 0; p < 5; ++p) {
        cv::Point2f sum(0, 0);
        for (int i = 0; i < Smoothing; ++i)
            sum += faces[first + i].points[p];
        mean.points[p] = sum / float(Smoothing);
    }
    return mean;
}

int verify(Options &o)
{
    const pid_t parent = ::getppid();
    const auto started = Clock::now();

    if (o.skipIfPassword && passwordTyped())
        return NoMatch;

    if (o.user.empty()) {
        const char *pamUser = std::getenv("PAM_USER");
        o.user = pamUser ? pamUser : currentUser();
    }
    if (!validUserName(o.user))
        return NotEnrolled;
    const passwd *pw = ::getpwnam(o.user.c_str());
    if (!pw)
        return NotEnrolled;
    const uid_t userUid = pw->pw_uid;

    if (remoteCaller()) {
        say(o, "remote", nullptr);
        return Refused;
    }

    // sudo and polkit only trust root's copy; the lock and login screens the user's
    Enrollment enrollment;
    std::string path;
    Enrollment::LoadResult loaded;
    if (!o.store.empty()) {
        path = o.store;
        loaded = enrollment.load(path, ::getuid());
    } else if (o.system) {
        path = systemStorePath(o.systemDir, o.user);
        loaded = enrollment.load(path, o.systemDir == SystemDir ? 0 : ::getuid());
    } else {
        path = userStorePath(o.user);
        loaded = path.empty() ? Enrollment::LoadResult::Missing : enrollment.load(path, userUid);
    }
    if (loaded != Enrollment::LoadResult::Ok) {
        if (o.verbose)
            std::fprintf(stderr, "no usable enrollment in %s\n", path.c_str());
        say(o, "noenroll", nullptr);
        return NotEnrolled;
    }

    FaceEngine engine(o.models + "/" + DetectorFile, o.models + "/" + RecognizerFile);
    if (!engine.isValid()) {
        std::fprintf(stderr, "lingmo-faceauth: can't load the models from %s\n", o.models.c_str());
        say(o, "error", nullptr);
        return Failure;
    }

    Source source(o);
    const auto status = source.open();
    if (status != V4l2Camera::Status::Ok) {
        if (status == V4l2Camera::Status::Busy)
            say(o, "busy", text("busy"));
        else
            say(o, "nocamera", text("nocamera"));
        return CameraUnavailable;
    }

    if (!dropPrivileges()) {
        std::fprintf(stderr, "lingmo-faceauth: can't drop privileges\n");
        return Failure;
    }
    followParent(parent);

    if (o.verbose)
        std::fprintf(stderr, "ready after %.0f ms (%s)\n", secondsSince(started) * 1000, source.name().c_str());
    say(o, "looking", text("look"));

    const auto looking = Clock::now();
    int matches = 0, frames = 0, faces = 0;
    float best = -1;
    // Landmarks of the matching frames: the first ones are the reference
    std::vector<Face> seen;
    Face reference;
    float parallax = 0;
    cv::Mat frame;
    bool matched = false;

    while (secondsSince(looking) < o.timeout) {
        if (!source.read(frame)) {
            say(o, "nocamera", text("nocamera"));
            return CameraUnavailable;
        }
        ++frames;

        const auto found = engine.detect(frame);
        Face face;
        if (engine.pick(frame, found, false, face) != FrameProblem::None)
            continue;
        ++faces;

        const cv::Mat feat = engine.feature(frame, face);
        float similarity = -1;
        for (const cv::Mat &f : enrollment.features)
            similarity = std::max(similarity, FaceEngine::similarity(feat, f));
        best = std::max(best, similarity);

        float moved = 0;
        if (similarity >= o.threshold) {
            ++matches;
            seen.push_back(face);
            if (int(seen.size()) == Smoothing)
                reference = averaged(seen, 0);
            else if (int(seen.size()) >= 2 * Smoothing)
                moved = FaceEngine::parallax(reference, averaged(seen, seen.size() - Smoothing));
            parallax = std::max(parallax, moved);
        }

        if (o.verbose)
            std::fprintf(stderr, "frame %d: similarity %.3f parallax %.3f\n", frames, similarity, moved);

        if (similarity < o.threshold)
            continue;

        if (matches >= RequiredMatches && parallax >= o.minParallax) {
            matched = true;
            break;
        }
    }

    if (o.verbose)
        std::fprintf(stderr, "%s: %d frames, %d faces, %d matches, best %.3f, head movement %.3f, %.2f s\n",
                     matched ? "match" : "no match", frames, faces, matches, best, parallax, secondsSince(started));

    if (matched) {
        say(o, "match", o.machine ? nullptr : text("match"));
        return Match;
    }
    if (matches >= RequiredMatches)
        say(o, "liveness", text("liveness"));
    else
        say(o, "nomatch", text("nomatch"));
    return NoMatch;
}

std::string base64(const std::vector<uchar> &data)
{
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < data.size(); i += 3) {
        const unsigned v = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
        out += table[(v >> 18) & 63];
        out += table[(v >> 12) & 63];
        out += table[(v >> 6) & 63];
        out += table[v & 63];
    }
    if (i < data.size()) {
        unsigned v = data[i] << 16;
        if (i + 1 < data.size())
            v |= data[i + 1] << 8;
        out += table[(v >> 18) & 63];
        out += table[(v >> 12) & 63];
        out += i + 1 < data.size() ? table[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

// A small mirrored picture with the face box, for Settings (never written to disk)
void sendPreview(const cv::Mat &frame, const Face *face, bool good)
{
    cv::Mat small;
    const double scale = 320.0 / frame.cols;
    cv::resize(frame, small, cv::Size(), scale, scale, cv::INTER_AREA);
    if (face) {
        const cv::Rect box(cvRound(face->box.x * scale), cvRound(face->box.y * scale),
                           cvRound(face->box.width * scale), cvRound(face->box.height * scale));
        cv::rectangle(small, box, good ? cv::Scalar(90, 200, 60) : cv::Scalar(0, 170, 255), 2, cv::LINE_AA);
    }
    cv::flip(small, small, 1);
    std::vector<uchar> jpeg;
    cv::imencode(".jpg", small, jpeg, {cv::IMWRITE_JPEG_QUALITY, 70});
    std::printf("preview %s\n", base64(jpeg).c_str());
    std::fflush(stdout);
}

const char *problemName(FrameProblem p)
{
    switch (p) {
    case FrameProblem::NoFace: return "noface";
    case FrameProblem::TooSmall: return "toofar";
    case FrameProblem::TooMany: return "toomany";
    case FrameProblem::Blurry: return "blurry";
    case FrameProblem::Dark: return "dark";
    case FrameProblem::None: break;
    }
    return "ok";
}

// Progress for Settings on stdout: "hint <what>", "progress <n> <total>",
// "preview <base64 jpeg>", then "done <file>" or "error <what>".
int enroll(Options &o)
{
    // The file belongs to the user and is written by the user
    if (privileged()) {
        std::fprintf(stderr, "lingmo-faceauth: run enroll as the user, without privileges\n");
        return Failure;
    }
    if (o.user.empty())
        o.user = currentUser();
    if (!validUserName(o.user) || !::getpwnam(o.user.c_str())) {
        std::printf("error nouser\n");
        return Failure;
    }
    if (o.out.empty())
        o.out = userStorePath(o.user);
    if (o.out.empty()) {
        std::printf("error nohome\n");
        return Failure;
    }

    FaceEngine engine(o.models + "/" + DetectorFile, o.models + "/" + RecognizerFile);
    if (!engine.isValid()) {
        std::printf("error models\n");
        std::fflush(stdout);
        return Failure;
    }

    Source source(o);
    const auto status = source.open();
    if (status != V4l2Camera::Status::Ok) {
        std::printf("error %s\n", status == V4l2Camera::Status::Busy ? "busy" : "nocamera");
        std::fflush(stdout);
        return CameraUnavailable;
    }
    followParent(::getppid());

    std::printf("progress 0 %d\nhint straight\n", EnrollFeatures);
    std::fflush(stdout);

    Enrollment enrollment;
    std::vector<cv::Mat> &kept = enrollment.features;
    const auto started = Clock::now();
    auto lastKept = started;
    auto lastPreview = started - std::chrono::seconds(1);
    std::string lastHint = "straight";
    int frames = 0;
    cv::Mat frame;

    auto hint = [&lastHint](const std::string &h) {
        if (h != lastHint) {
            std::printf("hint %s\n", h.c_str());
            std::fflush(stdout);
            lastHint = h;
        }
    };

    while (int(kept.size()) < EnrollFeatures && secondsSince(started) < o.duration) {
        if (!source.read(frame)) {
            std::printf("error nocamera\n");
            std::fflush(stdout);
            return CameraUnavailable;
        }
        ++frames;

        const auto found = engine.detect(frame);
        Face face;
        const FrameProblem problem = engine.pick(frame, found, true, face);

        if (o.preview && secondsSince(lastPreview) >= 0.1) {
            sendPreview(frame, found.empty() ? nullptr : &found.front(), problem == FrameProblem::None);
            lastPreview = Clock::now();
        }

        if (problem != FrameProblem::None) {
            hint(problemName(problem));
            continue;
        }
        if (std::chrono::duration<double>(Clock::now() - lastKept).count() < 0.25)
            continue;

        const cv::Mat feat = engine.feature(frame, face);
        if (feat.empty())
            continue;

        float closest = -1;
        for (const cv::Mat &f : kept)
            closest = std::max(closest, FaceEngine::similarity(feat, f));
        if (!kept.empty() && FaceEngine::similarity(feat, kept.front()) < EnrollMinSimilarity) {
            hint("toomany");
            continue;
        }
        if (closest > EnrollMaxSimilarity) {
            // The same picture again: ask for another angle
            hint("turn");
            continue;
        }

        kept.push_back(feat);
        lastKept = Clock::now();
        if (o.verbose)
            std::fprintf(stderr, "sample %zu at frame %d, closest %.3f\n", kept.size(), frames, closest);
        std::printf("progress %zu %d\n", kept.size(), EnrollFeatures);
        std::fflush(stdout);
        hint("turn");
    }

    if (int(kept.size()) < EnrollFeatures) {
        std::printf("error timeout\n");
        std::fflush(stdout);
        return NoMatch;
    }

    enrollment.created = std::time(nullptr);
    std::string error;
    if (!enrollment.save(o.out, 0700, &error)) {
        std::fprintf(stderr, "lingmo-faceauth: %s\n", error.c_str());
        std::printf("error save\n");
        std::fflush(stdout);
        return Failure;
    }
    std::printf("done %s\n", o.out.c_str());
    std::fflush(stdout);
    return Match;
}

// root: copy the user's enrollment to the directory only sudo and polkit read
int installSystem(const Options &o)
{
    const bool test = o.systemDir != SystemDir;
    if (!test && ::geteuid() != 0) {
        std::fprintf(stderr, "lingmo-faceauth: install-system must run as root\n");
        return Failure;
    }
    const passwd *pw = validUserName(o.user) ? ::getpwnam(o.user.c_str()) : nullptr;
    if (!pw) {
        std::fprintf(stderr, "lingmo-faceauth: unknown user\n");
        return Failure;
    }
    const std::string from = o.store.empty() ? userStorePath(o.user) : o.store;
    Enrollment enrollment;
    const auto loaded = enrollment.load(from, test ? ::getuid() : pw->pw_uid);
    if (loaded != Enrollment::LoadResult::Ok) {
        std::fprintf(stderr, "lingmo-faceauth: %s: %s\n", from.c_str(),
                     loaded == Enrollment::LoadResult::Missing ? "no enrolled face" : "not a valid enrollment");
        return NotEnrolled;
    }
    std::string error;
    ::umask(077);
    if (!enrollment.save(systemStorePath(o.systemDir, o.user), 0700, &error)) {
        std::fprintf(stderr, "lingmo-faceauth: %s\n", error.c_str());
        return Failure;
    }
    // Others may see whether a file exists (Settings shows it), not read it
    ::chmod(o.systemDir.c_str(), 0711);
    return Match;
}

int removeSystem(const Options &o)
{
    const bool test = o.systemDir != SystemDir;
    if (!test && ::geteuid() != 0) {
        std::fprintf(stderr, "lingmo-faceauth: remove-system must run as root\n");
        return Failure;
    }
    std::vector<std::string> names;
    if (o.all) {
        if (DIR *dir = ::opendir(o.systemDir.c_str())) {
            while (dirent *e = ::readdir(dir)) {
                const std::string n = e->d_name;
                if (n.size() > 4 && n.compare(n.size() - 4, 4, ".dat") == 0)
                    names.push_back(n.substr(0, n.size() - 4));
            }
            ::closedir(dir);
        }
    } else if (validUserName(o.user)) {
        names.push_back(o.user);
    } else {
        std::fprintf(stderr, "lingmo-faceauth: unknown user\n");
        return Failure;
    }
    int result = Match;
    for (const auto &name : names) {
        if (!validUserName(name))
            continue;
        const std::string path = systemStorePath(o.systemDir, name);
        if (::unlink(path.c_str()) < 0 && errno != ENOENT) {
            std::fprintf(stderr, "lingmo-faceauth: can't remove %s: %s\n", path.c_str(), std::strerror(errno));
            result = Failure;
        }
    }
    return result;
}

bool parse(int argc, char **argv, Options &o)
{
    if (argc < 2)
        return false;
    o.command = argv[1];
    bool testOption = false;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](std::string &target) {
            if (i + 1 >= argc)
                return false;
            target = argv[++i];
            return true;
        };
        auto number = [&](int &target, int min, int max) {
            std::string v;
            if (!value(v))
                return false;
            char *end = nullptr;
            const long n = std::strtol(v.c_str(), &end, 10);
            if (!end || *end || n < min || n > max)
                return false;
            target = int(n);
            return true;
        };

        if (a == "--user") {
            if (!value(o.user))
                return false;
        } else if (a == "--out") {
            if (!value(o.out))
                return false;
        } else if (a == "--timeout") {
            if (!number(o.timeout, 1, 30))
                return false;
        } else if (a == "--duration") {
            if (!number(o.duration, 3, 120))
                return false;
        } else if (a == "--system") {
            o.system = true;
        } else if (a == "--machine") {
            o.machine = true;
        } else if (a == "--skip-if-password") {
            o.skipIfPassword = true;
        } else if (a == "--preview") {
            o.preview = true;
        } else if (a == "--verbose") {
            o.verbose = true;
        } else if (a == "--all") {
            o.all = true;
        } else if (a == "--min-parallax" || a == "--threshold") {
            testOption = true;
            std::string v;
            if (!value(v))
                return false;
            (a == "--threshold" ? o.threshold : o.minParallax) = std::strtof(v.c_str(), nullptr);
        } else if (a == "--warp") {
            o.warp = true;
        } else if (a == "--store") {
            testOption = true;
            if (!value(o.store))
                return false;
        } else if (a == "--system-dir") {
            testOption = true;
            if (!value(o.systemDir))
                return false;
        } else if (a == "--models") {
            testOption = true;
            if (!value(o.models))
                return false;
        } else if (a == "--device") {
            testOption = true;
            if (!value(o.device))
                return false;
        } else if (a == "--image") {
            testOption = true;
            std::string img;
            if (!value(img))
                return false;
            o.images.push_back(img);
        } else {
            return false;
        }
    }

    // Paths chosen by the caller are for tests and for the user's own enrollment,
    // never with privileges (a PAM stack or a setuid caller).
    if (testOption && privileged()) {
        std::fprintf(stderr, "lingmo-faceauth: test options are refused with privileges\n");
        std::exit(Failure);
    }
    if (!o.out.empty() && privileged() && o.command != "enroll") {
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    // pam_exec forwards stdout line by line: don't sit on the messages
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    ::umask(077);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    // Plenty for 640x480, and a login doesn't take every core
    cv::setNumThreads(2);

    Options o;
    if (!parse(argc, argv, o)) {
        usage();
        return Failure;
    }

    if (o.command == "verify") {
        // Never hang a login: whatever happens, give up a little after the timeout
        ::alarm(unsigned(o.timeout) + 8);
        return verify(o);
    }
    if (o.command == "enroll")
        return enroll(o);
    if (o.command == "install-system")
        return installSystem(o);
    if (o.command == "remove-system")
        return removeSystem(o);
    if (o.command == "cameras") {
        for (const auto &c : V4l2Camera::cameras())
            std::printf("%s\n", c.c_str());
        return 0;
    }
    usage();
    return Failure;
}
