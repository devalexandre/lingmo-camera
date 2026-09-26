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

#include "framer.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace {
// Faces are searched in a copy this wide: plenty for faces at meeting distance, and
// fast enough to run on every frame
constexpr int DetectWidth = 480;
// Never zoom in more than this, or the picture gets soft
constexpr float MaxZoom = 2.0f;
// How much of the crop's height the face takes, per zoom setting...
constexpr float FaceShare[] = {0.20f, 0.28f, 0.38f};
// ...and the least each setting zooms in, which leaves room to follow someone who
// moves sideways even when they sit close to the camera
constexpr float MinZoom[] = {1.0f, 1.25f, 1.5f};
// Where the face sits, from the top of the crop (a bit above the middle, like a portrait)
constexpr float FaceLine = 0.40f;
// Move only when the people moved more than this share of the crop
constexpr float DeadZone = 0.08f;
// Time constants in seconds, so the motion is the same at 15 or 30 fps: how fast the
// crop follows and how fast the face box follows the raw detections
constexpr float FollowTime = 0.30f;
constexpr float SmoothTime = 0.10f;
// Seconds without a face before going back to the whole picture
constexpr float Patience = 2.5f;

cv::Rect2f lerp(const cv::Rect2f &a, const cv::Rect2f &b, float t)
{
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
            a.width + (b.width - a.width) * t, a.height + (b.height - a.height) * t};
}
}

cv::Rect2f Framer::whole(const cv::Size &size)
{
    // The widest 16:9 part of the picture, so a 4:3 camera isn't stretched
    const float w = std::min<float>(size.width, size.height * 16.0f / 9.0f);
    const float h = w * 9.0f / 16.0f;
    return {(size.width - w) / 2, (size.height - h) / 2, w, h};
}

Framer::Framer(const std::string &model)
{
    try {
        m_detector = cv::FaceDetectorYN::create(model, "", cv::Size(320, 320), 0.75f, 0.3f, 20);
    } catch (const cv::Exception &e) {
        m_detector.release();
    }
}

void Framer::reset()
{
    m_frameSize = {};
    m_hasPeople = false;
}

bool Framer::detect(const cv::Mat &bgr, cv::Rect2f &people)
{
    const float scale = float(bgr.cols) / DetectWidth;
    const cv::Size size(DetectWidth, std::lround(bgr.rows / scale));
    if (size != m_detectSize) {
        m_detector->setInputSize(size);
        m_detectSize = size;
    }
    cv::resize(bgr, m_small, size, 0, 0, cv::INTER_AREA);

    cv::Mat faces;
    m_detector->detect(m_small, faces);
    if (faces.rows == 0)
        return false;

    // Everyone at the table, but not the tiny faces far behind (a poster, a passer-by)
    float largest = 0;
    for (int i = 0; i < faces.rows; ++i)
        largest = std::max(largest, faces.at<float>(i, 3));

    cv::Rect2f box;
    bool first = true;
    for (int i = 0; i < faces.rows; ++i) {
        const cv::Rect2f face(faces.at<float>(i, 0), faces.at<float>(i, 1),
                              faces.at<float>(i, 2), faces.at<float>(i, 3));
        if (face.height < largest * 0.55f)
            continue;
        box = first ? face : (box | face);
        first = false;
    }

    people = {box.x * scale, box.y * scale, box.width * scale, box.height * scale};
    return true;
}

cv::Rect2f Framer::cropFor(const cv::Rect2f &people, Zoom zoom) const
{
    const float fw = m_frameSize.width, fh = m_frameSize.height;
    const float aspect = 16.0f / 9.0f;

    // Tall enough for the face to take its share, wide enough for everyone plus room
    // on the sides
    float h = people.height / FaceShare[zoom];
    h = std::max(h, people.width * 1.8f / aspect);
    h = std::clamp(h, fh / MaxZoom, fh / MinZoom[zoom]);
    float w = std::min(h * aspect, fw);
    h = w / aspect;

    const float cx = people.x + people.width / 2;
    const float cy = people.y + people.height / 2;
    const float x = std::clamp(cx - w / 2, 0.0f, fw - w);
    const float y = std::clamp(cy - h * FaceLine, 0.0f, fh - h);
    return {x, y, w, h};
}

cv::Rect2f Framer::frame(const cv::Mat &bgr, Zoom zoom)
{
    const cv::Rect2f all = whole(bgr.size());
    if (m_detector.empty())
        return all;

    const auto now = std::chrono::steady_clock::now();
    if (bgr.size() != m_frameSize) {
        m_frameSize = bgr.size();
        m_target = m_current = all;
        m_hasPeople = false;
        m_lastFrame = m_lastSeen = now;
    }
    const float dt = std::min(std::chrono::duration<float>(now - m_lastFrame).count(), 0.5f);
    m_lastFrame = now;

    cv::Rect2f people;
    if (detect(bgr, people)) {
        // Smooth the raw detections, which wobble a few pixels between frames
        m_people = m_hasPeople ? lerp(m_people, people, 1 - std::exp(-dt / SmoothTime)) : people;
        m_hasPeople = true;
        m_lastSeen = now;
    } else if (m_hasPeople && std::chrono::duration<float>(now - m_lastSeen).count() > Patience) {
        // Everybody left: show the whole room
        m_hasPeople = false;
        m_target = all;
    }

    if (m_hasPeople) {
        const cv::Rect2f wanted = cropFor(m_people, zoom);
        const float dx = std::abs((wanted.x + wanted.width / 2) - (m_target.x + m_target.width / 2));
        const float dy = std::abs((wanted.y + wanted.height / 2) - (m_target.y + m_target.height / 2));
        const float dh = std::abs(wanted.height - m_target.height);
        if (dx > m_target.width * DeadZone || dy > m_target.height * DeadZone
            || dh > m_target.height * DeadZone * 1.5f)
            m_target = wanted;
    }

    m_current = lerp(m_current, m_target, 1 - std::exp(-dt / FollowTime));
    return m_current;
}
