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

#ifndef FRAMER_H
#define FRAMER_H

#include <opencv2/core.hpp>
#include <opencv2/objdetect/face.hpp>

#include <chrono>
#include <string>

// Auto framing: finds the faces in each frame and moves a 16:9 crop over the camera
// image so the people stay centred, like a camera operator would. The crop only moves
// when the people really moved (a dead zone ignores small head motion) and eases
// towards its new place, so the picture never jitters.
class Framer
{
public:
    enum Zoom { Wide, Medium, Close };

    explicit Framer(const std::string &model);
    bool isValid() const { return !m_detector.empty(); }

    // Returns the part of the frame to show
    cv::Rect2f frame(const cv::Mat &bgr, Zoom zoom);
    // The whole picture, cut to 16:9
    static cv::Rect2f whole(const cv::Size &size);
    // Forget where people were (new camera, framing turned back on)
    void reset();

private:
    bool detect(const cv::Mat &bgr, cv::Rect2f &people);
    cv::Rect2f cropFor(const cv::Rect2f &people, Zoom zoom) const;

    cv::Ptr<cv::FaceDetectorYN> m_detector;
    cv::Size m_frameSize;
    cv::Size m_detectSize;
    cv::Mat m_small;

    cv::Rect2f m_people;      // smoothed face box
    bool m_hasPeople = false;
    std::chrono::steady_clock::time_point m_lastSeen;   // last frame with a face
    std::chrono::steady_clock::time_point m_lastFrame;
    cv::Rect2f m_target;      // where the crop is heading
    cv::Rect2f m_current;     // where the crop is now
};

#endif // FRAMER_H
