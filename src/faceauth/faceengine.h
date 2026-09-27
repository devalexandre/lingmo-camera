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

#ifndef FACEAUTH_FACEENGINE_H
#define FACEAUTH_FACEENGINE_H

#include <opencv2/core.hpp>
#include <opencv2/objdetect/face.hpp>

#include <string>
#include <vector>

// One face found in a frame by YuNet.
struct Face {
    cv::Mat row;          // YuNet's 1x15 row: box, 5 landmarks, score
    cv::Rect2f box;
    float score = 0;
    // Right eye, left eye, nose tip, right and left corner of the mouth
    cv::Point2f points[5];
};

// Why a frame can't be used.
enum class FrameProblem { None, NoFace, TooSmall, TooMany, Blurry, Dark };

// YuNet (detection + 5 landmarks) and SFace (128-d features, compared by cosine
// similarity), both from opencv_zoo.
class FaceEngine
{
public:
    FaceEngine(const std::string &detectorModel, const std::string &recognizerModel);
    bool isValid() const { return !m_detector.empty() && !m_recognizer.empty(); }

    // Every face in the frame, the biggest first.
    std::vector<Face> detect(const cv::Mat &bgr);

    // The face to use in a frame: the biggest, when it is big and sharp enough.
    // With strict set (enrollment) a second face of similar size is refused too.
    FrameProblem pick(const cv::Mat &bgr, const std::vector<Face> &faces, bool strict, Face &face);

    // L2-normalized SFace feature (1x128 CV_32F) of a face.
    cv::Mat feature(const cv::Mat &bgr, const Face &face);

    // Cosine similarity of two normalized features.
    static float similarity(const cv::Mat &a, const cv::Mat &b);

    // How far the nose tip is from where a flat face would put it, in eye
    // distances. The eyes and the mouth corners of a picture moved, turned or
    // tilted in front of the camera follow a homography (a plane seen in
    // perspective), and so does the nose printed on it; the nose of a real head
    // stands out of that plane and leaves it when the head turns or nods.
    static float parallax(const Face &reference, const Face &face);

    // The 3x3 homography taking the 4 points of src to those of dst.
    static cv::Mat homography(const cv::Point2f src[4], const cv::Point2f dst[4]);

private:
    cv::Ptr<cv::FaceDetectorYN> m_detector;
    cv::Ptr<cv::FaceRecognizerSF> m_recognizer;
    cv::Size m_inputSize;
};

#endif
