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

#include "faceengine.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace {
// YuNet's own confidence; below this it also finds faces in the curtains
constexpr float DetectScore = 0.85f;
// The face must take this share of the picture height (about 80 px at 480p,
// someone at arm's length), or SFace sees too few details
constexpr float MinFaceShare = 0.16f;
// A second face at least this big (relative to the first) makes enrollment ambiguous
constexpr float SecondFaceShare = 0.5f;
// Variance of the Laplacian of the aligned 112x112 face: below it, motion blur
constexpr double MinSharpness = 20.0;
// Mean brightness of the face below this: too dark to tell anybody apart
constexpr double MinBrightness = 35.0;

}

FaceEngine::FaceEngine(const std::string &detectorModel, const std::string &recognizerModel)
{
    try {
        m_detector = cv::FaceDetectorYN::create(detectorModel, "", cv::Size(320, 320), DetectScore, 0.3f, 50);
        m_recognizer = cv::FaceRecognizerSF::create(recognizerModel, "");
    } catch (const cv::Exception &) {
        m_detector.release();
        m_recognizer.release();
    }
}

std::vector<Face> FaceEngine::detect(const cv::Mat &bgr)
{
    std::vector<Face> list;
    if (!isValid() || bgr.empty())
        return list;

    if (bgr.size() != m_inputSize) {
        m_detector->setInputSize(bgr.size());
        m_inputSize = bgr.size();
    }

    cv::Mat faces;
    try {
        m_detector->detect(bgr, faces);
    } catch (const cv::Exception &) {
        return list;
    }

    for (int i = 0; i < faces.rows; ++i) {
        Face face;
        face.row = faces.row(i).clone();
        face.box = cv::Rect2f(faces.at<float>(i, 0), faces.at<float>(i, 1),
                              faces.at<float>(i, 2), faces.at<float>(i, 3));
        face.score = faces.at<float>(i, 14);
        for (int p = 0; p < 5; ++p)
            face.points[p] = cv::Point2f(faces.at<float>(i, 4 + 2 * p), faces.at<float>(i, 5 + 2 * p));
        list.push_back(face);
    }
    std::sort(list.begin(), list.end(), [](const Face &a, const Face &b) {
        return a.box.area() > b.box.area();
    });
    return list;
}

FrameProblem FaceEngine::pick(const cv::Mat &bgr, const std::vector<Face> &faces, bool strict, Face &face)
{
    if (faces.empty())
        return FrameProblem::NoFace;

    const Face &first = faces.front();
    if (strict && faces.size() > 1 && faces[1].box.height >= first.box.height * SecondFaceShare)
        return FrameProblem::TooMany;
    if (first.box.height < bgr.rows * MinFaceShare)
        return FrameProblem::TooSmall;

    const cv::Rect inside = cv::Rect(first.box) & cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (inside.area() <= 0)
        return FrameProblem::NoFace;
    cv::Mat gray;
    cv::cvtColor(bgr(inside), gray, cv::COLOR_BGR2GRAY);
    if (cv::mean(gray)[0] < MinBrightness)
        return FrameProblem::Dark;

    cv::Mat aligned, alignedGray, laplacian;
    try {
        m_recognizer->alignCrop(bgr, first.row, aligned);
    } catch (const cv::Exception &) {
        return FrameProblem::NoFace;
    }
    cv::cvtColor(aligned, alignedGray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(alignedGray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    if (stddev[0] * stddev[0] < MinSharpness)
        return FrameProblem::Blurry;

    face = first;
    return FrameProblem::None;
}

cv::Mat FaceEngine::feature(const cv::Mat &bgr, const Face &face)
{
    cv::Mat aligned, feat;
    try {
        m_recognizer->alignCrop(bgr, face.row, aligned);
        m_recognizer->feature(aligned, feat);
    } catch (const cv::Exception &) {
        return cv::Mat();
    }
    feat = feat.reshape(1, 1);
    feat.convertTo(feat, CV_32F);
    const double norm = cv::norm(feat);
    if (!(norm > 1e-6))
        return cv::Mat();
    return feat / norm;
}

float FaceEngine::similarity(const cv::Mat &a, const cv::Mat &b)
{
    if (a.empty() || b.empty() || a.total() != b.total())
        return -1;
    return float(a.dot(b));
}

cv::Mat FaceEngine::homography(const cv::Point2f src[4], const cv::Point2f dst[4])
{
    cv::Mat a(8, 8, CV_64F, cv::Scalar(0)), b(8, 1, CV_64F), x;
    for (int i = 0; i < 4; ++i) {
        double *r0 = a.ptr<double>(i), *r1 = a.ptr<double>(i + 4);
        r0[0] = r1[3] = src[i].x;
        r0[1] = r1[4] = src[i].y;
        r0[2] = r1[5] = 1;
        r0[6] = -src[i].x * dst[i].x;
        r0[7] = -src[i].y * dst[i].x;
        r1[6] = -src[i].x * dst[i].y;
        r1[7] = -src[i].y * dst[i].y;
        b.at<double>(i) = dst[i].x;
        b.at<double>(i + 4) = dst[i].y;
    }
    if (!cv::solve(a, b, x, cv::DECOMP_LU))
        return cv::Mat();
    cv::Mat m(3, 3, CV_64F);
    for (int i = 0; i < 8; ++i)
        m.at<double>(i / 3, i % 3) = x.at<double>(i);
    m.at<double>(2, 2) = 1;
    return m;
}

float FaceEngine::parallax(const Face &reference, const Face &face)
{
    const cv::Point2f src[4] = {reference.points[0], reference.points[1], reference.points[3], reference.points[4]};
    const cv::Point2f dst[4] = {face.points[0], face.points[1], face.points[3], face.points[4]};
    const cv::Mat h = homography(src, dst);
    const float eyes = float(cv::norm(face.points[1] - face.points[0]));
    if (h.empty() || eyes < 1)
        return 0;

    const cv::Point2f n = reference.points[2];
    const double w = h.at<double>(2, 0) * n.x + h.at<double>(2, 1) * n.y + h.at<double>(2, 2);
    if (std::abs(w) < 1e-9)
        return 0;
    const cv::Point2f predicted(float((h.at<double>(0, 0) * n.x + h.at<double>(0, 1) * n.y + h.at<double>(0, 2)) / w),
                                float((h.at<double>(1, 0) * n.x + h.at<double>(1, 1) * n.y + h.at<double>(1, 2)) / w));
    return float(cv::norm(face.points[2] - predicted)) / eyes;
}
