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

#include "background.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <utility>
#include <vector>

namespace {
// The models' picture: RVM was converted for this size (tools/rvm-for-opencv.py)
const cv::Size ModelSize(640, 360);
// The selfie segmenter's own input (the landscape model)
const cv::Size SegmenterSize(256, 144);
// RVM's memory between frames: 1 x channels x rows x columns, for ModelSize
const int Recurrent[4][4] = {{1, 16, 45, 80}, {1, 20, 23, 40}, {1, 40, 12, 20}, {1, 64, 6, 10}};
// The segmenter's probability in [Low, High] becomes 0..1 for the inside of the person...
// (low: a gaming chair right behind the person is half person to both models; better
// all of it than a see-through chair)
constexpr float Low = 0.3f;
constexpr float High = 0.6f;
// ...pulled away from the outline by this many pixels (at ModelSize), where the matte is
// the better judge
constexpr int Inset = 7;
// The final matte's S curve goes from EdgeLow to EdgeHigh
constexpr float EdgeLow = 0.1f;
constexpr float EdgeHigh = 0.9f;
// How much of the previous mask stays where the new one is unsure (0 where it is sure,
// so a moving arm isn't left behind): the segmenter's, then the combined matte's
constexpr float KeepPerson = 0.9f;
constexpr float KeepMatte = 0.8f;
// The blur is made on a small copy: how much smaller than the frame, and the blur there
constexpr int Shrink[] = {4, 8};
constexpr double Sigma[] = {2.0, 3.0};
// With an image, the room behind the soft edge is guessed from a copy this much smaller
constexpr int RoomShrink = 8;
constexpr double RoomSigma = 2.0;
// Some weight everywhere, so a spot covered by the person still gets a colour
constexpr float MinWeight = 0.002f;

// MediaPipe's segmentation smoothing: new + (previous - new) * keep * uncertainty(new),
// where uncertainty is ~1 at 0.5 and 0 at 0 and 1 (a polynomial fit of the entropy)
void smooth(cv::Mat &previous, const cv::Mat &current, float keep)
{
    if (previous.size() != current.size()) {
        current.copyTo(previous);
        return;
    }
    const int n = int(current.total());
    const float *in = current.ptr<float>();
    float *out = previous.ptr<float>();
    for (int i = 0; i < n; ++i) {
        const float t = in[i] - 0.5f;
        const float x = t * t;
        const float certain = std::min(1.0f, x * (5.68842f + x * (-0.748699f + x * (-57.8051f + x * (291.309f + x * -624.717f)))));
        out[i] = in[i] + (out[i] - in[i]) * keep * (1.0f - certain);
    }
}
}

Background::Background(const std::string &segmentationModel, const std::string &mattingModel, bool threaded)
    : m_threaded(threaded)
{
    try {
        m_segmenter = cv::dnn::readNet(segmentationModel);
        m_matting = cv::dnn::readNet(mattingModel);
        m_valid = !m_segmenter.empty() && !m_matting.empty();
    } catch (const cv::Exception &) {
        m_valid = false;
    }
    clearState();
    if (m_valid && m_threaded)
        m_thread = std::thread(&Background::work, this);
}

Background::~Background()
{
    if (m_thread.joinable()) {
        {
            std::lock_guard lock(m_mutex);
            m_quit = true;
        }
        m_wake.notify_all();
        m_thread.join();
    }
}

void Background::clearState()
{
    for (int i = 0; i < 4; ++i)
        m_recurrent[i] = cv::Mat(4, Recurrent[i], CV_32F, cv::Scalar(0));
    m_person.release();
    m_matte.release();
}

void Background::reset()
{
    m_hasMask = false;
    std::lock_guard lock(m_mutex);
    m_result.release();
    if (m_threaded)
        m_clear = true;
    else
        clearState();
}

void Background::setImage(const std::string &path)
{
    if (path == m_imagePath)
        return;
    m_imagePath = path;
    m_image.release();
    m_imageFitted.release();
    if (!path.empty())
        m_image = cv::imread(path, cv::IMREAD_COLOR);
}

cv::Mat Background::estimate(const cv::Mat &input)
{
    // The inside of the person, from the segmenter: RGB 0..1
    m_segmenter.setInput(cv::dnn::blobFromImage(input, 1.0 / 255, SegmenterSize, cv::Scalar(), true, false));
    const cv::Mat segmented = m_segmenter.forward();
    CV_Assert(segmented.total() == size_t(SegmenterSize.area()));
    smooth(m_person, cv::Mat(SegmenterSize, CV_32F, const_cast<float *>(segmented.ptr<float>())), KeepPerson);
    cv::Mat inside;
    cv::resize(m_person, inside, ModelSize, 0, 0, cv::INTER_LINEAR);
    inside.convertTo(inside, CV_32F, 1.0 / (High - Low), -Low / (High - Low));
    cv::erode(inside, inside, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(Inset, Inset)));

    // The outline, from the matting model: RGB 0..1 and its memory of the last frames
    static const std::vector<std::string> names = {"fgr", "pha", "r1o", "r2o", "r3o", "r4o"};
    m_matting.setInput(cv::dnn::blobFromImage(input, 1.0 / 255, ModelSize, cv::Scalar(), true, false), "src");
    for (int i = 0; i < 4; ++i)
        m_matting.setInput(m_recurrent[i], "r" + std::to_string(i + 1) + "i");
    std::vector<cv::Mat> outs;
    m_matting.forward(outs, names);
    CV_Assert(outs.size() == names.size() && outs[1].total() == size_t(ModelSize.area()));
    for (int i = 0; i < 4; ++i)
        m_recurrent[i] = outs[2 + i].clone();

    cv::Mat matte = cv::max(cv::Mat(ModelSize, CV_32F, outs[1].ptr<float>()), inside);
    cv::min(matte, 1.0, matte);
    smooth(m_matte, matte, KeepMatte);
    // A gentle S curve: what both models leave half transparent goes one way or the other,
    // the edge of the hair stays soft
    static const cv::Mat curve = [] {
        cv::Mat lut(1, 256, CV_8U);
        for (int i = 0; i < 256; ++i) {
            const float t = std::clamp((i / 255.0f - EdgeLow) / (EdgeHigh - EdgeLow), 0.0f, 1.0f);
            lut.at<uchar>(i) = cv::saturate_cast<uchar>(255 * t * t * (3 - 2 * t));
        }
        return lut;
    }();
    cv::Mat alpha;
    m_matte.convertTo(alpha, CV_8U, 255.0);
    cv::LUT(alpha, curve, alpha);
    return alpha;
}

void Background::work()
{
    std::unique_lock lock(m_mutex);
    while (true) {
        m_wake.wait(lock, [this] { return m_quit || !m_input.empty(); });
        if (m_quit)
            return;
        cv::Mat input;
        std::swap(input, m_input);
        const bool clear = std::exchange(m_clear, false);
        m_busy = true;
        lock.unlock();

        if (clear)
            clearState();
        cv::Mat result;
        try {
            result = estimate(input);
        } catch (const cv::Exception &) {
            clearState();
        }

        lock.lock();
        m_busy = false;
        if (!result.empty() && !m_clear)
            m_result = result;
    }
}

void Background::room(const cv::Mat &bgr, Strength strength, cv::Mat &out)
{
    // The room without the person in it, blurred: each pixel is weighted by how much
    // background it is and the sum divided by the weights (a normalised convolution), so
    // the person's colours don't spread into it
    const int shrink = m_image.empty() ? Shrink[strength] : RoomShrink;
    const double sigma = m_image.empty() ? Sigma[strength] : RoomSigma;
    const cv::Size size(bgr.cols / shrink, bgr.rows / shrink);
    cv::resize(m_half, m_small, size, 0, 0, cv::INTER_AREA);

    if (m_alphaSmall.empty()) {
        m_weight = cv::Mat(size, CV_32F, cv::Scalar(1));
    } else {
        cv::resize(m_alphaSmall, m_smallAlpha, size, 0, 0, cv::INTER_AREA);
        // A pixel more, so the soft edge (half person) counts as person too
        cv::dilate(m_smallAlpha, m_smallAlpha, cv::Mat());
        m_weight.create(size, CV_32F);
        for (int y = 0; y < size.height; ++y) {
            const uchar *a = m_smallAlpha.ptr<uchar>(y);
            float *w = m_weight.ptr<float>(y);
            for (int x = 0; x < size.width; ++x) {
                const float room = 1.0f - a[x] / 255.0f;
                w[x] = room * room + MinWeight;
            }
        }
    }

    m_sum.create(size, CV_32FC3);
    for (int y = 0; y < size.height; ++y) {
        const uchar *src = m_small.ptr<uchar>(y);
        const float *w = m_weight.ptr<float>(y);
        float *dst = m_sum.ptr<float>(y);
        for (int x = 0; x < size.width; ++x) {
            dst[3 * x] = src[3 * x] * w[x];
            dst[3 * x + 1] = src[3 * x + 1] * w[x];
            dst[3 * x + 2] = src[3 * x + 2] * w[x];
        }
    }
    cv::GaussianBlur(m_sum, m_sum, cv::Size(), sigma);
    cv::GaussianBlur(m_weight, m_weight, cv::Size(), sigma);
    for (int y = 0; y < size.height; ++y) {
        const float *w = m_weight.ptr<float>(y);
        const float *src = m_sum.ptr<float>(y);
        uchar *px = m_small.ptr<uchar>(y);
        for (int x = 0; x < size.width; ++x) {
            const float k = 1.0f / std::max(w[x], 1e-6f);
            px[3 * x] = cv::saturate_cast<uchar>(src[3 * x] * k);
            px[3 * x + 1] = cv::saturate_cast<uchar>(src[3 * x + 1] * k);
            px[3 * x + 2] = cv::saturate_cast<uchar>(src[3 * x + 2] * k);
        }
    }
    cv::resize(m_small, out, bgr.size(), 0, 0, cv::INTER_LINEAR);
}

void Background::apply(cv::Mat &bgr, Strength strength)
{
    if (!m_valid || bgr.empty())
        return;

    // One small copy serves the models and the blur
    cv::resize(bgr, m_half, ModelSize, 0, 0, cv::INTER_AREA);

    // Segment this frame, or hand it to the worker and take its last result
    cv::Mat result;
    if (m_threaded) {
        std::lock_guard lock(m_mutex);
        std::swap(result, m_result);
        if (!m_busy && m_input.empty()) {
            m_input = m_half.clone();
            m_wake.notify_one();
        }
    } else {
        try {
            result = estimate(m_half);
        } catch (const cv::Exception &) {
            clearState();
        }
    }

    if (!result.empty()) {
        m_alphaSmall = result;
        m_hasMask = true;
        cv::resize(m_alphaSmall, m_alpha, bgr.size(), 0, 0, cv::INTER_LINEAR);
    }
    if (!m_hasMask) {
        // Nothing yet (the first frames): better blur it all than show the room
        m_alphaSmall.release();
        m_alpha = cv::Mat::zeros(bgr.size(), CV_8U);
    } else if (m_alpha.size() != bgr.size()) {
        cv::resize(m_alphaSmall, m_alpha, bgr.size(), 0, 0, cv::INTER_LINEAR);
    }

    room(bgr, strength, m_room);
    const bool image = !m_image.empty();
    if (image && m_imageFitted.size() != bgr.size()) {
        // Cover the frame, cut to its shape
        const double scale = std::max(double(bgr.cols) / m_image.cols, double(bgr.rows) / m_image.rows);
        cv::Mat scaled;
        cv::resize(m_image, scaled, cv::Size(), scale, scale, scale < 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
        const cv::Rect roi((scaled.cols - bgr.cols) / 2, (scaled.rows - bgr.rows) / 2, bgr.cols, bgr.rows);
        m_imageFitted = scaled(roi).clone();
    }

    // Person over the background. With an image, a soft edge pixel is part person, part
    // old room: I = a F + (1 - a) R. Putting the image B there means a F + (1 - a) B =
    // a I + (1 - a) B + (1 - a) (I - R); the correction is weighted by a once more, so
    // where it is mostly room the room's own detail (I - blurred R) doesn't come through.
    const cv::Mat &alpha = m_alpha;
    const cv::Mat &back = image ? m_imageFitted : m_room;
    const cv::Mat &roomColour = m_room;
    cv::parallel_for_(cv::Range(0, bgr.rows), [&](const cv::Range &rows) {
        for (int y = rows.start; y < rows.end; ++y) {
            uchar *fg = bgr.ptr<uchar>(y);
            const uchar *bg = back.ptr<uchar>(y);
            const uchar *r = roomColour.ptr<uchar>(y);
            const uchar *a = alpha.ptr<uchar>(y);
            for (int x = 0; x < bgr.cols; ++x) {
                const int k = a[x];
                const int i = 3 * x;
                if (k == 255)
                    continue;
                if (k == 0) {
                    fg[i] = bg[i];
                    fg[i + 1] = bg[i + 1];
                    fg[i + 2] = bg[i + 2];
                    continue;
                }
                // Only the edge gets here
                const int wrap = image ? k * (255 - k) : 0;
                for (int c = i; c < i + 3; ++c) {
                    const int v = (fg[c] * k + bg[c] * (255 - k)) * 255 + wrap * (fg[c] - r[c]);
                    fg[c] = cv::saturate_cast<uchar>((v + 32512) / 65025);
                }
            }
        }
    });
}
