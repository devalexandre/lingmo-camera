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
#include <cmath>

namespace {
// The model's input
const cv::Size ModelSize(192, 192);
// How fast the mask follows a new result, in seconds: short enough that a waving hand
// isn't left behind, long enough to calm the flicker on the edges
constexpr float SmoothTime = 0.06f;
// Probabilities below Low are background, above High are person; in between is the
// soft edge
constexpr float Low = 0.20f;
constexpr float High = 0.60f;
// Feathering of the edge, in pixels of the quarter-size mask
constexpr double Feather = 1.5;
// The blur is made on a small copy: how much smaller, and the blur there
constexpr int Shrink[] = {4, 8};
constexpr double Sigma[] = {2.0, 3.0};
// Some weight everywhere, so a spot covered by the person still gets a colour
constexpr float MinWeight = 0.02f;
}

Background::Background(const std::string &model, bool threaded)
    : m_threaded(threaded)
{
    try {
        m_net = cv::dnn::readNet(model);
        m_valid = !m_net.empty();
    } catch (const cv::Exception &) {
        m_valid = false;
    }
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

void Background::reset()
{
    std::lock_guard lock(m_mutex);
    m_result.release();
    m_hasMask = false;
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

cv::Mat Background::segment(const cv::Mat &input)
{
    // (x / 255 - 0.5) / 0.5, RGB
    const cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 127.5, ModelSize,
                                                cv::Scalar(127.5, 127.5, 127.5), true, false);
    m_net.setInput(blob);
    const cv::Mat out = m_net.forward();
    // 1 x 2 x H x W softmax: background, person
    return cv::Mat(ModelSize, CV_32F, const_cast<float *>(out.ptr<float>(0, 1))).clone();
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
        m_busy = true;
        lock.unlock();

        cv::Mat result;
        try {
            result = segment(input);
        } catch (const cv::Exception &) {
        }

        lock.lock();
        m_busy = false;
        if (!result.empty())
            m_result = result;
    }
}

void Background::updateAlpha(const cv::Mat &probability)
{
    const auto now = std::chrono::steady_clock::now();
    if (!m_hasMask) {
        probability.copyTo(m_mask);
        m_hasMask = true;
    } else {
        const float dt = std::min(std::chrono::duration<float>(now - m_lastMask).count(), 0.5f);
        const double t = 1 - std::exp(-dt / SmoothTime);
        cv::addWeighted(m_mask, 1 - t, probability, t, 0, m_mask);
    }
    m_lastMask = now;
}

void Background::blurred(const cv::Mat &bgr, Strength strength, cv::Mat &out)
{
    const int shrink = Shrink[strength];
    const cv::Size size(bgr.cols / shrink, bgr.rows / shrink);
    if (size == m_quarter.size())
        m_quarter.copyTo(m_small);
    else
        cv::resize(m_quarter, m_small, size, 0, 0, cv::INTER_AREA);

    // Blur the room without the person in it, or they leave a glow around themselves:
    // each pixel is weighted by how much background it is, and the sum divided by the
    // weights (a normalised convolution)
    cv::Mat weight;
    if (m_alphaSmall.empty())
        weight = cv::Mat(size, CV_32F, cv::Scalar(1));
    else {
        cv::resize(m_alphaSmall, weight, size, 0, 0, cv::INTER_AREA);
        weight.convertTo(weight, CV_32F, -(1 - MinWeight) / 255.0, 1);
    }

    m_weight.create(size, CV_32FC3);
    for (int y = 0; y < size.height; ++y) {
        const uchar *src = m_small.ptr<uchar>(y);
        const float *w = weight.ptr<float>(y);
        float *dst = m_weight.ptr<float>(y);
        for (int x = 0; x < size.width; ++x) {
            dst[3 * x] = src[3 * x] * w[x];
            dst[3 * x + 1] = src[3 * x + 1] * w[x];
            dst[3 * x + 2] = src[3 * x + 2] * w[x];
        }
    }
    cv::GaussianBlur(m_weight, m_weight, cv::Size(), Sigma[strength]);
    cv::GaussianBlur(weight, weight, cv::Size(), Sigma[strength]);
    for (int y = 0; y < size.height; ++y) {
        const float *w = weight.ptr<float>(y);
        float *dst = m_weight.ptr<float>(y);
        uchar *px = m_small.ptr<uchar>(y);
        for (int x = 0; x < size.width; ++x) {
            const float k = 1.0f / std::max(w[x], 1e-4f);
            px[3 * x] = cv::saturate_cast<uchar>(dst[3 * x] * k);
            px[3 * x + 1] = cv::saturate_cast<uchar>(dst[3 * x + 1] * k);
            px[3 * x + 2] = cv::saturate_cast<uchar>(dst[3 * x + 2] * k);
        }
    }
    cv::resize(m_small, out, bgr.size(), 0, 0, cv::INTER_LINEAR);
}

void Background::apply(cv::Mat &bgr, Strength strength)
{
    if (!m_valid || bgr.empty())
        return;

    // One small copy serves the model and the blur
    const cv::Size quarter(bgr.cols / 4, bgr.rows / 4);
    cv::resize(bgr, m_quarter, quarter, 0, 0, cv::INTER_AREA);

    // Segment this frame, or hand it to the worker and take its last result
    cv::Mat input;
    cv::resize(m_quarter, input, ModelSize, 0, 0, cv::INTER_LINEAR);
    cv::Mat result;
    if (m_threaded) {
        std::lock_guard lock(m_mutex);
        std::swap(result, m_result);
        if (!m_busy && m_input.empty()) {
            m_input = input;
            m_wake.notify_one();
        }
    } else {
        result = segment(input);
    }

    if (!result.empty()) {
        updateAlpha(result);
        // Back to the frame's shape, a hard-ish edge, then feathered
        cv::Mat mask;
        cv::resize(m_mask, mask, quarter, 0, 0, cv::INTER_LINEAR);
        mask.convertTo(m_alphaSmall, CV_8U, 255.0 / (High - Low), -255.0 * Low / (High - Low));
        cv::GaussianBlur(m_alphaSmall, m_alphaSmall, cv::Size(), Feather);
        cv::resize(m_alphaSmall, m_alpha, bgr.size(), 0, 0, cv::INTER_LINEAR);
    }
    if (!m_hasMask) {
        // Nothing yet (the first frames): better blur it all than show the room
        m_alphaSmall.release();
        m_alpha = cv::Mat::zeros(bgr.size(), CV_8U);
    } else if (m_alpha.size() != bgr.size()) {
        cv::resize(m_alphaSmall, m_alpha, bgr.size(), 0, 0, cv::INTER_LINEAR);
    }

    if (!m_image.empty()) {
        if (m_imageFitted.size() != bgr.size()) {
            // Cover the frame, cut to its shape
            const double scale = std::max(double(bgr.cols) / m_image.cols, double(bgr.rows) / m_image.rows);
            cv::Mat scaled;
            cv::resize(m_image, scaled, cv::Size(), scale, scale, scale < 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
            const cv::Rect roi((scaled.cols - bgr.cols) / 2, (scaled.rows - bgr.rows) / 2, bgr.cols, bgr.rows);
            m_imageFitted = scaled(roi).clone();
        }
        m_background = m_imageFitted;
    } else {
        blurred(bgr, strength, m_background);
    }

    // Person over the background
    const cv::Mat &alpha = m_alpha;
    const cv::Mat &back = m_background;
    cv::parallel_for_(cv::Range(0, bgr.rows), [&](const cv::Range &rows) {
        for (int y = rows.start; y < rows.end; ++y) {
            uchar *fg = bgr.ptr<uchar>(y);
            const uchar *bg = back.ptr<uchar>(y);
            const uchar *a = alpha.ptr<uchar>(y);
            for (int x = 0; x < bgr.cols; ++x) {
                const int k = a[x];
                for (int c = 0; c < 3; ++c) {
                    const int i = 3 * x + c;
                    fg[i] = uchar((fg[i] * k + bg[i] * (255 - k) + 127) / 255);
                }
            }
        }
    }, 4);
}
