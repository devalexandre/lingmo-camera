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

#ifndef BACKGROUND_H
#define BACKGROUND_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

// Background blur: finds the people in the picture with a small segmentation model
// (PP-HumanSeg) and blurs everything else, or puts an image behind them.
//
// The model takes longer than a frame on a slow CPU, so it runs on its own thread: each
// frame hands it a small copy of the picture when it is free and uses the latest mask it
// made. The mask is smoothed over time and feathered at the edges, so it neither
// flickers nor cuts a hard outline around the hair.
class Background
{
public:
    enum Strength { Light, Strong };

    // threaded = false segments every frame before returning (for tests)
    explicit Background(const std::string &model, bool threaded = true);
    ~Background();
    bool isValid() const { return m_valid; }

    // An image to show instead of the blurred room; "" to blur
    void setImage(const std::string &path);
    // Changes the frame in place
    void apply(cv::Mat &bgr, Strength strength);
    // Forget the last mask (new camera, effect turned back on)
    void reset();

private:
    void work();
    cv::Mat segment(const cv::Mat &input);
    void updateAlpha(const cv::Mat &probability);
    void blurred(const cv::Mat &bgr, Strength strength, cv::Mat &out);

    cv::dnn::Net m_net;
    bool m_valid = false;
    bool m_threaded;

    // Worker
    std::thread m_thread;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    cv::Mat m_input;        // picture waiting to be segmented
    cv::Mat m_result;       // mask the worker made, not taken yet
    bool m_busy = false;
    bool m_quit = false;

    // Mask
    cv::Mat m_mask;         // smoothed person probability, model size
    bool m_hasMask = false;
    std::chrono::steady_clock::time_point m_lastMask;
    cv::Mat m_alphaSmall;   // 0..255, a quarter of the frame
    cv::Mat m_alpha;        // 0..255, frame size

    // Background
    std::string m_imagePath;
    cv::Mat m_image;        // loaded image
    cv::Mat m_imageFitted;  // cut and scaled to the frame
    cv::Mat m_quarter;      // the frame at a quarter of its size
    cv::Mat m_small, m_weight, m_background;
};

#endif // BACKGROUND_H
