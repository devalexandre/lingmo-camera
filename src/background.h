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

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

// Background blur: finds the people in the picture and blurs everything else, or puts an
// image behind them.
//
// Two small models look at a 640x360 copy of each frame, on their own thread (they take
// longer than a frame on a slow CPU; each frame hands the worker a copy when it is free
// and uses the latest mask it made):
// - Robust Video Matting draws the outline: a soft alpha matte that keeps the strands of
//   hair and the thin headset band, and follows the person from frame to frame (it is
//   recurrent, so it doesn't flicker).
// - MediaPipe's selfie segmenter (the model behind Google Meet's blur) fills the inside:
//   matting leaves dark clothes on a dark chair half transparent, the segmenter doesn't.
// The mask is made at a quarter of the frame and scaled up; the room is blurred with the
// person taken out of it, so they don't leave a glow, and with an image the colour of the
// room is taken out of the soft edge, so the hair doesn't keep a rim of the old wall.
class Background
{
public:
    enum Strength { Light, Strong };

    // threaded = false segments every frame before returning (for tests)
    Background(const std::string &segmentationModel, const std::string &mattingModel,
               bool threaded = true);
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
    // Worker side: a ModelSize picture in, the person's alpha (CV_8U, ModelSize) out
    cv::Mat estimate(const cv::Mat &input);
    void clearState();
    void room(const cv::Mat &bgr, Strength strength, cv::Mat &out);

    cv::dnn::Net m_segmenter;
    cv::dnn::Net m_matting;
    bool m_valid = false;
    bool m_threaded;

    // Worker
    std::thread m_thread;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    cv::Mat m_input;        // picture waiting to be segmented
    cv::Mat m_result;       // alpha the worker made, not taken yet
    bool m_busy = false;
    bool m_quit = false;
    bool m_clear = false;   // start the models' memory over before the next picture

    // Models' memory, worker only
    cv::Mat m_recurrent[4]; // RVM's state between frames
    cv::Mat m_person;       // smoothed segmenter output
    cv::Mat m_matte;        // smoothed combined alpha, float

    // Camera thread
    bool m_hasMask = false;
    cv::Mat m_half;         // the frame at the models' size
    cv::Mat m_alphaSmall;   // latest alpha, 0..255, models' size
    cv::Mat m_alpha;        // 0..255, frame size

    // Background
    std::string m_imagePath;
    cv::Mat m_image;        // loaded image
    cv::Mat m_imageFitted;  // cut and scaled to the frame
    cv::Mat m_small, m_smallAlpha, m_weight, m_sum, m_room;
};

#endif // BACKGROUND_H
