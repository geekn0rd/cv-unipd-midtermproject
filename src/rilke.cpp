#include "rilke.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <algorithm>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

FeatureTracker::FeatureTracker()
{
    morphKernelSize = 5;
}

void FeatureTracker::initTracks(const vector<Point2f> &pts, vector<Track> &tracks)
{
    tracks.clear();
    for (auto &pt : pts)
    {
        Track t;
        t.history.push_back(pt);
        t.age = 1;
        t.score = 0;
        tracks.push_back(t);
    }
}

Rect FeatureTracker::run(const vector<string> &imageFiles)
{
    if (imageFiles.empty())
        return Rect();

    vector<string> files = imageFiles;
    reverse(files.begin(), files.end());

    // Background subtractor
    Ptr<BackgroundSubtractor> fgbg =
        createBackgroundSubtractorMOG2(500, 32, false);

    // Morphology kernel for mask cleanup
    Mat kernel = getStructuringElement(
        MORPH_ELLIPSE,
        Size(morphKernelSize, morphKernelSize));

    // ================= BUILD BACKGROUND MODEL =================

    Mat lastValidFrame;

    for (size_t f = 0; f < files.size(); f++)
    {
        Mat frame = imread(files[f]);

        if (frame.empty())
            continue;

        frame.copyTo(lastValidFrame); // Keep last valid frame for calculating final box

        Mat fgmask;
        fgbg->apply(frame, fgmask, 0.02);

        morphologyEx(fgmask, fgmask, MORPH_OPEN, kernel); // Clean up noise
        morphologyEx(fgmask, fgmask, MORPH_DILATE, kernel);
    }

    if (lastValidFrame.empty())
        return Rect();

    // ================= FINAL BOX =================
    // Start from last frame and move backward until valid mask found

    for (int i = (int)files.size() - 1; i >= 0; i--)
    {
        Mat frame = imread(files[i]);

        if (frame.empty())
            continue;

        Mat finalMask;
        fgbg->apply(frame, finalMask, 0);

        morphologyEx(finalMask, finalMask, MORPH_OPEN, kernel);
        morphologyEx(finalMask, finalMask, MORPH_DILATE, kernel);

        // Optional threshold for stronger foreground only
        threshold(finalMask, finalMask, 50, 255, THRESH_BINARY);

        vector<Point> points;
        findNonZero(finalMask, points);

        if (!points.empty())
        {
            Rect box = boundingRect(points);
            return box;
        }
    }

    // Nothing found
    return Rect();
}

namespace fs = std::filesystem;

std::vector<std::string> getImageList(const std::string &folder_name, const std::string &extension)
{
    std::vector<std::string> files;
    // Construct the path to the specific category folder
    fs::path dir_path = fs::path("dataset/data") / folder_name;

    if (!fs::exists(dir_path) || !fs::is_directory(dir_path))
    {
        return files;
    }

    // Iterate through directory
    for (const auto &entry : fs::directory_iterator(dir_path))
    {
        if (entry.is_regular_file() && entry.path().extension() == "." + extension)
        {
            files.push_back(entry.path().string());
        }
    }

    // Sort paths alphabetically so the sequence is in order (0000, 0001, ...)
    std::sort(files.begin(), files.end());

    return files;
}