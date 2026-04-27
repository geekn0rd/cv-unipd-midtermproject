#include "FeatureTrackerRilke.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <algorithm>
#include <iostream>

using namespace cv;
using namespace std;

FeatureTrackerR::FeatureTrackerR()
{
    NumberPoints = 1000;
    featureQuality = 0.2f;
    minDistance = 7.0;

    motionThreshold = 2.0f;
    foregroundThreshold = 0.6f;

    lkWindowSize = 15;
    lkMaxLevel = 2;

    morphKernelSize = 5;

    foregroundScoreBoost = 2;
    motionScoreBoost = 1;

    finalScoreThreshold = 30;
    framesToKeep = 2;
}

void FeatureTrackerR::initTracks(const vector<Point2f> &pts,
                                 vector<Track> &tracks)
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

Rect FeatureTrackerR::run(VideoCapture &capture)
{
    Mat first_frame;
    capture >> first_frame;

    if (first_frame.empty())
        return Rect();

    Mat oldGrey;
    cvtColor(first_frame, oldGrey, COLOR_BGR2GRAY);

    Ptr<BackgroundSubtractor> fgbg =
        createBackgroundSubtractorMOG2();

    Mat kernel = getStructuringElement(
        MORPH_ELLIPSE,
        Size(morphKernelSize, morphKernelSize));

    vector<Point2f> p0, p1;
    goodFeaturesToTrack(oldGrey, p0,
                        NumberPoints,
                        featureQuality,
                        minDistance);

    vector<Track> tracks;
    initTracks(p0, tracks);

    Mat prev_frame = first_frame.clone();

    int frame_id = 0;

    while (true)
    {
        Mat frame;
        capture >> frame;
        if (frame.empty())
            break;

        frame_id++;

        Mat grey;
        cvtColor(frame, grey, COLOR_BGR2GRAY);

        Mat fgmask;
        fgbg->apply(frame, fgmask);

        morphologyEx(fgmask, fgmask, MORPH_OPEN, kernel);
        morphologyEx(fgmask, fgmask, MORPH_DILATE, kernel);

        vector<uchar> status;
        vector<float> err;

        calcOpticalFlowPyrLK(
            oldGrey, grey,
            p0, p1,
            status, err,
            Size(lkWindowSize, lkWindowSize),
            lkMaxLevel);

        vector<Point2f> new_p0;
        vector<Track> new_tracks;

        for (size_t i = 0; i < p1.size(); i++)
        {
            if (!status[i])
                continue;

            Point2f prev = p0[i];
            Point2f curr = p1[i];

            if (curr.x < 0 || curr.y < 0 ||
                curr.x >= fgmask.cols ||
                curr.y >= fgmask.rows)
                continue;

            float fg = fgmask.at<uchar>((int)curr.y, (int)curr.x) / 255.0f;
            float motion = norm(curr - prev);

            Track t = tracks[i];

            t.history.push_back(curr);
            t.age++;

            if (fg > foregroundThreshold)
                t.score += foregroundScoreBoost;

            if (motion > motionThreshold)
                t.score += motionScoreBoost;

            new_p0.push_back(curr);
            new_tracks.push_back(t);
        }

        p0 = new_p0;
        tracks = new_tracks;
        oldGrey = grey.clone();
        prev_frame = frame.clone();
    }

    // ===== FINAL BOX =====
    vector<Point2f> bestPoints;

    for (auto &t : tracks)
    {
        if (t.score > finalScoreThreshold)
        {
            int start = max(0, (int)t.history.size() - framesToKeep);

            for (int i = start; i < (int)t.history.size(); i++)
                bestPoints.push_back(t.history[i]);
        }
    }

    if (bestPoints.empty())
        return Rect();

    Rect box = boundingRect(bestPoints);

    int pad = 5;
    box.x -= pad;
    box.y -= pad;
    box.width += 2 * pad;
    box.height += 2 * pad;

    box &= Rect(0, 0, first_frame.cols, first_frame.rows);

    return box;
}