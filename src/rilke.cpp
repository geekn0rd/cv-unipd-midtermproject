#include "rilke.hpp"

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
    minDistance = 4.0;

    motionThreshold = 1.5f;
    foregroundThreshold = 0.5f;

    lkWindowSize = 15;
    lkMaxLevel = 2;

    morphKernelSize = 5;

    foregroundScoreBoost = 4;
    motionScoreBoost = 1;

    finalScoreThreshold = 40;
    framesToKeep = 2;
}

void FeatureTrackerR::initTracks(const vector<Point2f> &pts, vector<Track> &tracks)
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
    if (!capture.isOpened())
        return Rect();

    // buffer all remaining frames
    vector<Mat> frames;
    Mat tmp;
    while (capture.read(tmp))
        frames.push_back(tmp.clone());

    if (frames.empty())
        return Rect();

    Ptr<BackgroundSubtractor> fgbg =
        createBackgroundSubtractorMOG2(500, 16, false);

    Mat kernel = getStructuringElement(
        MORPH_ELLIPSE,
        Size(morphKernelSize, morphKernelSize));

    Mat firstFrame = frames[0];

    Mat oldGrey;
    cvtColor(firstFrame, oldGrey, COLOR_BGR2GRAY);

    vector<Point2f> p0, p1;
    goodFeaturesToTrack(oldGrey, p0, NumberPoints, featureQuality, minDistance);

    if (p0.empty())
        return Rect();

    vector<Track> tracks;
    initTracks(p0, tracks);

    Mat lastFrame = firstFrame.clone();

    // ================= MAIN LOOP =================
    for (size_t f = 1; f < frames.size(); f++)
    {
        Mat frame = frames[f];

        if (frame.empty())
            continue;

        frame.copyTo(lastFrame);

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

        // ================= ADAPTIVE WEIGHT =================

        double fgPixels = countNonZero(fgmask);
        double totalPixels = fgmask.rows * fgmask.cols;

        float fgStrength = (totalPixels > 0)
                               ? (float)(fgPixels / totalPixels)
                               : 0.5f;

        float bgWeight = std::clamp(fgStrength, 0.1f, 0.9f);
        float flowWeight = 1.0f - bgWeight;

        // ================= FUSION SCORE MAP =================

        Mat score = Mat::zeros(frame.size(), CV_32F);

        for (size_t i = 0; i < p1.size(); i++)
        {
            if (!status[i])
                continue;

            Point2f prev = p0[i];
            Point2f curr = p1[i];

            if (curr.x < 0 || curr.y < 0 ||
                curr.x >= fgmask.cols || curr.y >= fgmask.rows)
                continue;

            float fg = fgmask.at<uchar>((int)curr.y, (int)curr.x) / 255.0f;

            float motion = norm(curr - prev);
            float motionScore = std::min(motion / 5.0f, 1.0f);

            float confidence =
                bgWeight * fg +
                flowWeight * motionScore;

            score.at<float>((int)curr.y, (int)curr.x) += confidence;
        }

        normalize(score, score, 0, 255, NORM_MINMAX);

        Mat finalMask;
        score.convertTo(finalMask, CV_8U);

        threshold(finalMask, finalMask, 80, 255, THRESH_BINARY);

        // ================= BOUNDING BOX =================

        vector<Point> points;
        findNonZero(finalMask, points);

        if (!points.empty())
        {
            Rect box = boundingRect(points);

            rectangle(frame, box, Scalar(0, 0, 255), 2);

            imshow("Tracking", frame);
            waitKey(30);
        }

        // update flow
        p0 = p1;
        oldGrey = grey.clone();
    }

    // ================= FINAL OUTPUT =================

    Mat finalMask;
    fgbg->apply(lastFrame, finalMask);

    morphologyEx(finalMask, finalMask, MORPH_OPEN, kernel);
    morphologyEx(finalMask, finalMask, MORPH_DILATE, kernel);

    vector<Point> points;
    findNonZero(finalMask, points);

    if (points.empty())
        return Rect();

    return boundingRect(points);
}