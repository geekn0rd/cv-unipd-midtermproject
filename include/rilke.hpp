#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

class FeatureTrackerR
{
public:
    FeatureTrackerR();

    cv::Rect run(cv::VideoCapture &capture);

private:
    struct Track
    {
        std::vector<cv::Point2f> history;
        int age = 0;
        int score = 0;
    };

    void initTracks(const std::vector<cv::Point2f> &pts,
                    std::vector<Track> &tracks);

    // ===== parameters =====
    int NumberPoints;
    float featureQuality;
    double minDistance;

    float motionThreshold;
    float foregroundThreshold;

    int lkWindowSize;
    int lkMaxLevel;

    int morphKernelSize;

    int foregroundScoreBoost;
    int motionScoreBoost;

    int finalScoreThreshold;
    int framesToKeep;
};