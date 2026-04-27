#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>

class FeatureTrackerR
{
public:
    FeatureTrackerR();
    Rect FeatureTrackerR::run(VideoCapture &capture);

private:
    struct Track
    {
        std::vector<cv::Point2f> history;
        int age = 0;
        int score = 0;
    };

    void initTracks(const std::vector<cv::Point2f> &pts,
                    std::vector<Track> &tracks);

private:
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