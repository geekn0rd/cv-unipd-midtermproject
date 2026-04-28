#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>

class BGRemoveDetecorr
{
public:
    BGRemoveDetecorr();

    cv::Rect run(const std::vector<std::string> &imageFiles);

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
    int morphKernelSize;
};

std::vector<std::string> getImageList(const std::string &folder_name, const std::string &extension);