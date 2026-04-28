// Author: Amirali
#ifndef AMIRALI_HPP
#define AMIRALI_HPP

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class OpticalFlowDetecor
{
public:
    OpticalFlowDetecor();
    Rect run(VideoCapture &capture, Mat &out_frame);

private:
    void detectFeatures(const Mat &gray, vector<Point2f> &points);
    float computeConfidence(const vector<int> &hits, int frame_id);

    // config
    int MAX_CORNERS;
    double QUALITY_LEVEL;
    double MIN_DISTANCE;
    float MOTION_THRESHOLD;
    int MAX_FRAMES;
    int MIN_HITS;
    float CONFIDENCE_THRESHOLD;
};

#endif