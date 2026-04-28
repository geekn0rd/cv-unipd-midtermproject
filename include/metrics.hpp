#ifndef METRICS_HPP
#define METRICS_HPP

#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

// Read ground-truth bounding box from label file
Rect readGroundTruthBox(const string &label_path);

// Compute IoU between predicted and ground truth box
float computeIoU(const Rect &pred, const Rect &gt);

// Check if detection is correct (IoU > 0.5)
bool isTruePositive(float IoU);

// Metrics container
struct DetectionMetrics
{
    float total_IoU = 0.0f;
    int total_objects = 0;
    int true_positives = 0;

    void update(float IoU);
    float get_mIoU() const;
    float get_accuracy() const;
};

#endif