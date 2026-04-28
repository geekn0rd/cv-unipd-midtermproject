#include "metrics.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

Rect readGroundTruthBox(const string &label_path)
{
    ifstream file(label_path);

    if (!file.is_open())
    {
        cout << "Error opening label file: "
             << label_path << endl;

        return Rect();
    }

    int x1, y1, x2, y2;

    file >> x1 >> y1 >> x2 >> y2;

    file.close();

    int width = x2 - x1;
    int height = y2 - y1;

    return Rect(x1, y1, width, height);
}

float computeIoU(const Rect &pred, const Rect &gt)
{
    int x_left = max(pred.x, gt.x);

    int y_top = max(pred.y, gt.y);

    int x_right = min(pred.x + pred.width, gt.x + gt.width);

    int y_bottom = min(pred.y + pred.height, gt.y + gt.height);

    // No overlap
    if (x_right <= x_left || y_bottom <= y_top)
        return 0.0f;

    int intersection_area = (x_right - x_left) * (y_bottom - y_top);

    int pred_area = pred.width * pred.height;

    int gt_area = gt.width * gt.height;

    int union_area = pred_area + gt_area - intersection_area;

    return (float)intersection_area / (float)union_area;
}

bool isTruePositive(float IoU)
{
    return IoU > 0.5f;
}

// ---- Metrics struct functions ----

void DetectionMetrics::update(float IoU)
{
    total_IoU += IoU;
    total_objects++;

    if (IoU > 0.5f)
        true_positives++;
}

float DetectionMetrics::get_mIoU() const
{
    if (total_objects == 0)
        return 0.0f;

    return total_IoU /
           total_objects;
}

float DetectionMetrics::get_accuracy() const
{
    if (total_objects == 0)
        return 0.0f;

    return (float)true_positives /
           total_objects;
}