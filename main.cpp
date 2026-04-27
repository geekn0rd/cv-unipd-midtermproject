#include "algorithm.hpp"
#include "metrics.hpp"
#include "FeatureTrackerRilke.hpp"
#include <iostream>

using namespace std;

int main()
{
    DetectionMetrics metrics;
    FeatureTrackerAmirali tracker1;
    FeatureTrackerR tracker2;

    vector<pair<string, string>> data_info = {
        {"bird", "png"},
        {"car", "jpg"},
        {"frog", "png"},
        {"sheep", "jpg"},
        {"squirrel", "png"}};

    for (auto &item : data_info)
    {
        string folder = item.first;
        string ext = item.second;

        cout << "Processing: " << folder << endl;

        string path =
            "dataset/data/" +
            folder +
            "/%04d." + ext;

        string label_path =
            "dataset/labels/" +
            folder +
            "/0000.txt";

        VideoCapture cap(path);

        if (!cap.isOpened())
            continue;

        Mat first_frame;
        Rect pred1 = tracker1.run(cap, first_frame);
        Rect gt = readGroundTruthBox(label_path);
        float IoU = computeIoU(pred1, gt);
        cout << "IoU for " << label_path << ": " << IoU << endl;

        Rect pred2 = tracker2.run(cap);
        Rect gt = readGroundTruthBox(label_path);
        float IoU = computeIoU(pred1, gt);
        cout << "IoU for " << label_path << ": " << IoU << endl;

        metrics.update(IoU);

        // Visualization now lives here
        if (!first_frame.empty() && !pred1.empty())
        {
            Mat result = first_frame.clone();
            rectangle(result, pred1, Scalar(0, 0, 255), 2); // red = predicted
            rectangle(result, gt, Scalar(255, 0, 0), 2);    // blue = ground truth
            imshow(folder, result);                         // use folder name as window title
            waitKey(0);
        }
    }

    cout << "mIoU: "
         << metrics.get_mIoU()
         << endl;

    cout << "Accuracy: "
         << metrics.get_accuracy()
         << endl;
}