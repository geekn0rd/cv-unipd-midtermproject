#include "algorithm.hpp"
#include "metrics.hpp"
#include <iostream>

using namespace std;

int main()
{
    DetectionMetrics metrics;
    FeatureTracker tracker;

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
        Rect pred = tracker.run(cap, first_frame);
        Rect gt = readGroundTruthBox(label_path);
        float IoU = computeIoU(pred, gt);

        metrics.update(IoU);

        cout << "IoU for " << label_path << ": " << IoU << endl;

        // Visualization now lives here
        if (!first_frame.empty() && !pred.empty())
        {
            Mat result = first_frame.clone();
            rectangle(result, pred, Scalar(0, 0, 255), 2); // red = predicted
            rectangle(result, gt, Scalar(255, 0, 0), 2);   // blue = ground truth
            imshow(folder, result);                        // use folder name as window title
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