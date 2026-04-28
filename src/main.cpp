#include "amirali.hpp"
#include "metrics.hpp"
#include "rilke.hpp"
#include <iostream>

using namespace std;

int main()
{
    DetectionMetrics metrics;
    OpticalFlowDetecor tracker1;
    BGRemoveDetecorr tracker2;

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

        Rect gt = readGroundTruthBox(label_path);

        // --- tracker1 ---
        VideoCapture cap1(path);
        if (!cap1.isOpened())
            continue;

        Mat first_frame;
        Rect pred1 = tracker1.run(cap1, first_frame);
        float IoU1 = computeIoU(pred1, gt);
        cout << "IoU1 for " << label_path << ": " << IoU1 << endl;

        // --- tracker2 ---
        vector<string> imageFiles = getImageList(folder, ext);
        Rect pred2 = tracker2.run(imageFiles);
        float IoU2 = computeIoU(pred2, gt);
        cout << "IoU2 for " << label_path << ": " << IoU2 << endl;

        metrics.update(max(IoU1, IoU2));

        // Visualization
        if (!first_frame.empty() && !pred1.empty())
        {
            Mat result = first_frame.clone();
            rectangle(result, pred1, Scalar(0, 0, 255), 2); // red = predicted1
            rectangle(result, pred2, Scalar(0, 255, 0), 2); // green = predicted2
            rectangle(result, gt, Scalar(255, 0, 0), 2);    // blue = ground truth
            imshow(folder, result);
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