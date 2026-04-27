#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Configuration Constants
const int MAX_CORNERS = 50;
const double QUALITY_LEVEL = 0.1;
const double MIN_DISTANCE = 7.0;
const int MIN_POINTS_TO_TRACK = 50; // Re-detect if points drop below this

void detectFeatures(const Mat& gray, vector<Point2f>& points) {
    goodFeaturesToTrack(gray, points, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, Mat(), 7, false, 0.04);
    if (!points.empty()) {
        cornerSubPix(gray, points, Size(5, 5), Size(-1, -1), 
                     TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <folder_path>" << endl;
        return -1;
    }

    string folder = argv[1];

    string path_jpg = "dataset/data/" + folder + "/%04d.jpg";
    string path_png = "dataset/data/" + folder + "/%04d.png";

    VideoCapture capture(path_jpg);

    if (!capture.isOpened()) {
        capture.open(path_png);
    }

    if (!capture.isOpened()) {
        cerr << "Error: Could not open image sequence." << endl;
        return -1;
    }

    cout << "Sequence opened successfully!" << endl;

    Mat old_frame, old_gray, mask;
    vector<Point2f> p0, p1;
    vector<Scalar> colors;
    RNG rng;

    // Initialize random colors
    for (int i = 0; i < MAX_CORNERS; i++) {
        colors.push_back(Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
    }

    capture >> old_frame;
    if (old_frame.empty()) return -1;
    imshow("old_frame", old_frame);
    waitKey();
    
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    detectFeatures(old_gray, p0);
    mask = Mat::zeros(old_frame.size(), old_frame.type());

    while (true) {
        Mat frame, frame_gray, display_img;
        capture >> frame;
        if (frame.empty()) break;

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // 1. Re-detect points if they are getting sparse
        if (p0.size() < MIN_POINTS_TO_TRACK) {
            vector<Point2f> new_points;
            detectFeatures(old_gray, new_points);
            p0.insert(p0.end(), new_points.begin(), new_points.end());
        }

        if (p0.empty()) {
            old_gray = frame_gray.clone();
            continue;
        }

        // 2. Optical Flow
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(30, 30), 3);

        vector<Point2f> good_new;
        vector<Point2f> moving_pts; // Specifically for the bounding box
        float motion_threshold = 2.0;

        for (size_t i = 0; i < p0.size(); i++) {
            if (status[i]) {
                good_new.push_back(p1[i]);
                
                float motion = norm(p1[i] - p0[i]);
                
                // Only use points with significant motion for the "Object Square"
                if (motion > motion_threshold) {
                    moving_pts.push_back(p1[i]);
                    line(mask, p1[i], p0[i], colors[i % colors.size()], 2);
                }
                circle(frame, p1[i], 3, colors[i % colors.size()], -1);
            }
        }

        add(frame, mask, display_img);

        // 3. Dynamic Square around MOVING points only
        if (moving_pts.size() > 5) {
            Rect boundingBox = boundingRect(moving_pts);
            
            // Convert Rect to Square (taking the larger dimension)
            int side = max(boundingBox.width, boundingBox.height);
            Point center(boundingBox.x + boundingBox.width/2, boundingBox.y + boundingBox.height/2);
            Rect square(center.x - side/2, center.y - side/2, side, side);

            rectangle(display_img, square, Scalar(0, 0, 255), 3);
            cout << boundingBox << endl;
        }

        imshow("Improved Tracking", display_img);

        // One waitKey is enough
        if (waitKey(100) == 'q') break;

        // Update for next iteration
        old_gray = frame_gray.clone();
        p0 = good_new;

        // Optional: Fade the mask over time so trails don't clutter the screen
        mask *= 0.95; 
    }

    return 0;
}