#include "amirali.hpp"
#include "metrics.hpp"
#include <iostream>

FeatureTrackerAmirali::FeatureTrackerAmirali()
{
    // Number of feature points we detect
    MAX_CORNERS = 80;

    // Controls how strong a corner must be to be accepted
    QUALITY_LEVEL = 0.1;

    MIN_DISTANCE = 5.0;

    // Minimum displacement in pixels
    MOTION_THRESHOLD = 2.0f;
    MAX_FRAMES = 25;

    // Number of the frames a point must have moved to be counted as part of the object
    MIN_HITS = 2;

    // Stopping criteria
    CONFIDENCE_THRESHOLD = 0.2f;
}

void FeatureTrackerAmirali::detectFeatures(const Mat &gray, vector<Point2f> &points)
{
    // Shi-Tomasi corner detector
    goodFeaturesToTrack(gray, points, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE);
    std::cout << "[detectFeatures] raw corners: " << points.size() << std::endl;

    if (!points.empty())
    {
        cornerSubPix(
            gray,
            points,
            Size(5, 5),
            Size(-1, -1),
            TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));

        std::cout << "[detectFeatures] refined corners: " << points.size() << std::endl;
    }
}

// How sure are we that we've found a moving object
float FeatureTrackerAmirali::computeConfidence(const vector<int> &hits, int frame_id)
{
    int strong = 0;

    for (int h : hits)
        if (h >= MIN_HITS)
            strong++;

    float ratio = (float)strong / max(1, (int)hits.size());

    float time_factor = min(1.0f, frame_id / (float)MAX_FRAMES);

    return 0.8f * ratio + 0.2f * time_factor;
}

Rect FeatureTrackerAmirali::run(VideoCapture &capture, Mat &out_frame)
{
    Mat first_frame, first_gray;
    capture >> first_frame;

    cvtColor(first_frame, first_gray, COLOR_BGR2GRAY);

    vector<Point2f> p0;
    detectFeatures(first_gray, p0);

    // Anchor
    vector<Point2f> frame0_pts = p0;
    vector<int> motion_hits(p0.size(), 0);

    Mat prev_gray = first_gray.clone();

    int frame_id = 0;
    bool locked = false;
    float confidence = 0;

    while (frame_id < MAX_FRAMES && !locked)
    {
        Mat frame, gray;
        capture >> frame;
        if (frame.empty())
            break;

        frame_id++;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Point2f> p1;
        vector<uchar> status;
        vector<float> err;

        // Lucas-Kanade Pyramidal Optical Flow
        calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            p0,
            p1,
            status,
            err);

        vector<Point2f> next_p0;
        vector<Point2f> next_frame0_pts;
        vector<int> next_motion_hits;

        for (size_t i = 0; i < p0.size(); i++)
        {
            if (!status[i])
                continue; // drop lost points entirely

            float d = norm(p1[i] - frame0_pts[i]);

            if (d > MOTION_THRESHOLD)
                next_motion_hits.push_back(motion_hits[i] + 1);
            else
                next_motion_hits.push_back(motion_hits[i]);

            next_p0.push_back(p1[i]);                 // current position for next LK call
            next_frame0_pts.push_back(frame0_pts[i]); // original position preserved
        }

        p0 = next_p0;
        frame0_pts = next_frame0_pts;
        motion_hits = next_motion_hits;

        prev_gray = gray.clone();

        confidence = computeConfidence(motion_hits, frame_id);

        if (confidence > CONFIDENCE_THRESHOLD && frame_id > 5)
        {
            locked = true;
        }
    }

    vector<Point> object_pts;

    for (size_t i = 0; i < motion_hits.size(); i++)
        if (motion_hits[i] >= MIN_HITS)
            object_pts.push_back(frame0_pts[i]);

    if (object_pts.size() < 3)
    {
        cout << "[FAIL] No object found!\n";
        return Rect();
    }

    Rect pred_box = boundingRect(object_pts);

    // Add padding
    int pad = 5;
    pred_box.x -= pad;
    pred_box.y -= pad;
    pred_box.width += 2 * pad;
    pred_box.height += 2 * pad;

    // Clamp to frame boundaries so the rect doesn't go negative or overflow
    pred_box &= Rect(0, 0, first_frame.cols, first_frame.rows);

    out_frame = first_frame.clone();
    return pred_box;
}