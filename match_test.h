/*
 * match_test.h
 * CSS587 Design Project - GaLAM Testing Framework
 * Implementation authors: Yu Dinh, Neha Kotwal, Anoop Prasad
 *
 * This file defines the MatchTest class for benchmarking feature matching
 * and outlier detection algorithms on the Oxford Affine Dataset.
 *
 * Purpose:
 * - Define interface for testing multiple detector/method combinations
 * - Provide metrics matching the GaLAM paper's Table 1 format
 * - Support CSV output for result analysis
 *
 */

#ifndef MATCH_TEST_H
#define MATCH_TEST_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "galam.h"
#include <vector>
#include <string>
#include <numeric>

class MatchTest {
public:
    enum class Detector { SIFT, ORB, AKAZE };
    enum class Method { NN_RT, RANSAC, LOGOS, GMS, GALAM };

    // Metrics struct
    // Stores evaluation results for a single image pair and method
    struct Metrics {
        int correspondences;
        double avg_error;    // Average projection error
        double inlier_pct;   // % inliers
        double he_pct;       // %H.E: error
        double runtime_ms;
    };
    
    MatchTest(const std::vector<Detector>& detectors, const std::vector<Method>& methods);
    void runTests(const std::string& dataPath, const std::string& csvPath);

private:
    std::vector<Detector> detectors_;
    std::vector<Method> methods_;

    void getFeatures(const cv::Mat& img, Detector det,
        std::vector<cv::KeyPoint>& kp, cv::Mat& desc);

    std::vector<cv::DMatch> filterOutliers(Method method,
        const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
        const cv::Mat& d1, const cv::Mat& d2,
        const cv::Size& imageSize1, const cv::Size& imageSize2,
        const std::vector<cv::DMatch>& matches,
        const std::vector<cv::DMatch>& nnMatches,
        const cv::Size& sz1, const cv::Size& sz2, double& runtime_ms);

    Metrics evaluateMatches(const std::vector<cv::KeyPoint>& kp1,
        const std::vector<cv::KeyPoint>& kp2,
        const std::vector<cv::DMatch>& matches,
        const cv::Mat& gtH, double runtime_ms);
};

#endif