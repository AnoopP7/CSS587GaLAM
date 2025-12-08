/*
 * match_test.h
 * Implements the GaLAM outlier detection algorithm in C++
 * Implementation authors: Yu Dinh, Neha Kotwal, Anoop Prasad
 * Paper title: GaLAM: Two-Stage Outlier Detection Algorithm
 * Paper authors: X. Lu, Z. Yan, Z. Fan
 *
 * This file defines the MatchTest class for benchmarking feature matching
 * and outlier detection algorithms on the Oxford Affine Dataset.
 *
 * Purpose:
 * - Define interface for testing multiple detector/method combinations
 * - Provide metrics matching the GaLAM paper's Table 1 format
 * - Support CSV output for result analysis
 * 
 * * Metrics computed:
 * - Correspondences: Number of matches after filtering
 * - Average Error: Mean reprojection error in pixels
 * - Inlier %: Percentage of matches with error < 3px (AP metric)
 * - %H.E: Percentage of matches with error < 1px (Homography Estimation)
 * - Runtime: Processing time in milliseconds
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
    enum class Method { NN_RT, RANSAC, GMS, GALAM };

    // Metrics struct
    // Stores evaluation results for a single image pair and method
    struct Metrics {
        int correspondences;
        double avg_error;    // Average projection error
        double inlier_pct;   // % inliers
        double he_pct;       // %H.E: error
        double runtime_ms;
    };
    // Constructor
    // Initializes the test framework with specified detectors and methods
    // Preconditions: detectors and methods vectors are non-empty
    // Postconditions: MatchTest object is ready to run tests
    
    // runTests
    // Executes the full benchmark suite on the Oxford Affine Dataset
    // Preconditions: dataPath contains valid Oxford Affine Dataset structure
    // Postconditions: Results are printed to console and saved to CSV file
    
    MatchTest(const std::vector<Detector>& detectors, const std::vector<Method>& methods);
    void runTests(const std::string& dataPath, const std::string& csvPath);

private:
    std::vector<Detector> detectors_; //  Feature detectors to test
    std::vector<Method> methods_;     // Outlier filtering methods to test


    // getFeatures
    // Extracts keypoints and descriptors from an image using specified detector
    // Preconditions: img is a valid grayscale image
    // Postconditions: kp and desc are populated with detected features 
    void getFeatures(const cv::Mat& img, Detector det,
        std::vector<cv::KeyPoint>& kp, cv::Mat& desc);

    // filterOutliers
    // Applies specified outlier filtering method to initial matches
    // Preconditions: Valid keypoints, descriptors, and matches are provided
    // Postconditions: Returns filtered matches and sets runtime_ms
    std::vector<cv::DMatch> filterOutliers(Method method,
        const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
        const cv::Mat& d1, const cv::Mat& d2,
        const cv::Size& imageSize1, const cv::Size& imageSize2,
        const std::vector<cv::DMatch>& matches,
        const std::vector<cv::DMatch>& nnMatches,
        const cv::Size& sz1, const cv::Size& sz2, double& runtime_ms);

    // evaluateMatches
    // Computes evaluation metrics by comparing matches against ground truth homography
    // Preconditions: Valid keypoints, matches, and ground truth homography are provided
    // Postconditions: Returns Metrics struct with computed values

    Metrics evaluateMatches(const std::vector<cv::KeyPoint>& kp1,
        const std::vector<cv::KeyPoint>& kp2,
        const std::vector<cv::DMatch>& matches,
        const cv::Mat& gtH, double runtime_ms);
};

#endif