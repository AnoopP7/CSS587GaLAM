/*
 * match_test.h
 * Implements testing for comparing the GaLAM outlier detection algorithm to others
 * Implementation authors: Yu Dinh, Neha Kotwal, Anoop Prasad
 * Paper title: GaLAM: Two-Stage Outlier Detection Algorithm
 * Paper authors: X. Lu, Z. Yan, Z. Fan
 *
 * This file defines the MatchTest class for benchmarking feature matching
 * and outlier detection algorithms on the Oxford Affine Dataset.
 *
 * This file runs a benchmarking experiment for different:
 *   - feature descriptors such as SIFT / ORB / AKAZE, and
 *   - outlier-removal methods like NN+RT, RANSAC, GMS, and GaLAM.
 *
 * For each scene and image pair, it:
 *   1. Detects features + descriptors in both images.
 *   2. Finds initial matches with nearest-neighbor + ratio test.
 *   3. Filters matches with the chosen method (RANSAC / GaLAM / or keeps NN+RT).
 *   4. Compares the filtered matches against the ground-truth homography.
 *   5. Logs metrics (% of accurate matches, etc.) to CSV and prints a summary.
 */
 
#ifndef MATCH_TEST_H
#define MATCH_TEST_H

#include "galam.h"

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <vector>
#include <string>
#include <numeric>

class MatchTest {
public:
    // enum for detector types
    enum class Detector { SIFT, ORB, AKAZE };

    // enum for matching/outlier detection types
    enum class Method { NN_RT, RANSAC, GMS, GALAM };

    // Metrics struct
    // Stores evaluation results for a single image pair and method
    struct Metrics {
        int correspondences; // Number of correspondences
        double avg_error;    // Average projection error
        double inlier_pct;   // % inliers
        double he_pct;       // %H.E: error
        double runtime_ms;   // Runtime in milliseconds
    };

    // Constructor
    // Stores which detectors and methods we want to test, provided as parameters
    // Preconditions: Non-empty detectors and methods vectors are provided
    // Postconditions: Creates MatchTest object based on provided parameters
    MatchTest(const std::vector<Detector>& detectors, const std::vector<Method>& methods);

    // runTests
    // Executes the full benchmark suite on the Oxford Affine Dataset
    // Preconditions: Path for dataset and results are provided, and the path for the dataset contains Oxford Affine dataset
    // Postconditions: Outputs summary table of tests as well as logs and saves results and visualizations
    void runTests(const std::string& dataPath, const std::string& csvPath);

private:
    std::vector<Detector> detectors_; //  Feature detectors to test
    std::vector<Method> methods_;     // Outlier filtering methods to test


    // getFeatures
    // Extract keypoints and descriptors from an image using a given detector.
    // Preconditions: Valid parameters provided, and img is a grayscale image
    // Postconditions: The kp and desc vector and Mat provided are populated with keypoints and descriptors
    void getFeatures(
        const cv::Mat& img,
        Detector det,
        std::vector<cv::KeyPoint>& kp,
        cv::Mat& desc);

    // filterOutliers
    // Applies specified outlier filtering method to initial matches
    // Preconditions: Valid keypoints, descriptors, matches, and other parameters provided
    // Postconditions: Returns a filtered set of matches with outlier removal applied and sets runtime_ms
    std::vector<cv::DMatch> filterOutliers(
        Method method,
        const std::vector<cv::KeyPoint>& kp1,
        const std::vector<cv::KeyPoint>& kp2,
        const cv::Mat& d1,
        const cv::Mat& d2,
        const cv::Size& imageSize1,
        const cv::Size& imageSize2,
        const std::vector<cv::DMatch>& matches,
        const std::vector<cv::DMatch>& nnMatches,
        double& runtime_ms);

    // evaluateMatches
    // Computes evaluation metrics by comparing matches against ground truth homography
    // Preconditions: Valid keypoints, matches, and ground truth homography are provided
    // Postconditions: Returns Metrics struct with computed values for the matches provided
    Metrics evaluateMatches(
        const std::vector<cv::KeyPoint>& kp1,
        const std::vector<cv::KeyPoint>& kp2,
        const std::vector<cv::DMatch>& matches,
        const cv::Mat& gtH,
        double runtime_ms);
};

#endif // MATCH_TEST_H