/*
 * match_test.cpp
 
 *
 * This file runs a benchmarking experiment for different:
 *   - feature descriptors such as SIFT / ORB / AKAZE, and
 *   - outlier-removal methods like NN+RT, RANSAC, GaLAM.
 *
 * For each scene and image pair, it:
 *   1. Detects features + descriptors in both images.
 *   2. Finds initial matches with nearest-neighbor + ratio test.
 *   3. Filters matches with the chosen method (RANSAC / GaLAM / or keeps NN+RT).
 *   4. Compares the filtered matches against the ground-truth homography.
 *   5. Logs metrics (% of accurate matches, etc.) to CSV and prints a summary.
 */

#include "match_test.h"

//#include<opencv2/xfeatures2d.hpp>;

#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>

namespace fs = std::filesystem;
/// Constructor simply stores which detectors and methods we want to test
MatchTest::MatchTest(const std::vector<Detector>& detectors, const std::vector<Method>& methods)
    : detectors_(detectors), methods_(methods) {}
/**
 * Extract keypoints and descriptors from an image using a given detector.
 *
 * img   : input grayscale image
 * det   : which descriptor to use SIFT, ORB, AKAZE
 * kp    : output keypoints
 * desc  : output descriptors i.e.one row per keypoint
 */

void MatchTest::getFeatures(const cv::Mat& img, Detector det,
                            std::vector<cv::KeyPoint>& kp, cv::Mat& desc) {
    cv::Ptr<cv::Feature2D> feature;
    switch (det) {
        case Detector::SIFT:  feature = cv::SIFT::create(); break;
        case Detector::ORB:   feature = cv::ORB::create(5000); break;
        case Detector::AKAZE: feature = cv::AKAZE::create(); break;
    }
    feature->detectAndCompute(img, cv::noArray(), kp, desc);
}
/**
 * Apply an outlier-filtering method to an initial set of matches.
 *
 * method   : which algorithm to use NN+RT baseline, RANSAC, GaLAM
 * kp1/kp2  : keypoints for image 1 and 2
 * d1/d2    : descriptors for image 1 and 2
 * matches  : initial matches after NN + ratio test
 * sz1/sz2  : image sizes (used by GaLAM)
 * runtime_ms: output – how long this filtering took, in milliseconds
 *
 * Returns a new set of matches after outlier removal.
 */
std::vector<cv::DMatch> MatchTest::filterOutliers(Method method,
    const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
    const cv::Mat& d1, const cv::Mat& d2,
    const std::vector<cv::DMatch>& matches,
    const std::vector<cv::DMatch>& nnMatches,
    const cv::Size& sz1, const cv::Size& sz2, double& runtime_ms) {
    //  Start timer for this method
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<cv::DMatch> result;

    switch (method) {
        case Method::NN_RT:
        // Baseline: do NOT remove outliers any further
        // Use NN+ratio-test matches directly
            result = nnMatches;
            break;

        case Method::RANSAC:
        // Use OpenCV's findHomography + RANSAC to filter mismatches.
            if (matches.size() >= 4) {
                std::vector<cv::Point2f> p1, p2;
                // Collect corresponding 2D points from keypoints
                for (const auto& match : matches) {
                    p1.push_back(kp1[match.queryIdx].pt);
                    p2.push_back(kp2[match.trainIdx].pt);
                }
                std::vector<char> mask;//// 0/1 flag indicating which matches are inliers
                cv::findHomography(p1, p2, cv::RANSAC, 3.0, mask);
                // Keep only inliers according to the RANSAC mask
                for (size_t i = 0; i < matches.size(); ++i)
                    if (mask[i]) result.push_back(matches[i]);
            }
            break;

        case Method::GMS:

            break;

        case Method::LOGOS:

            break;

        case Method::GALAM: 
            // Our custom GaLAM implementation: two-stage outlier detection.
            // 1) Local affine verification around seed matches.
            // 2) Global geometric verification with PROSAC + fundamental matrix.
            GaLAM::InputParameters params;// uses default parameters
            GaLAM galam(params);
            std::vector<cv::KeyPoint> k1 = kp1, k2 = kp2;
            // detectOutliers returns a struct here we only need the final matches
            result = galam.detectOutliers(k1, k2, d1, d2, matches, sz1, sz2).finalMatches;
            break;
        
    }
    // Stop timer and compute elapsed time in milliseconds
    auto end = std::chrono::high_resolution_clock::now();
    runtime_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}
/**
 * Evaluate how good the final matches are using the ground-truth homography.
 *
 * kp1, kp2      : keypoints in image 1 and 2
 * matches       : filtered matches we want to evaluate
 * gtHomography  : ground-truth homography that maps points in img1 -> img2
 * runtime_ms  : time taken by the outlier-filtering step (just passed through)
 *
 * Returns Metrics:
 *   - correspondences : number of matches
 *   - he_pct          : % of matches with < 1px reprojection error (%H.E)
 *   - ap_pct          : % of matches with < 3px reprojection error (AP)
 *   - runtime_ms      : passed-through timing
 */
MatchTest::Metrics MatchTest::evaluateMatches(
    const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches, const cv::Mat& gtHomography, double runtime_ms) {
    
    Metrics met;
    met.correspondences = (int)matches.size();
    met.avg_error = 0.0;
    met.inlier_pct = 0.0;
    met.he_pct = 0.0;
    met.runtime_ms = runtime_ms;
    //If we have no matches or no ground-truth, we can't evaluate anything
    if (matches.empty() || gtHomography.empty()) return met;
    
    // Convert ground-truth homography to double for safer math.
    cv::Mat homography;
    gtHomography.convertTo(homography, CV_64F);
    int he_count = 0, inlier_count = 0;
    double total_error = 0.0;

    //   For each match:
    //   1) take point in img1
    //   2) warp it using homography to predict its location in img2
    //   3) measure Euclidean distance to actual point in img2
    for (const auto& match : matches) {
        // Homogeneous transform: homography * [x, y, 1]^T as mentioned in paper
        cv::Point2f p1 = kp1[match.queryIdx].pt, p2 = kp2[match.trainIdx].pt;
        double scalingW = homography.at<double>(2,0)*p1.x + homography.at<double>(2,1)*p1.y + homography.at<double>(2,2);
        double projX = (homography.at<double>(0,0)*p1.x + homography.at<double>(0,1)*p1.y + homography.at<double>(0,2)) / scalingW;
        double projY = (homography.at<double>(1,0)*p1.x + homography.at<double>(1,1)*p1.y + homography.at<double>(1,2)) / scalingW;
        double error = std::sqrt((projX-p2.x)*(projX-p2.x) + (projY-p2.y)*(projY-p2.y));
        
        total_error += error;
        if (error < 1.0) ++he_count;// Count as high-accuracy if error < 1 pixel
        if (error < 3.0) ++inlier_count;//Convert counts to percentages of total matches
    }
    // Convert counts to percentages of total matche
    met.avg_error = total_error / matches.size();
    met.he_pct = 100.0 * he_count / matches.size();
    met.inlier_pct = 100.0 * inlier_count / matches.size();
    return met;
}
/**
 * Main benchmarking loop:
 *
 *  - Iterates over HPatches-like scenes (bark, bikes, ...).
 *  - For each scene and each image pair (img1 vs img2..img6):
 *      * Loads ground-truth homography H1toX.
 *      * For each detector (SIFT/ORB/AKAZE):
 *          - Extracts features and descriptors in both images.
 *          - Performs NN matching + ratio test (0.8).
 *          - For each method (NN+RT, RANSAC, GaLAM):
 *              + Filters matches.
 *              + Evaluates %H.E, AP, and runtime.
 *              + Logs results to CSV and console.
 *  - Finally prints a summary table for:
 *      * viewpoint scenes (graf, wall)
 *      * illumination scenes (leuven)
 */
void MatchTest::runTests(const std::string& dataPath, const std::string& csvPath) {
    // Scene names from HPatches dataset
    std::vector<std::string> scenes = {"bark","bikes","boat","graf","leuven","trees","ubc","wall"};
    
    // Open output CSV file and write header row
    std::ofstream csv(csvPath);
    csv << "Scene,Pair,Method,Correspondences,AvgError,Inlier%,H.E%,Runtime_ms\n";

    // Helper lambda to load a 3x3 homography matrix from plain-text file
    auto loadHomography = [](const std::string& path) {
        cv::Mat homography = cv::Mat::eye(3,3,CV_64F);
        std::ifstream input(path);
        if (input) for (int i = 0; i < 9; ++i) input >> homography.at<double>(i/3, i%3);
        // If loading failed, return empty Mat so caller can skip
        return input ? homography : cv::Mat();
    };

    // Helper lambda to convert Method enum to a human-readable name
    auto methodName = [](Method method) {
        switch(method) { case Method::NN_RT: return "NN+RT"; case Method::RANSAC: return "RANSAC"; case Method::GALAM: return "GaLAM"; }
        return "";
    };

    // Load image with .ppm or .pgm extension


    auto loadImage = [](const std::string& base) {
        cv::Mat img = cv::imread(base + ".ppm", cv::IMREAD_GRAYSCALE);
        if (img.empty()) img = cv::imread(base + ".pgm", cv::IMREAD_GRAYSCALE);
        return img;
    };

    // Accumulators for summary
    // Accumulators for viewpoint-change scenes (graf, wall) and
    // illumination-change scene (leuven).
    // We store all %H.E and AP values per method and average them at the end.
    std::map<std::string, std::vector<double>> all_corr, all_err, all_inlier, all_time;
    // Accumulators for Table 1 (viewpoint and light)
    std::map<std::string, std::vector<double>> vp_he, vp_ap, lt_he, lt_ap, bl_he, bl_ap, zr_he, zr_ap, jp_he, jp_ap;

    // Nice formatted console printing
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Scene\tPair\tMethod\tCorr\tAvgErr\tInlier%\tTime(ms)\n";
    std::cout << std::string(60,'-') << "\n";

    // Loop over all scenes
    for (const auto& scene : scenes) {
        std::string scenePath = dataPath + "/" + scene;
        if (!fs::exists(scenePath)) continue;// Skip if folder doesn't exist

        // Load reference image img1 for this scene
        cv::Mat img1 = loadImage(scenePath + "/img1");
        if (img1.empty()) continue;

        bool isViewpoint = (scene == "graf" || scene == "wall");
        bool isLight = (scene == "leuven");
        bool isBlur = (scene == "bikes" || scene == "trees");
        bool isZoomRot = (scene == "bark" || scene == "boat");

        // HPatches convention: compare img1 with img2..img6
        for (int i = 2; i <= 6; ++i) {
            cv::Mat img2 = loadImage(scenePath + "/img" + std::to_string(i));
            cv::Mat gtHomography = loadHomography(scenePath + "/H1to" + std::to_string(i) + "p");
            if (img2.empty() || gtHomography.empty()) continue;// skip if missing

            // For each feature detector    
            for (auto det : detectors_) {
                std::vector<cv::KeyPoint> kp1, kp2;
                cv::Mat d1, d2;
                // Extract features + descriptors in both images
                getFeatures(img1, det, kp1, d1);
                getFeatures(img2, det, kp2, d2);


                // Choose distance metric depending on descriptor type:
                //  - ORB => binary => NORM_HAMMING
                //  - SIFT / AKAZE (default float) => NORM_L
                cv::BFMatcher matcher(det == Detector::ORB ? cv::NORM_HAMMING : cv::NORM_L2);
                
                // KNN matching: find 2 nearest neighbors in img2 for each descriptor in img1
                std::vector<std::vector<cv::DMatch>> knn;
                matcher.knnMatch(d1, d2, knn, 2);

                // Apply Lowe's ratio test (0.8) to get initial matches
                std::vector<cv::DMatch> matches;
                std::vector<cv::DMatch> nnMatches;  
                for (const auto& knnMatch : knn) {
                    matches.push_back(knnMatch[0]);
                    if (knnMatch.size() >= 2 && knnMatch[0].distance < 0.8 * knnMatch[1].distance) {
                        nnMatches.push_back(knnMatch[0]);
                    }
                }

                // For each outlier-removal method
                for (auto method : methods_) {
                    double rt;
                    // Apply method to filter outliers
                    auto filtered = filterOutliers(method, kp1, kp2, d1, d2, matches, nnMatches, img1.size(), img2.size(), rt);
                    // Evaluate filtered matches against ground-truth homography
                    Metrics met = evaluateMatches(kp1, kp2, filtered, gtHomography, rt);

                    // Save visualizations, once
                    if (i == 2) {
                        std::string filehead = "./output/" + scene + "_img" + std::to_string(i) + "_";

                        switch (det) {
                        case Detector::AKAZE:
                            filehead += "AKAZE_";
                            break;

                        case Detector::ORB:
                            filehead += "ORB_";
                            break;

                        case Detector::SIFT:
                            filehead += "SIFT_";
                        }

                        cv::Mat output;
                        cv::drawMatches(img1, kp1, img2, kp2, matches, output);
                        cv::imwrite(filehead + "initial.jpg", output);
                        std::cout << "  Saved: " << filehead << "initial.jpg" << std::endl;

                        cv::drawMatches(img1, kp1, img2, kp2, filtered, output);
                        switch (method) {
                        case Method::NN_RT:
                            cv::imwrite(filehead + "NNRT_filtered.jpg", output);
                            std::cout << "  Saved: " << filehead << "NNRT_filtered.jpg" << std::endl;
                            break;
        
                        case Method::RANSAC:
                            cv::imwrite(filehead + "RANSAC_filtered.jpg", output);
                            std::cout << "  Saved: " << filehead << "RANSAC_filtered.jpg" << std::endl;
                            break;

                        case Method::GALAM:
                            cv::imwrite(filehead + "GALAM_filtered.jpg", output);
                            std::cout << "  Saved: " << filehead << "GALAM_filtered.jpg" << std::endl;
                        }
                        
                    }

                    std::string mname = methodName(method);
                    // Write a row in CSV: scene, image pair, method, metrics
                    csv << scene << "," << "1-" << i << "," << mname << ","
                        << met.correspondences << "," << met.avg_error << "," << met.inlier_pct << "," 
                        << met.he_pct << "," << met.runtime_ms << "\n";
                    // Print to console for quick inspection
                    std::cout << scene << "\t1-" << i << "\t" << mname << "\t"
                              << met.correspondences << "\t" << met.avg_error << "\t" 
                              << met.inlier_pct << "\t" << met.runtime_ms << "\n";

                    // Accumulate for summary
                    all_corr[mname].push_back(met.correspondences);
                    all_err[mname].push_back(met.avg_error);
                    all_inlier[mname].push_back(met.inlier_pct);
                    all_time[mname].push_back(met.runtime_ms);

                    // Accumulate for Table 1
                    // Collect results for summary averages:
                    //  - viewpoint scenes: graf, wall
                    //  - lighting scene: leuven
                    if (isViewpoint) {
                        vp_he[mname].push_back(met.he_pct); vp_ap[mname].push_back(met.inlier_pct);
                    }
                    else if (isLight) {
                        lt_he[mname].push_back(met.he_pct); lt_ap[mname].push_back(met.inlier_pct);
                    }
                    else if (isBlur) {
                        bl_he[mname].push_back(met.he_pct); bl_ap[mname].push_back(met.inlier_pct);
                    }
                    else if (isZoomRot) {
                        zr_he[mname].push_back(met.he_pct); zr_ap[mname].push_back(met.inlier_pct);
                    }
                    else {
                        jp_he[mname].push_back(met.he_pct); jp_ap[mname].push_back(met.inlier_pct);
                    }
                }
            }
        }
    }
    // Helper to compute the average of a vector, or 0.0 if empty
    auto avg = [](const std::vector<double>& vec) { 
        return vec.empty() ? 0.0 : std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size(); 
    };

    // Homography Estimation
    std::cout << "\n===== HOMOGRAPHY ESTIMATION =====\n";
    std::cout << std::left << std::setw(10) << "Method" 
              << std::right << std::setw(12) << "View %H.E" << std::setw(10) << "View AP"
              << std::setw(12) << "Light %H.E" << std::setw(10) << "Light AP"
              << std::setw(12) << "Blur %H.E" << std::setw(10) << "Blur AP" 
              << std::setw(12) << "Zoom+Rot %H.E" << std::setw(10) << "Zoom+Rot AP" 
              << std::setw(12) << "Comp %H.E" << std::setw(10) << "Comp AP" << "\n";
    std::cout << std::string(54, '-') << "\n";

    for (const auto& mname : {"NN+RT", "RANSAC", "GaLAM"}) {
        std::cout << std::left << std::setw(10) << mname
                  << std::right << std::setw(12) << avg(vp_he[mname]) << std::setw(10) << avg(vp_ap[mname])
                  << std::setw(12) << avg(lt_he[mname]) << std::setw(10) << avg(lt_ap[mname]) 
                  << std::setw(12) << avg(bl_he[mname]) << std::setw(10) << avg(bl_ap[mname]) 
                  << std::setw(12) << avg(zr_he[mname]) << std::setw(10) << avg(zr_ap[mname]) 
                  << std::setw(12) << avg(jp_he[mname]) << std::setw(10) << avg(jp_ap[mname]) << "\n";
    }

    // Summary
    std::cout << "\n===== SUMMARY =====\n";
    std::cout << std::left << std::setw(10) << "Method"
              << std::right << std::setw(12) << "Avg Corr"
              << std::setw(12) << "Avg Error"
              << std::setw(12) << "Inlier %"
              << std::setw(12) << "Runtime(ms)" << "\n";
    std::cout << std::string(58, '-') << "\n";

    for (const auto& mname : {"NN+RT", "RANSAC", "GaLAM"}) {
        std::cout << std::left << std::setw(10) << mname
                  << std::right << std::setw(12) << std::fixed << std::setprecision(1) << avg(all_corr[mname])
                  << std::setw(12) << std::setprecision(2) << avg(all_err[mname])
                  << std::setw(12) << avg(all_inlier[mname])
                  << std::setw(12) << avg(all_time[mname]) << "\n";
    }

    std::cout << "\nSaved: " << csvPath << "\n";
}