/*
 * match_test.cpp
 */

#include "match_test.h"
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>

namespace fs = std::filesystem;

MatchTest::MatchTest(const std::vector<Detector>& detectors, const std::vector<Method>& methods)
    : detectors_(detectors), methods_(methods) {}

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

std::vector<cv::DMatch> MatchTest::filterOutliers(Method method,
    const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
    const cv::Mat& d1, const cv::Mat& d2,
    const std::vector<cv::DMatch>& matches,
    const std::vector<cv::DMatch>& nnMatches,
    const cv::Size& sz1, const cv::Size& sz2, double& runtime_ms) {
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<cv::DMatch> result;

    switch (method) {
        case Method::NN_RT:
            result = nnMatches;
            break;

        case Method::RANSAC:
            if (matches.size() >= 4) {
                std::vector<cv::Point2f> p1, p2;
                for (const auto& match : matches) {
                    p1.push_back(kp1[match.queryIdx].pt);
                    p2.push_back(kp2[match.trainIdx].pt);
                }
                std::vector<char> mask;
                cv::findHomography(p1, p2, cv::RANSAC, 3.0, mask);
                for (size_t i = 0; i < matches.size(); ++i)
                    if (mask[i]) result.push_back(matches[i]);
            }
            break;

        case Method::GALAM: {
            GaLAM::InputParameters params;
            GaLAM galam(params);
            std::vector<cv::KeyPoint> k1 = kp1, k2 = kp2;
            result = galam.detectOutliers(k1, k2, d1, d2, matches, sz1, sz2).finalMatches;
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    runtime_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

MatchTest::Metrics MatchTest::evaluateMatches(
    const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches, const cv::Mat& gtHomography, double runtime_ms) {
    
    Metrics met;
    met.correspondences = (int)matches.size();
    met.avg_error = 0.0;
    met.inlier_pct = 0.0;
    met.he_pct = 0.0;
    met.runtime_ms = runtime_ms;
    
    if (matches.empty() || gtHomography.empty()) return met;

    cv::Mat homography;
    gtHomography.convertTo(homography, CV_64F);
    int he_count = 0, inlier_count = 0;
    double total_error = 0.0;

    for (const auto& match : matches) {
        cv::Point2f p1 = kp1[match.queryIdx].pt, p2 = kp2[match.trainIdx].pt;
        double scalingW = homography.at<double>(2,0)*p1.x + homography.at<double>(2,1)*p1.y + homography.at<double>(2,2);
        double projX = (homography.at<double>(0,0)*p1.x + homography.at<double>(0,1)*p1.y + homography.at<double>(0,2)) / scalingW;
        double projY = (homography.at<double>(1,0)*p1.x + homography.at<double>(1,1)*p1.y + homography.at<double>(1,2)) / scalingW;
        double error = std::sqrt((projX-p2.x)*(projX-p2.x) + (projY-p2.y)*(projY-p2.y));
        
        total_error += error;
        if (error < 1.0) ++he_count;
        if (error < 3.0) ++inlier_count;
    }

    met.avg_error = total_error / matches.size();
    met.he_pct = 100.0 * he_count / matches.size();
    met.inlier_pct = 100.0 * inlier_count / matches.size();
    return met;
}

void MatchTest::runTests(const std::string& dataPath, const std::string& csvPath) {
    std::vector<std::string> scenes = {"bark","bikes","boat","graf","leuven","trees","ubc","wall"};
    std::ofstream csv(csvPath);
    csv << "Scene,Pair,Method,Correspondences,AvgError,Inlier%,H.E%,Runtime_ms\n";

    auto loadHomography = [](const std::string& path) {
        cv::Mat homography = cv::Mat::eye(3,3,CV_64F);
        std::ifstream input(path);
        if (input) for (int i = 0; i < 9; ++i) input >> homography.at<double>(i/3, i%3);
        return input ? homography : cv::Mat();
    };

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
    std::map<std::string, std::vector<double>> all_corr, all_err, all_inlier, all_time;
    // Accumulators for Table 1 (viewpoint and light)
    std::map<std::string, std::vector<double>> vp_he, vp_ap, lt_he, lt_ap, bl_he, bl_ap, zr_he, zr_ap, jp_he, jp_ap;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Scene\tPair\tMethod\tCorr\tAvgErr\tInlier%\tTime(ms)\n";
    std::cout << std::string(60,'-') << "\n";

    for (const auto& scene : scenes) {
        std::string scenePath = dataPath + "/" + scene;
        if (!fs::exists(scenePath)) continue;

        cv::Mat img1 = loadImage(scenePath + "/img1");
        if (img1.empty()) continue;

        bool isViewpoint = (scene == "graf" || scene == "wall");
        bool isLight = (scene == "leuven");
        bool isBlur = (scene == "bikes" || scene == "trees");
        bool isZoomRot = (scene == "bark" || scene == "boat");

        for (int i = 2; i <= 6; ++i) {
            cv::Mat img2 = loadImage(scenePath + "/img" + std::to_string(i));
            cv::Mat gtHomography = loadHomography(scenePath + "/H1to" + std::to_string(i) + "p");
            if (img2.empty() || gtHomography.empty()) continue;

            for (auto det : detectors_) {
                std::vector<cv::KeyPoint> kp1, kp2;
                cv::Mat d1, d2;
                getFeatures(img1, det, kp1, d1);
                getFeatures(img2, det, kp2, d2);

                cv::BFMatcher matcher(det == Detector::ORB ? cv::NORM_HAMMING : cv::NORM_L2);
                std::vector<std::vector<cv::DMatch>> knn;
                matcher.knnMatch(d1, d2, knn, 2);
                std::vector<cv::DMatch> matches;
                std::vector<cv::DMatch> nnMatches;
                for (const auto& knnMatch : knn) {
                    matches.push_back(knnMatch[0]);
                    if (knnMatch.size() >= 2 && knnMatch[0].distance < 0.8 * knnMatch[1].distance) {
                        nnMatches.push_back(knnMatch[0]);
                    }
                }

                for (auto method : methods_) {
                    double rt;
                    auto filtered = filterOutliers(method, kp1, kp2, d1, d2, matches, nnMatches, img1.size(), img2.size(), rt);
                    Metrics met = evaluateMatches(kp1, kp2, filtered, gtHomography, rt);

                    // Save visualizations, once
                    if (i == 2) {
                        std::string filehead = "./output/" + scene + "_img" + std::to_string(i) + "_";

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
                    csv << scene << "," << "1-" << i << "," << mname << ","
                        << met.correspondences << "," << met.avg_error << "," << met.inlier_pct << "," 
                        << met.he_pct << "," << met.runtime_ms << "\n";
                    std::cout << scene << "\t1-" << i << "\t" << mname << "\t"
                              << met.correspondences << "\t" << met.avg_error << "\t" 
                              << met.inlier_pct << "\t" << met.runtime_ms << "\n";

                    // Accumulate for summary
                    all_corr[mname].push_back(met.correspondences);
                    all_err[mname].push_back(met.avg_error);
                    all_inlier[mname].push_back(met.inlier_pct);
                    all_time[mname].push_back(met.runtime_ms);

                    // Accumulate for Table 1
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