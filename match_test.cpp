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
    cv::Ptr<cv::Feature2D> f;
    switch (det) {
        case Detector::SIFT:  f = cv::SIFT::create(); break;
        case Detector::ORB:   f = cv::ORB::create(5000); break;
        case Detector::AKAZE: f = cv::AKAZE::create(); break;
    }
    f->detectAndCompute(img, cv::noArray(), kp, desc);
}

std::vector<cv::DMatch> MatchTest::filterOutliers(Method method,
    const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
    const cv::Mat& d1, const cv::Mat& d2,
    const std::vector<cv::DMatch>& matches,
    const cv::Size& sz1, const cv::Size& sz2, double& runtime_ms) {
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<cv::DMatch> result;

    switch (method) {
        case Method::NN_RT:
            result = matches;
            break;

        case Method::RANSAC:
            if (matches.size() >= 4) {
                std::vector<cv::Point2f> p1, p2;
                for (const auto& m : matches) {
                    p1.push_back(kp1[m.queryIdx].pt);
                    p2.push_back(kp2[m.trainIdx].pt);
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
    const std::vector<cv::DMatch>& matches, const cv::Mat& gtH, double runtime_ms) {
    
    Metrics m;
    m.correspondences = (int)matches.size();
    m.avg_error = 0.0;
    m.inlier_pct = 0.0;
    m.he_pct = 0.0;
    m.runtime_ms = runtime_ms;
    
    if (matches.empty() || gtH.empty()) return m;

    cv::Mat H;
    gtH.convertTo(H, CV_64F);
    int he_count = 0, inlier_count = 0;
    double total_error = 0.0;

    for (const auto& match : matches) {
        cv::Point2f p1 = kp1[match.queryIdx].pt, p2 = kp2[match.trainIdx].pt;
        double w = H.at<double>(2,0)*p1.x + H.at<double>(2,1)*p1.y + H.at<double>(2,2);
        double px = (H.at<double>(0,0)*p1.x + H.at<double>(0,1)*p1.y + H.at<double>(0,2)) / w;
        double py = (H.at<double>(1,0)*p1.x + H.at<double>(1,1)*p1.y + H.at<double>(1,2)) / w;
        double e = std::sqrt((px-p2.x)*(px-p2.x) + (py-p2.y)*(py-p2.y));
        
        total_error += e;
        if (e < 1.0) ++he_count;
        if (e < 3.0) ++inlier_count;
    }

    m.avg_error = total_error / matches.size();
    m.he_pct = 100.0 * he_count / matches.size();
    m.inlier_pct = 100.0 * inlier_count / matches.size();
    return m;
}

void MatchTest::runTests(const std::string& dataPath, const std::string& csvPath) {
    std::vector<std::string> scenes = {"bark","bikes","boat","graf","leuven","trees","ubc","wall"};
    std::ofstream csv(csvPath);
    csv << "Scene,Pair,Method,Correspondences,AvgError,Inlier%,H.E%,Runtime_ms\n";

    auto loadH = [](const std::string& p) {
        cv::Mat H = cv::Mat::eye(3,3,CV_64F);
        std::ifstream f(p);
        if (f) for (int i = 0; i < 9; ++i) f >> H.at<double>(i/3, i%3);
        return f ? H : cv::Mat();
    };

    auto methodName = [](Method m) {
        switch(m) { case Method::NN_RT: return "NN+RT"; case Method::RANSAC: return "RANSAC"; case Method::GALAM: return "GaLAM"; }
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
    std::map<std::string, std::vector<double>> vp_he, vp_ap, lt_he, lt_ap;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Scene\tPair\tMethod\tCorr\tAvgErr\tInlier%\tTime(ms)\n";
    std::cout << std::string(60,'-') << "\n";

    for (const auto& scene : scenes) {
        std::string sp = dataPath + "/" + scene;
        if (!fs::exists(sp)) continue;

        cv::Mat img1 = loadImage(sp + "/img1");
        if (img1.empty()) continue;

        bool isViewpoint = (scene == "graf" || scene == "wall");
        bool isLight = (scene == "leuven");

        for (int i = 2; i <= 6; ++i) {
            cv::Mat img2 = loadImage(sp + "/img" + std::to_string(i));
            cv::Mat gtH = loadH(sp + "/H1to" + std::to_string(i) + "p");
            if (img2.empty() || gtH.empty()) continue;

            for (auto det : detectors_) {
                std::vector<cv::KeyPoint> kp1, kp2;
                cv::Mat d1, d2;
                getFeatures(img1, det, kp1, d1);
                getFeatures(img2, det, kp2, d2);

                cv::BFMatcher matcher(det == Detector::ORB ? cv::NORM_HAMMING : cv::NORM_L2);
                std::vector<std::vector<cv::DMatch>> knn;
                matcher.knnMatch(d1, d2, knn, 2);
                std::vector<cv::DMatch> matches;
                for (const auto& k : knn)
                    if (k.size() >= 2 && k[0].distance < 0.8 * k[1].distance)
                        matches.push_back(k[0]);

                for (auto method : methods_) {
                    double rt;
                    auto filtered = filterOutliers(method, kp1, kp2, d1, d2, matches, img1.size(), img2.size(), rt);
                    Metrics m = evaluateMatches(kp1, kp2, filtered, gtH, rt);

                    std::string mname = methodName(method);
                    csv << scene << "," << "1-" << i << "," << mname << ","
                        << m.correspondences << "," << m.avg_error << "," << m.inlier_pct << "," 
                        << m.he_pct << "," << m.runtime_ms << "\n";
                    std::cout << scene << "\t1-" << i << "\t" << mname << "\t"
                              << m.correspondences << "\t" << m.avg_error << "\t" 
                              << m.inlier_pct << "\t" << m.runtime_ms << "\n";

                    // Accumulate for summary
                    all_corr[mname].push_back(m.correspondences);
                    all_err[mname].push_back(m.avg_error);
                    all_inlier[mname].push_back(m.inlier_pct);
                    all_time[mname].push_back(m.runtime_ms);

                    // Accumulate for Table 1
                    if (isViewpoint) { vp_he[mname].push_back(m.he_pct); vp_ap[mname].push_back(m.inlier_pct); }
                    if (isLight) { lt_he[mname].push_back(m.he_pct); lt_ap[mname].push_back(m.inlier_pct); }
                }
            }
        }
    }

    auto avg = [](const std::vector<double>& v) { 
        return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); 
    };

    // Homography Estimation
    std::cout << "\n===== HOMOGRAPHY ESTIMATION =====\n";
    std::cout << std::left << std::setw(10) << "Method" 
              << std::right << std::setw(12) << "VP %H.E" << std::setw(10) << "VP AP"
              << std::setw(12) << "Light %H.E" << std::setw(10) << "Light AP" << "\n";
    std::cout << std::string(54, '-') << "\n";

    for (const auto& mname : {"NN+RT", "RANSAC", "GaLAM"}) {
        std::cout << std::left << std::setw(10) << mname
                  << std::right << std::setw(12) << avg(vp_he[mname]) << std::setw(10) << avg(vp_ap[mname])
                  << std::setw(12) << avg(lt_he[mname]) << std::setw(10) << avg(lt_ap[mname]) << "\n";
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