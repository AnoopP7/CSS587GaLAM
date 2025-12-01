#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

#include "galam.h"

int main(int argc, char **argv)
{
    std::cout << "========================================" << std::endl;
    std::cout << "GaLAM Implementation" << std::endl;
    std::cout << "========================================\n"
              << std::endl;

    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " <img1> <img2>" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0] << " img1.jpg img2.jpg" << std::endl;
        return 1;
    }

    // Load images
    cv::Mat img1 = cv::imread(argv[1]);
    cv::Mat img2 = cv::imread(argv[2]);

    if (img1.empty() || img2.empty())
    {
        std::cerr << "Error: Could not load images!" << std::endl;
        return 1;
    }

    std::cout << "Images loaded:" << std::endl;
    std::cout << "  Image 1: " << img1.cols << "x" << img1.rows << std::endl;
    std::cout << "  Image 2: " << img2.cols << "x" << img2.rows << std::endl;

    // double scalingFactor = 0.25;
    // cv::resize(img1, img1, cv::Size(), scalingFactor, scalingFactor);
    // cv::resize(img2, img2, cv::Size(), scalingFactor, scalingFactor);

    // Convert to grayscale
    cv::Mat gray1, gray2;
    if (img1.channels() == 3)
    {
        cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray1 = img1.clone();
    }

    if (img2.channels() == 3)
    {
        cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray2 = img2.clone();
    }

    // Detect SIFT features
    std::cout << "\nDetecting SIFT features..." << std::endl;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    sift->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);

    std::cout << "  Keypoints detected: " << keypoints1.size()
              << " and " << keypoints2.size() << std::endl;

    // Initial matching
    std::cout << "\nPerforming initial matching..." << std::endl;
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> initialMatches;
    matcher.match(descriptors1, descriptors2, initialMatches);
    std::cout << "  Initial matches: " << initialMatches.size() << std::endl;

    // Apply GaLAM
    std::cout << "\nApplying GaLAM filtering..." << std::endl;
    GaLAM::InputParameters params;
    GaLAM galam(params);

    GaLAM::StageResults results = galam.detectOutliers(
        keypoints1, keypoints2,
        descriptors1, descriptors2,
        initialMatches,
        cv::Size(img1.cols, img1.rows),
        cv::Size(img2.cols, img2.rows));

    const auto &seedMatches = results.seedMatches;
    const auto &stage1Matches = results.stage1Matches;
    const auto &finalMatches = results.finalMatches;

    // Results summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Initial matches:   " << initialMatches.size() << std::endl;
    std::cout << "  Seed points:       " << seedMatches.size() << std::endl;
    std::cout << "  Stage 1 inliers:   " << stage1Matches.size() << std::endl;
    std::cout << "  Stage 2 (final):   " << finalMatches.size() << std::endl;

    double reductionPercent = 0.0;
    if (!initialMatches.empty())
    {
        reductionPercent =
            100.0 * (static_cast<double>(initialMatches.size() - finalMatches.size()) / static_cast<double>(initialMatches.size()));
    }
    std::cout << "  Outliers removed:  " << (initialMatches.size() - finalMatches.size()) << std::endl;
    std::cout << "  Reduction:         " << std::fixed << std::setprecision(1)
              << reductionPercent << "%" << std::endl;
    std::cout << "========================================" << std::endl;

    // Visualization
    std::cout << "\nGenerating visualizations..." << std::endl;
    cv::Mat output;

    // 1. Initial matches
    cv::drawMatches(img1, keypoints1, img2, keypoints2, initialMatches, output,
                    cv::Scalar(0, 255, 255));
    cv::imwrite("galam_1_initial.jpg", output);
    std::cout << "  Saved: galam_1_initial.jpg" << std::endl;

    // 2. Seed points
    cv::drawMatches(img1, keypoints1, img2, keypoints2, seedMatches, output,
                    cv::Scalar(255, 100, 0));
    cv::imwrite("galam_2_seeds.jpg", output);
    std::cout << "  Saved: galam_2_seeds.jpg" << std::endl;

    // 3. Stage 1 inliers
    cv::drawMatches(img1, keypoints1, img2, keypoints2, stage1Matches, output,
                    cv::Scalar(255, 255, 0));
    cv::imwrite("galam_3_stage1.jpg", output);
    std::cout << "  Saved: galam_3_stage1.jpg" << std::endl;

    // 4. Stage 2 final matches
    cv::drawMatches(img1, keypoints1, img2, keypoints2, finalMatches, output,
                    cv::Scalar(0, 255, 0));
    cv::imwrite("galam_4_final.jpg", output);
    std::cout << "  Saved: galam_4_final.jpg" << std::endl;

    return 0;
}
