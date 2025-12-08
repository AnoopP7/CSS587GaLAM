/*
 * main.cpp
 * CSS587 Design Project - GaLAM Testing
 * Implementation authors: Yu Dinh, Neha Kotwal, Anoop Prasad
 * Paper title: GaLAM: Two-Stage Outlier Detection Algorithm
 * Paper authors: X. Lu, Z. Yan, Z. Fan
 *
 * This file contains the main function to run GaLAM tests over a dataset.
 * Purpose:
 * - Provide a command-line interface for testing the GaLAM algorithm
 * - Facilitate the evaluation of different feature detectors and matching methods
 * - Generate and save test results for analysis
 */

#include "galam.h"
#include "match_test.h"


// demo
// Simply finds matches between two images, for testing and demo purposes
// Preconditions: Valid paths to two images are provided
// Postconditions: Logs of GaLAM progress are output to console and visualizations are saved to computer
static int demo(const std::string &imagePath1, const std::string &imagePath2)
{
    std::cout << "========================================" << std::endl;
    std::cout << "GaLAM Implementation" << std::endl;
    std::cout << "========================================\n"
              << std::endl;

    // Check that two paths were given
    if (imagePath1.empty() || imagePath2.empty())
    {
        std::cerr << "Need to provide two image paths. Received:" << std::endl;
        std::cerr << "Image 1: " << imagePath1 << std::endl;
        std::cerr << "Image 2: " << imagePath2 << std::endl;
    }

    // Load images
    cv::Mat img1 = cv::imread(imagePath1);
    cv::Mat img2 = cv::imread(imagePath2);

    if (img1.empty() || img2.empty())
    {
        std::cerr << "Error: Could not load images!" << std::endl;
        return 1;
    }

    std::cout << "Images loaded:" << std::endl;
    std::cout << "  Image 1: " << img1.cols << "x" << img1.rows << std::endl;
    std::cout << "  Image 2: " << img2.cols << "x" << img2.rows << std::endl;

    // Scale down images for faster computation
    double scalingFactor = 0.5;
    cv::resize(img1, img1, cv::Size(), scalingFactor, scalingFactor);
    cv::resize(img2, img2, cv::Size(), scalingFactor, scalingFactor);

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

    // Detect ORB features
    // std::cout << "\nDetecting ORB features..." << std::endl;
    // // ORB parameters: you can tune nfeatures if needed
    // cv::Ptr<cv::ORB> orb = cv::ORB::create(
    //     5000, // number of features
    //     1.2f, // scale factor
    //     8     // pyramid levels
    // );

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    sift->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);
    // orb->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    // orb->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);

    std::cout << "  Keypoints detected: " << keypoints1.size()
              << " and " << keypoints2.size() << std::endl;

    // Initial matching
    std::cout << "\nPerforming initial matching..." << std::endl;
    cv::BFMatcher matcher(cv::NORM_L2);
    // cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> initialMatches;
    matcher.match(descriptors1, descriptors2, initialMatches);
    std::cout << "  Initial matches: " << initialMatches.size() << std::endl;

    // Apply GaLAM filtering
    std::cout << "\nApplying GaLAM filtering..." << std::endl;
    GaLAM::InputParameters params(100, 0.8, 1.0, 3.0, 3.0, 0.8, 10.0, 0.5, 0.9, 128, 8);
    //GaLAM::InputParameters params(100, 0.8, 1.0, 3.0, 3.0, 1.1, 10.0, 0.5, 0.9, 128, 8);
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

    // Generate visualizations
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

// main
// Runs GaLAM filtering, either on the Oxford dataset or two provided images
// Preconditions: Either two arguments of the data path and output file path are provided to use the Oxford dataset,
//          or three arguments of "match" followed by two image file paths for demo purposes
// Postconditions: Either Oxford results or image matching results are output and saved based on command line arguments
int main(int argc, char **argv)
{
    // Not enough arguments
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <data_path> [output.csv]\nor\n"
                  << argv[0] << "match <imgpath1> <imgpath2>\n";
        return 1;
    }

    // Simple image matching demo
    if (argc == 4)
    {
        std::cout << "Matching: " << argv[2] << " with " << argv[3] << std::endl;
        std::string path1 = std::string(argv[2]);
        std::string path2 = std::string(argv[3]);
        return demo(path1, path2);
    }

    // Create tester and run tests on Oxford dataset
    MatchTest tester(
        {MatchTest::Detector::SIFT},
        {MatchTest::Method::NN_RT, MatchTest::Method::RANSAC, MatchTest::Method::GMS, MatchTest::Method::GALAM});

    tester.runTests(argv[1], argc >= 3 ? argv[2] : "./output/results.csv");
    return 0;
}