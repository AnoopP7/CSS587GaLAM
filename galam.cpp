#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>

namespace galam {

class GaLAM {
public:

    struct Parameters {
    };

    GaLAM(const Parameters& params = Parameters()) : params_(params) {}

    // Main detection pipeline (stub for now)
    std::vector<cv::DMatch> detectOutliers(
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2,
        const std::vector<cv::DMatch>& candidateMatches,
        const cv::Size& imageSize1,
        const cv::Size& imageSize2
    ) {
        std::cout << "GaLAM: Processing " << candidateMatches.size() << " candidate matches..." << std::endl;

        // TODO: Stage 1 - Local affine verification
        // TODO: Stage 2 - Global geometric consistency

        // For now, just return
        std::vector<cv::DMatch> filtered;
        for (size_t i = 0; i < candidateMatches.size(); ++i) {
            filtered.push_back(candidateMatches[i]);
        }

        std::cout << "GaLAM: Filtered to " << filtered.size() << " matches" << std::endl;
        return filtered;
    }

private:
    Parameters params_;
};

}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "GaLAM Implementation" << std::endl;
    std::cout << "========================================\n" << std::endl;

    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <img1> <img2>" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0] << " img1.jpg img2.jpg" << std::endl;
        return 1;
    }

    // Load images
    cv::Mat img1 = cv::imread(argv[1]);
    cv::Mat img2 = cv::imread(argv[2]);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load images!" << std::endl;
        return 1;
    }

    std::cout << "Images loaded:" << std::endl;
    std::cout << "  Image 1: " << img1.cols << "x" << img1.rows << std::endl;
    std::cout << "  Image 2: " << img2.cols << "x" << img2.rows << std::endl;

    // Convert to grayscale
    cv::Mat gray1, gray2;
    if (img1.channels() == 3) cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    else gray1 = img1.clone();
    if (img2.channels() == 3) cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    else gray2 = img2.clone();

    // Detect SIFT features
    std::cout << "\nDetecting SIFT features..." << std::endl;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    sift->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);

    std::cout << "  Keypoints detected: " << keypoints1.size() << " and " << keypoints2.size() << std::endl;

    // Initial matching
    std::cout << "\nPerforming initial matching..." << std::endl;
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> initialMatches;
    matcher.match(descriptors1, descriptors2, initialMatches);

    std::cout << "  Initial matches: " << initialMatches.size() << std::endl;

    // Apply GaLAM (skeleton version)
    std::cout << "\nApplying GaLAM filtering..." << std::endl;
    galam::GaLAM galam;
    std::vector<cv::DMatch> filteredMatches = galam.detectOutliers(
        keypoints1, keypoints2,
        descriptors1, descriptors2,
        initialMatches,
        cv::Size(img1.cols, img1.rows),
        cv::Size(img2.cols, img2.rows)
    );

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Initial matches:  " << initialMatches.size() << std::endl;
    std::cout << "  Filtered matches: " << filteredMatches.size() << std::endl;
    std::cout << "  Reduction: " << (100.0 * (initialMatches.size() - filteredMatches.size()) / initialMatches.size()) << "%" << std::endl;
    std::cout << "========================================" << std::endl;

    // visualization
    cv::Mat imgMatches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, filteredMatches, imgMatches);
    cv::imwrite("matches_output.jpg", imgMatches);
    std::cout << "\nVisualization saved to: matches_output.jpg" << std::endl;

    // TODO: Run AdaLam

    // Compare results

    return 0;
}