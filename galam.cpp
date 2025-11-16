#include "galam.h"

#include <iostream>

namespace galam {

GaLAM::GaLAM(const Parameters& params)
    : params_(params) {}

// Main detection pipeline (stub for now)
std::vector<cv::DMatch> GaLAM::detectOutliers(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    const std::vector<cv::DMatch>& candidateMatches,
    const cv::Size& imageSize1,
    const cv::Size& imageSize2
) const
{
    std::cout << "GaLAM: Processing " << candidateMatches.size()
              << " candidate matches..." << std::endl;

    // TODO: Stage 1 - Local affine verification
    // TODO: Stage 2 - Global geometric consistency

    // For now, just return all matches (no filtering yet)
    std::vector<cv::DMatch> filtered;
    filtered.reserve(candidateMatches.size());
    for (const auto& m : candidateMatches) {
        filtered.push_back(m);
    }

    std::cout << "GaLAM: Filtered to " << filtered.size()
              << " matches" << std::endl;

    return filtered;
}

} // namespace galam
