#include "galam.h"

#include <opencv2/features2d.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cmath>

//namespace galam {

GaLAM::GaLAM(const InputParameters& params)
    : params_(params) {}

std::vector<GaLAM::ScoredMatch> GaLAM::selectSeedPoints(
    const std::vector<ScoredMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints1,
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    const cv::Size& imageSize1
) const
{
    return std::vector<ScoredMatch>();
}

std::vector<GaLAM::ScoredMatch> GaLAM::filterBidirectionalNN(
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2
) const
{
    std::vector<std::vector<cv::DMatch>> knn12, knn21;

    // If slow, FLANN instead
    cv::BFMatcher matcher(cv::NORM_L2);
    matcher.knnMatch(descriptors1, descriptors2, knn12, 2);
    matcher.knnMatch(descriptors2, descriptors1, knn21, 2);

    std::vector<ScoredMatch> validMatches;

    for (int i = 0; i < static_cast<int>(knn12.size()); ++i) {
        if (knn12[i].size() < 2) continue;

        const cv::DMatch& bestMatch = knn12[i][0];
        const cv::DMatch& secondBestMatch = knn12[i][1];

        if (bestMatch.distance >= params_.rt_threshold * secondBestMatch.distance)
            continue;

        int queryIdx = bestMatch.queryIdx;
        int trainIdx = bestMatch.trainIdx;

        if (trainIdx < 0 || trainIdx >= static_cast<int>(knn21.size())) continue;
        if (knn21[trainIdx].empty()) continue;

        const cv::DMatch& reverseMatch = knn21[trainIdx][0];
        if (reverseMatch.trainIdx != queryIdx) continue;

        ScoredMatch scored;
        scored.match = bestMatch;
        scored.confidence = 0.0;
        validMatches.push_back(scored);
    }

    std::cout << "GaLAM: Bidirectional NN matches = " << validMatches.size() << std::endl;
    return validMatches;
}

// 2) Assign confidence score (reciprocal of distance)
void GaLAM::assignConfidenceScore(
    std::vector<ScoredMatch>& matches
) const
{
    for (auto& scored : matches) {
        double distance = std::max(static_cast<double>(scored.match.distance), 1e-6);
        scored.confidence = 1.0 / distance;
    }
}

// 3) Select seed points using non-maximum suppression
// This R is global for the whole image pair.
// It is used ONLY for NMS—NOT the local neighborhood radius (R1, R2).
std::vector<GaLAM::ScoredMatch> GaLAM::selectPoints(
    const std::vector<ScoredMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints1,
    const cv::Size& imageSize1
) const
{
    if (matches.empty()) return {};

    // ----------------------------------------------------------------------
    // Compute GaLAM global NMS radius R
    //
    // Paper (Implementation details):
    //     "For the first stage of our approach, we define the radius R 
    //.     for seed point selection to maintain a fixed ratio ra between
    //      the area of the non-maximum suppression circle πR^2 and the area 
    //      of the image wh. Specifically, we set ra=100 and calculate R 
    //.     as follows R = sqrt(wh / (π r_a))"
    //
    // Meaning:
    //     - The NMS region area πR² = total_image_area / r_a
    //     - Ensures seed points are evenly distributed spatially
    // ----------------------------------------------------------------------
    double globalSeedRadius_R;
    {
        double imageArea = static_cast<double>(imageSize1.width) *
                           static_cast<double>(imageSize1.height);
        globalSeedRadius_R = std::sqrt(imageArea / (CV_PI * params_.ratio));
    }

    // Sort indices by confidence descending
    std::vector<int> sortedIndices(matches.size());
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
    std::sort(sortedIndices.begin(), sortedIndices.end(),
              [&](int a, int b) {
                  return matches[a].confidence > matches[b].confidence;
              });

    std::vector<bool> isSuppressed(matches.size(), false);
    std::vector<ScoredMatch> seedPoints;
    seedPoints.reserve(matches.size() / 4);

    // ----------------------------------------------------------------------
    // Non-maximum suppression:
    // Keep the highest-confidence match, suppress all within radius R.
    // ----------------------------------------------------------------------
    for (int idx : sortedIndices) {
        if (isSuppressed[idx]) continue;

        const auto& seedMatch = matches[idx];
        const cv::Point2f& seedPoint = keypoints1[seedMatch.match.queryIdx].pt;

        // Suppress all matches inside the NMS radius
        for (size_t j = 0; j < matches.size(); ++j) {
            if (isSuppressed[j]) continue;

            const auto& otherMatch = matches[j];
            const cv::Point2f& otherPoint = keypoints1[otherMatch.match.queryIdx].pt;

            double dx = seedPoint.x - otherPoint.x;
            double dy = seedPoint.y - otherPoint.y;
            if (std::sqrt(dx * dx + dy * dy) <= globalSeedRadius_R) {
                isSuppressed[j] = true;
            }
        }

        seedPoints.push_back(seedMatch);
    }

    std::cout << "GaLAM: Seed points selected = " << seedPoints.size() << std::endl;
    return seedPoints;
}

std::vector<GaLAM::ScoredMatch> GaLAM::localNeighborhoodSelection(
    const std::vector<ScoredMatch>& matches, 
    const std::vector<ScoredMatch>& seedPoints, 
    const std::vector<cv::KeyPoint>& keypoints1, 
    const std::vector<cv::KeyPoint>& keypoints2,
    const cv::Size& imageSize1,
    const cv::Size& imageSize2
) const
{
    std::vector<ScoredMatch> filtered = filterByDistance(matches, seedPoints, keypoints1, keypoints2);

    filtered = filterByScaleRotation(filtered, seedPoints, keypoints1, keypoints2);

    filtered = filterByImageScale(filtered, seedPoints, keypoints1, keypoints2, imageSize1, imageSize2);

    return filtered;
}

// Actually assuming that R1 and R2 are arbitrary, and different for each correspondence
std::vector<GaLAM::ScoredMatch> GaLAM::filterByDistance(
    const std::vector<ScoredMatch>& matches, 
    const std::vector<ScoredMatch>& seedPoints, 
    const std::vector<cv::KeyPoint>& keypoints1, 
    const std::vector<cv::KeyPoint>& keypoints2
) const
{
    return std::vector<ScoredMatch>();
}

std::vector<GaLAM::ScoredMatch> GaLAM::filterByScaleRotation(
    const std::vector<ScoredMatch>& matches, 
    const std::vector<ScoredMatch>& seedPoints, 
    const std::vector<cv::KeyPoint>& keypoints1, 
    const std::vector<cv::KeyPoint>& keypoints2
) const
{
    return std::vector<ScoredMatch>();
}

// Actually assuming that R1 and R2 are arbitrary, and different for each correspondence
std::vector<GaLAM::ScoredMatch> GaLAM::filterByImageScale(
    const std::vector<ScoredMatch>& matches, 
    const std::vector<ScoredMatch>& seedPoints, 
    const std::vector<cv::KeyPoint>& keypoints1, 
    const std::vector<cv::KeyPoint>& keypoints2, 
    const cv::Size& imageSize1, 
    const cv::Size& imageSize2
) const
{
}

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

    // 1) Bidirectional NN + ratio test
    std::vector<ScoredMatch> filtered = filterBidirectionalNN(descriptors1, descriptors2);
    if (filtered.empty()) {
        std::cout << "GaLAM: No matches after bidirectional NN." << std::endl;
        return {};
    }

    // 2) Assign confidence (1 / distance)
    assignConfidenceScore(filtered);

    // 3) Seed selection with non-maximum suppression
    std::vector<ScoredMatch> seeds = selectPoints(filtered, keypoints1, imageSize1);

    // Convert seeds to plain cv::DMatch for the outside world
    std::vector<cv::DMatch> seedMatches;
    seedMatches.reserve(seeds.size());
    for (const auto& seed : seeds) {
        seedMatches.push_back(seed.match);
    }

    std::cout << "GaLAM: Returning " << seedMatches.size()
              << " seed matches (Stage 1 only)" << std::endl;

    // TODO: Stage 1 - Local affine verification
    // TODO: Stage 2 - Global geometric consistency

    return seedMatches;
}

//} // namespace galam
