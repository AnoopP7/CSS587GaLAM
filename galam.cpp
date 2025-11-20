#include "galam.h"

#include <opencv2/features2d.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cmath>

//namespace galam {

GaLAM::GaLAM(const InputParameters& params)
    : params_(params) {}

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

    std::vector<ScoredMatch> goodMatches;

    for (int i = 0; i < static_cast<int>(knn12.size()); ++i) {
        if (knn12[i].size() < 2) continue;

        const cv::DMatch& match1 = knn12[i][0];
        const cv::DMatch& match2 = knn12[i][1];

        // Ratio test
        if (match1.distance >= params_.rt_threshold * match2.distance)
            continue;

        int query = match1.queryIdx;
        int train = match1.trainIdx;

        if (train < 0 || train >= static_cast<int>(knn21.size())) continue;
        if (knn21[train].empty()) continue;

        const cv::DMatch& back = knn21[train][0];
        if (back.trainIdx != query) continue;

        ScoredMatch scored;
        scored.match = match1;
        scored.confidence = 0.0;
        goodMatches.push_back(scored);
    }

    std::cout << "GaLAM: Bidirectional NN matches = " << goodMatches.size() << std::endl;
    return goodMatches;
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
std::vector<GaLAM::ScoredMatch> GaLAM::selectSeedPoints(
    const std::vector<ScoredMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints1,
    const cv::Size& imageSize1
) const
{
    if (matches.empty()) return {};

    double radius;
    if (params_.ratio > 0.0) {
        double area = static_cast<double>(imageSize1.width) *
                      static_cast<double>(imageSize1.height);
        radius = std::sqrt(area / (CV_PI * params_.ratio));
    } else {
        radius = params_.radius;
    }

    // Sort by confidence descending
    std::vector<int> order(matches.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) {
                  return matches[a].confidence > matches[b].confidence;
              });

    std::vector<bool> suppressed(matches.size(), false);
    std::vector<ScoredMatch> seeds;
    seeds.reserve(matches.size() / 4);

    for (int idx : order) {
        if (suppressed[idx]) continue;

        const auto& matchSeed = matches[idx];
        const cv::Point2f& point = keypoints1[matchSeed.match.queryIdx].pt;

        // suppress neighbors around this seed
        for (size_t j = 0; j < matches.size(); ++j) {
            if (suppressed[j]) continue;
            const auto& nextMatch = matches[j];
            const cv::Point2f& queryPoint = keypoints1[nextMatch.match.queryIdx].pt;

            double dx = point.x - queryPoint.x;
            double dy = point.y - queryPoint.y;
            if (std::sqrt(dx * dx + dy * dy) <= radius) {
                suppressed[j] = true;
            }
        }

        seeds.push_back(matchSeed);
    }

    std::cout << "GaLAM: Seed points selected = " << seeds.size() << std::endl;
    return seeds;
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
    std::vector<ScoredMatch> seeds = selectSeedPoints(filtered, keypoints1, imageSize1);

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
