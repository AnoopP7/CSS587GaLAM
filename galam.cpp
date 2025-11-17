#include "galam.h"

#include <opencv2/features2d.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cmath>

namespace galam {

GaLAM::GaLAM(const InputParameters& params)
    : params_(params) {}

std::vector<GaLAM::ScoredMatch> GaLAM::filterBidirectionalNN(
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2
) const
{
    std::vector<std::vector<cv::DMatch>> knn12, knn21;

    cv::BFMatcher matcher(cv::NORM_L2);
    matcher.knnMatch(descriptors1, descriptors2, knn12, 2);
    matcher.knnMatch(descriptors2, descriptors1, knn21, 2);

    std::vector<ScoredMatch> good;

    for (int i = 0; i < static_cast<int>(knn12.size()); ++i) {
        if (knn12[i].size() < 2) continue;

        const cv::DMatch& m1 = knn12[i][0];
        const cv::DMatch& m2 = knn12[i][1];

        // Ratio test
        if (m1.distance >= params_.rt_threshold * m2.distance)
            continue;

        int q = m1.queryIdx;
        int t = m1.trainIdx;

        if (t < 0 || t >= static_cast<int>(knn21.size())) continue;
        if (knn21[t].empty()) continue;

        const cv::DMatch& back = knn21[t][0];
        if (back.trainIdx != q) continue;

        ScoredMatch sm;
        sm.match = m1;
        sm.confidence = 0.0;
        good.push_back(sm);
    }

    std::cout << "GaLAM: Bidirectional NN matches = " << good.size() << std::endl;
    return good;
}

// 2) Assign confidence score (reciprocal of distance)
void GaLAM::assignConfidenceScore(
    std::vector<ScoredMatch>& matches
) const
{
    for (auto& sm : matches) {
        double d = std::max(static_cast<double>(sm.match.distance), 1e-6);
        sm.confidence = 1.0 / d;
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

    double R;
    if (params_.ra > 0.0) {
        double area = static_cast<double>(imageSize1.width) *
                      static_cast<double>(imageSize1.height);
        R = std::sqrt(area / (CV_PI * params_.ra));
    } else {
        R = params_.radius;
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

        const auto& mSeed = matches[idx];
        const cv::Point2f& p = keypoints1[mSeed.match.queryIdx].pt;

        // suppress neighbors around this seed
        for (size_t j = 0; j < matches.size(); ++j) {
            if (suppressed[j]) continue;
            const auto& m = matches[j];
            const cv::Point2f& q = keypoints1[m.match.queryIdx].pt;

            double dx = p.x - q.x;
            double dy = p.y - q.y;
            if (std::sqrt(dx * dx + dy * dy) <= R) {
                suppressed[j] = true;
            }
        }

        seeds.push_back(mSeed);
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
    for (const auto& s : seeds) {
        seedMatches.push_back(s.match);
    }

    std::cout << "GaLAM: Returning " << seedMatches.size()
              << " seed matches (Stage 1 only)" << std::endl;

    // TODO: Stage 1 - Local affine verification
    // TODO: Stage 2 - Global geometric consistency

    return seedMatches;
}

} // namespace galam
