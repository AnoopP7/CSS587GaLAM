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
    double globalSeedRadius_R1;
    {
        double imageArea = static_cast<double>(imageSize1.width) *
                           static_cast<double>(imageSize1.height);
        globalSeedRadius_R1 = std::sqrt(imageArea / (CV_PI * params_.ratio));
    }

    double globalSeedRadius_R2;
    {
        double imageArea = static_cast<double>(imageSize2.width) *
                           static_cast<double>(imageSize2.height);
        globalSeedRadius_R2 = std::sqrt(imageArea / (CV_PI * params_.ratio));
    }
    std::vector<ScoredMatch> filtered = filterByDistance(matches, seedPoints, keypoints1, keypoints2);

    filtered = filterByScaleRotation(filtered, seedPoints, keypoints1, keypoints2);

    filtered = filterByImageScale(filtered, seedPoints, keypoints1, keypoints2, imageSize1, imageSize2);

    return filtered;
}

// Assuming R1 = R image size 1 
// Assuming R2 = R image size 2
// Actually assuming that R1 and R2 are arbitrary, and different for each correspondence
// std::vector<GaLAM::ScoredMatch> GaLAM::filterByDistance(
//     const std::vector<ScoredMatch>& matches, 
//     const std::vector<ScoredMatch>& seedPoints, 
//     const std::vector<cv::KeyPoint>& keypoints1, 
//     const std::vector<cv::KeyPoint>& keypoints2,
//     const double radius1,
//     const double radius2
// ) const
// {
//     // Distance beetween the first keypoints of the match and the seed point
//     // seedpoints correspondences
//     for (const auto& seed : seedPoints) {
//         // calculate index in image 1
//         // calculate index in image 2
//     }
//     return std::vector<ScoredMatch>();
// }

// std::vector<GaLAM::ScoredMatch> GaLAM::filterByScaleRotation(
//     const std::vector<ScoredMatch>& matches, 
//     const std::vector<ScoredMatch>& seedPoints, 
//     const std::vector<cv::KeyPoint>& keypoints1, 
//     const std::vector<cv::KeyPoint>& keypoints2
// ) const
// {
//     return std::vector<ScoredMatch>();
// }

// // Actually assuming that R1 and R2 are arbitrary, and different for each correspondence
// std::vector<GaLAM::ScoredMatch> GaLAM::filterByImageScale(
//     const std::vector<ScoredMatch>& matches, 
//     const std::vector<ScoredMatch>& seedPoints, 
//     const std::vector<cv::KeyPoint>& keypoints1, 
//     const std::vector<cv::KeyPoint>& keypoints2, 
//     const cv::Size& imageSize1, 
//     const cv::Size& imageSize2,
//     const double radius1,
//     const double radius2
// ) const
// {
// }

std::vector<std::set<int>> GaLAM::localNeighborhoodSelection(
    const std::vector<ScoredMatch>& matches, 
    const std::vector<ScoredMatch>& seedPoints, 
    const std::vector<cv::KeyPoint>& keypoints1, 
    const std::vector<cv::KeyPoint>& keypoints2, 
    const cv::Size& imageSize1, 
    const cv::Size& imageSize2
) const 
{
    // each seed point will have a set of indices of matches that are in its neighborhood
    std::vector<std::set<int>> neighborhoods;

    // early exit if no matches or no seed points
    if (matches.empty() || seedPoints.empty()) {
        return neighborhoods;
    }

    // compute R1_base = sqrt(wh / (π * ra))
    double imageArea = static_cast<double>(imageSize1.width) *
                       static_cast<double>(imageSize1.height);

    double R1_base = std::sqrt(imageArea / (CV_PI * params_.ratio));

    // prepare output to make sure each seed gets one neighborhood set
    neighborhoods.resize(seedPoints.size());

    // Outer loop:
    // loop through each seed point
    for (size_t s = 0; s < seedPoints.size(); ++s) {

        const ScoredMatch& seed = seedPoints[s];

        int seedIdxImg1 = seed.match.queryIdx;
        int seedIdxImg2 = seed.match.trainIdx;

        const cv::KeyPoint& seedKP1 = keypoints1[seedIdxImg1];
        const cv::KeyPoint& seedKP2 = keypoints2[seedIdxImg2];

        // seed scale ratio σ_S = σ2 / σ1
        double sigma1S = std::max((double)seedKP1.size, 1e-6);
        double sigma2S = std::max((double)seedKP2.size, 1e-6);
        double sigmaSeed = sigma2S / sigma1S;

        // compute radii R1_i and R2_i
        double R1 = R1_base;
        double R2 = R1 / std::max(sigmaSeed, 1e-6);

        // seed rotation α_S
        double alphaSeed = seedKP2.angle - seedKP1.angle;
        alphaSeed = std::fmod(alphaSeed + 540.0, 360.0) - 180.0;

        // Inner loop:
        // loop through all candidate matches
        for (size_t i = 0; i < matches.size(); ++i) {

            const ScoredMatch& cand = matches[i];

            int idx1 = cand.match.queryIdx;
            int idx2 = cand.match.trainIdx;

            const cv::KeyPoint& candKP1 = keypoints1[idx1];
            const cv::KeyPoint& candKP2 = keypoints2[idx2];

            //1) Distance constraint (Eq.1)
            double dx1 = candKP1.pt.x - seedKP1.pt.x;
            double dy1 = candKP1.pt.y - seedKP1.pt.y;
            double dist1 = std::sqrt(dx1*dx1 + dy1*dy1);

            double dx2 = candKP2.pt.x - seedKP2.pt.x;
            double dy2 = candKP2.pt.y - seedKP2.pt.y;
            double dist2 = std::sqrt(dx2*dx2 + dy2*dy2);

            bool distanceOK =
                (dist1 <= params_.lambda1 * R1) &&
                (dist2 <= params_.lambda1 * R2);

            if (!distanceOK) continue;

            //2) Rotation + scale constraint (Eq.2)
            double alphaCand = candKP2.angle - candKP1.angle;
            alphaCand = std::fmod(alphaCand + 540.0, 360.0) - 180.0;

            // rotation difference
            double dAlpha = std::fabs(alphaSeed - alphaCand);
            if (dAlpha > 180.0) dAlpha = 360.0 - dAlpha;

            // check rotation threshold
            bool rotationOK = (dAlpha <= params_.tAlpha);

            // scale ratio σ_C = σ2 / σ1
            double sigma1 = std::max((double)candKP1.size, 1e-6);
            double sigma2 = std::max((double)candKP2.size, 1e-6);
            double sigmaCand = sigma2 / sigma1;

            // log scale difference
            double logScaleDiff = std::fabs(std::log(sigmaSeed / sigmaCand));
            bool scaleOK = (logScaleDiff <= params_.tSigma);

            //3) Image-scale consistency (it is always true for now)
            // We can implement image-scale consistency check based on image dimensions
            bool imageScaleOK = true;

            if (distanceOK && rotationOK && scaleOK && imageScaleOK) {
                neighborhoods[s].insert((int)i);
            }
        }
    }

    return neighborhoods;
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
