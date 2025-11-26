#include "galam.h"
#include <opencv2/features2d.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <set>

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

// std::vector<GaLAM::ScoredMatch> GaLAM::localNeighborhoodSelection(
//     const std::vector<ScoredMatch>& matches, 
//     const std::vector<ScoredMatch>& seedPoints, 
//     const std::vector<cv::KeyPoint>& keypoints1, 
//     const std::vector<cv::KeyPoint>& keypoints2,
//     const cv::Size& imageSize1,
//     const cv::Size& imageSize2
// ) const
// {
//     double globalSeedRadius_R1;
//     {
//         double imageArea = static_cast<double>(imageSize1.width) *
//                            static_cast<double>(imageSize1.height);
//         globalSeedRadius_R1 = std::sqrt(imageArea / (CV_PI * params_.ratio));
//     }

//     double globalSeedRadius_R2;
//     {
//         double imageArea = static_cast<double>(imageSize2.width) *
//                            static_cast<double>(imageSize2.height);
//         globalSeedRadius_R2 = std::sqrt(imageArea / (CV_PI * params_.ratio));
//     }
//     std::vector<ScoredMatch> filtered = filterByDistance(matches, seedPoints, keypoints1, keypoints2);

//     filtered = filterByScaleRotation(filtered, seedPoints, keypoints1, keypoints2);

//     filtered = filterByImageScale(filtered, seedPoints, keypoints1, keypoints2, imageSize1, imageSize2);

//     return filtered;
// }

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

// helper: normalize angle difference (in degrees) to range [-180, 180)
static double normalizeAngle(double angle) {
    while (angle <= -180.0) angle += 360.0;
    while (angle >  180.0)  angle -= 360.0;
    return angle;
}

// Compute base radius R1 from image size
// Larger images --> larger R1
double GaLAM::computeBaseRadius(const cv::Size& imageSize) const
{
    // compute total number of pixels in the first image
    double area = static_cast<double>(imageSize.width) *
                  static_cast<double>(imageSize.height);
                  
    // ra = ratio from the paper
    // larger ra --> smaller Radius
    double ra = params_.ratio;

    // R1 = sqrt(area / (π * ra))
    double R1 = std::sqrt(area / (CV_PI * ra));
    return R1;
}

// Local neighborhood selection
// for each seed point, find matches in its neighborhood
// according to distance, rotation, scale constraints
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

    // precompute base radius R1
    double R1 = computeBaseRadius(imageSize1);
    double R2 = computeBaseRadius(imageSize2);
    // allocate one neighborhood per seed match
    neighborhoods.resize(seedPoints.size());

    // read GaLAM parameters
    double lambda1 = params_.lambda1; // spatial multiplier fro R1, R2
    double tAlpha  = params_.tAlpha;  // max allowed rotation difference
    double tSigma  = params_.tSigma;  // max allowed log-scale difference

    // Outer loop: for each seed point
    // compute R2 based on seed scale
    for (size_t s = 0; s < seedPoints.size(); ++s) {
        const ScoredMatch& seed = seedPoints[s];
        // extract the corresponding keypoints in image 1 and 2 for this seed
        const cv::KeyPoint& kp1Seed = keypoints1[seed.match.queryIdx];
        const cv::KeyPoint& kp2Seed = keypoints2[seed.match.trainIdx];

        // seed scale in each image: ratio σ_S = σ2 / σ1
        // clamp to avoid division by zero or log(0)
        double sigma1Seed = std::max(static_cast<double>(kp1Seed.size), 1e-6);
        double sigma2Seed = std::max(static_cast<double>(kp2Seed.size), 1e-6);
        double sigmaSeed  = sigma2Seed / sigma1Seed;
        // compute R2 if needed 
        //double R2 = R1 / sigmaSeed;

        // Seed's relative rotation between image 1 and 2
        // seed rotation α_S
        // angle(image2) - angle(image1), normalized to a canonical range
        double alphaSeed = normalizeAngle(kp2Seed.angle - kp1Seed.angle);

        // get reference to this seed's neighborhood set
        std::set<int>& neigh = neighborhoods[s];

        // Inner loop:
        // For each match, test spatial, rotation, scale constraints if it belongs to this seed's neighborhood
        for (size_t i = 0; i < matches.size(); ++i) {
            const ScoredMatch& m = matches[i];
            const cv::KeyPoint& kp1 = keypoints1[m.match.queryIdx];
            const cv::KeyPoint& kp2 = keypoints2[m.match.trainIdx];

            // 1) Distance constraint (Eq.1)
            // spatial constraint in image1
            double dx1 = kp1.pt.x - kp1Seed.pt.x;
            double dy1 = kp1.pt.y - kp1Seed.pt.y;
            double d1  = std::sqrt(dx1 * dx1 + dy1 * dy1);
            if (d1 > lambda1 * R1) continue;

            // spatial constraint in image2
            double dx2 = kp2.pt.x - kp2Seed.pt.x;
            double dy2 = kp2.pt.y - kp2Seed.pt.y;
            double d2  = std::sqrt(dx2 * dx2 + dy2 * dy2);
            if (d2 > lambda1 * R2) continue;

            // 2) Rotation constraint (Eq.2)
            // rotation constraint
            // compute candidate match relative rotation
            double alphaCand = normalizeAngle(kp2.angle - kp1.angle);
            // absolute difference in rotation between candidate and seed
            double dAlpha    = std::fabs(alphaCand - alphaSeed);

            // if the rotation difference is too large, reject this candidate
            if (dAlpha > tAlpha) continue;

            // Scale constraint in log domain (Eq.2)
            // similar to rotation constraint but for scale: | log( sigmaCand / sigmaSeed ) | <= tSigma
            double sigma1C = std::max(static_cast<double>(kp1.size), 1e-6);
            double sigma2C = std::max(static_cast<double>(kp2.size), 1e-6);
            double sigmaCand = sigma2C / sigma1C;
            double ratio     = sigmaCand / sigmaSeed;
            // if scale difference in log domain is too large, reject this candidate
            if (std::fabs(std::log(ratio)) > tSigma) continue;

            // if we reach here, the match passes all constraints for this seed
            // add index 'i' to the neighborhood of seed 's'
            neigh.insert(static_cast<int>(i));
        }

        // debug output
        std::cout << "GaLAM: Neighborhood " << s
                  << " size = " << neigh.size() << std::endl;
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
