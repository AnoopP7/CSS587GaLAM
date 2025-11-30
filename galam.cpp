#include "galam.h"
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <set>

//namespace galam {

GaLAM::GaLAM(const InputParameters& params)
    : params_(params) {}

std::vector<GaLAM::ScoredMatch> GaLAM::selectSeedPoints(
    std::vector<ScoredMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints1,
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    const cv::Size& imageSize1
) const
{
    // 1) Bidirectional NN + ratio test
    matches = filterBidirectionalNN(descriptors1, descriptors2);
    if (matches.empty()) {
        std::cout << "GaLAM: No matches after bidirectional NN." << std::endl;
        return std::vector<ScoredMatch>();
    }

    // 2) Assign confidence (1 / distance)
    assignConfidenceScore(matches);

    // 3) Seed selection with non-maximum suppression
    std::vector<ScoredMatch> seeds = selectPoints(matches, keypoints1, imageSize1);

    return seeds;
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
        scored.secondMatch = secondBestMatch;
        scored.confidence = 0.0;
        validMatches.push_back(scored);
    }

    std::cout << "GaLAM: Bidirectional NN matches = " << validMatches.size() << std::endl;
    return validMatches;
}

// 2) Assign confidence score (reciprocal of ratio test value)
void GaLAM::assignConfidenceScore(
    std::vector<ScoredMatch>& matches
) const
{
    for (auto& scored : matches) {
        double distance = std::max(static_cast<double>(scored.match.distance), 1e-6);
        double secondDistance = std::max(static_cast<double>(scored.secondMatch.distance), 1e-6);

        // Calculate ratio test value
        scored.confidence = 1.0 / (distance / secondDistance);
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
        if (s == 100) {
            std::cout << s << std::endl;
        }
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

            // Check that this is not a seed point
            if (seed.match.queryIdx == m.match.queryIdx && seed.match.trainIdx == m.match.trainIdx) continue;

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
            double dAlpha    = std::fabs(normalizeAngle(alphaCand - alphaSeed));

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

            // 3) Scale constraint (Eq.3)
            // TODO: Test this since they might not be exactly equal
            if (R2 == R1 / sigmaSeed) continue;

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

void GaLAM::affineVerification(
    std::vector<ScoredMatch>& matches,
    std::vector<ScoredMatch>& seedPoints,
    std::vector<cv::KeyPoint>& keypoints1,
    std::vector<cv::KeyPoint>& keypoints2,
    std::vector<std::set<int>>& neighborhoods,
    const cv::Size& imageSize1,
    const cv::Size& imageSize2
) const
{
    // Standardize coordinates
    std::vector<cv::Point2f> normalizedKeypoints1(keypoints1.size());
    std::vector<cv::Point2f> normalizedKeypoints2(keypoints2.size());

    preprocessSets(matches, seedPoints, keypoints1, keypoints2, neighborhoods, imageSize1, imageSize2, normalizedKeypoints1, normalizedKeypoints2);

    // Get affine transformations for each neighborhood and filter out points
    std::vector<cv::Mat> transformations = fitTransformationMatrix(matches, seedPoints, keypoints1, keypoints2, neighborhoods,
        imageSize2, normalizedKeypoints1, normalizedKeypoints2);
}

// Assuming that R1 and R2 are the same R1 and R2 from earlier and that we don't normalize the coordinates of the seed points
void GaLAM::preprocessSets(
    const std::vector<ScoredMatch>& matches,
    const std::vector<ScoredMatch>& seedPoints,
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    std::vector<std::set<int>>& neighborhoods,
    const cv::Size& imageSize1,
    const cv::Size& imageSize2,
    std::vector<cv::Point2f>& normalizedKeypoints1,
    std::vector<cv::Point2f>& normalizedKeypoints2
) const
{
    // precompute base radius R1 and R2
    double R1 = computeBaseRadius(imageSize1);
    double R2 = computeBaseRadius(imageSize2);

    // Go through each neighborhood
    for (size_t i = 0; i < neighborhoods.size(); i++) {
        // Go through matches in this neighborhood
        for (size_t neighborhoodMatch : neighborhoods[i]) {
            // Get the coordinates for the seed points
            const ScoredMatch& seedPoint = seedPoints[i];
            const cv::DMatch& seedMatch = seedPoint.match;
            int query = seedMatch.queryIdx;
            int train = seedMatch.trainIdx;
            const cv::KeyPoint& seedPoint1 = keypoints1[query];
            const cv::KeyPoint& seedPoint2 = keypoints2[train];

            // Get the coordinates of each of the keypoints
            const ScoredMatch& match = matches[neighborhoodMatch];
            const cv::DMatch& dmatch = match.match;
            query = dmatch.queryIdx;
            train = dmatch.trainIdx;
            const cv::KeyPoint& keypoint1 = keypoints1[query];
            const cv::KeyPoint& keypoint2 = keypoints2[train];

            // Modify the coordinates
            normalizedKeypoints1[query].x = (keypoint1.pt.x - seedPoint1.pt.x) / (params_.lambda1 * R1);
            normalizedKeypoints1[query].y = (keypoint1.pt.y - seedPoint1.pt.y) / (params_.lambda1 * R1);

            normalizedKeypoints2[train].x = (keypoint2.pt.x - seedPoint2.pt.x) / (params_.lambda1 * R2);
            normalizedKeypoints2[train].y = (keypoint2.pt.y - seedPoint2.pt.y) / (params_.lambda1 * R2);
            //keypoint1.pt.x = (keypoint1.pt.x - seedPoint1.pt.x) / (params_.lambda1 * R1);
            //keypoint2.pt.x = (keypoint2.pt.x - seedPoint2.pt.x) / (params_.lambda1 * R2);
        }
    }
}

// Assuming that we need TWO, not THREE and that we should remove if not
// Might be selecting one with RANSAC
std::vector<cv::Mat> GaLAM::fitTransformationMatrix(
    std::vector<ScoredMatch>& matches,
    std::vector<ScoredMatch>& seedPoints,
    std::vector<cv::KeyPoint>& keypoints1,
    std::vector<cv::KeyPoint>& keypoints2,
    std::vector<std::set<int>>& neighborhoods,
    const cv::Size& imageSize2,
    std::vector<cv::Point2f>& normalizedKeypoints1,
    std::vector<cv::Point2f>& normalizedKeypoints2
) const
{
    // Check that we have at least 2 points; if we don't, remove this seed point and neighborhood
    for (size_t i = 0; i < neighborhoods.size(); i++) {
        if (neighborhoods[i].size() < 2) {
            neighborhoods.erase(neighborhoods.begin() + i);
            seedPoints.erase(seedPoints.begin() + i);
            --i;    // decrement index if we removed an item
        }
    }

    // Create vector of affine transformations
    std::vector<cv::Mat> transforms;

    // Use RANSAC to fit an affine transformation matrix and evaluate it, maintaining the best one
    // for loop 128
        // fit
        // evaluate
        // score
    // select the best one and filter based on it FOR THAT SEED POINT
    
    // Get threshold (same unless R2 is different for each seed point)
    double R2 = computeBaseRadius(imageSize2);
    double threshold = params_.lambda2 / (params_.lambda1 * R2);

    // Iterate through each seed point's neighborhood
    for (size_t neighborhood = 0; neighborhood < neighborhoods.size(); neighborhood++) {
        // Build the vectors to use for fitting
        std::vector<cv::Point2f> points1;
        std::vector<cv::Point2f> points2;

        for (int match : neighborhoods[neighborhood]) {
            points1.push_back(normalizedKeypoints1[matches[match].match.queryIdx]);
            points2.push_back(normalizedKeypoints2[matches[match].match.trainIdx]);
            //points1.push_back(keypoints1[matches[match].match.queryIdx].pt);
            //points2.push_back(keypoints2[matches[match].match.trainIdx].pt);
        }

        // Use RANSAC to fit the affine transformation matrix
        // TODO: Verify that this approach is consistent with the paper
        cv::Mat optimalTransformation;
        int bestScore = -1;

        for (int j = 0; j < params_.num_iterations; j++) {
            cv::Mat transformation = cv::estimateAffinePartial2D(points1, points2, cv::noArray(), cv::RANSAC, 3, 1, 0.99, 10);

            // Move on if we couldn't fit a matrix
            if (transformation.empty()) {
                //std::cout << "Couldn't fit a matrix" << std::endl;
                continue;
            }

            // Find the residual rk for each correspondence point pair in the neighborhood
            int score = 0;
            for (int match : neighborhoods[neighborhood]) {
                double rk = measureAffineResidual(transformation, matches[match], normalizedKeypoints1, normalizedKeypoints2);

                // Compare rk against threshold and count the number of rk below threshold
                if (rk <= threshold) ++score;
            }

            // If this transformation has more rk below threshold, select it as the optimal transformation
            if (score > bestScore) {
                optimalTransformation = transformation;
                bestScore = score;
            }
        }
        
        if (optimalTransformation.empty()) {
            std::cerr << "Couldn't find an affine transformation for neighborhood " << neighborhood << std::endl;
        }
        // Keep best affine transformation
        transforms.push_back(optimalTransformation);

        // Candidate correspondences with residuals below threshold are kept, others removed
        // need to iterate through matches, check if their residual was less than threshold, and keep if so
        std::vector<int> toRemove;
        for (int match : neighborhoods[neighborhood]) {
            double rk = measureAffineResidual(optimalTransformation, matches[match], normalizedKeypoints1, normalizedKeypoints2);

            // Compare rk against threshold and remove the match if above threshold
            if (rk > threshold) {
                toRemove.push_back(match); // TODO: Make sure this actually erases the correct element
            }
        }
        for (int idx : toRemove) {
            neighborhoods[neighborhood].erase(idx);
        }
    }

    return transforms;
}

double GaLAM::measureAffineResidual(
    const cv::Mat& transformation,
    const ScoredMatch& correspondence,
    const std::vector<cv::Point2f>& normalizedKeypoints1,
    const std::vector<cv::Point2f>& normalizedKeypoints2
) const
{
    // Get the normalized points
    const cv::Point2f& point1 = normalizedKeypoints1[correspondence.match.queryIdx];
    const cv::Point2f& point2 = normalizedKeypoints2[correspondence.match.trainIdx];

    // Convert points to Mat for matrix multiplication
    // TODO: Do we need to transpose?
    cv::Mat_<double> matPoint1(3, 1);
    matPoint1(0, 0) = point1.x;
    matPoint1(1, 0) = point1.y;
    matPoint1(2, 0) = 1.0;

    //std::cout << "(" << point1.x << ", " << point1.y << ") = <" << matPoint1 << ">" << std::endl;

    cv::Mat_<double> matPoint2(2, 1);
    matPoint2(0, 0) = point2.x;
    matPoint2(1, 0) = point2.y;

    // Compute matrix multiplication
    cv::Mat result = (transformation * matPoint1) - matPoint2;
    //std::cout << "transformation:\n" << transformation << "\n\nmatPoint1:\n" << matPoint1 << "\n\nmatPoint2:\n" << matPoint2 << std::endl;

    // Get norm of resulting vector
    // Should this be L2 norm or something else?
    double residual = cv::norm(result, cv::NORM_L2);

    return residual;
}

// Stage 2
std::vector<cv::DMatch> GaLAM::globalGeometryVerification(
    const std::vector<ScoredMatch>& matches, 
    const std::vector<ScoredMatch>& seedPoints, 
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<std::set<int>>& neighborhoods
) const
{
    // 1.Use RANSAC to sample minimal sets of 8 seeds
    // 2.Estimate fundamental matrix from each set (8 points)
    // 3.For every seed, compute epipolar deviation r_i (F_t)
    // Equation 7,8
    // 4.Model siupport o_t = number of seed with r_i (epsilon)
    // 5.After all models, compute o_max = max_t o_t
    // 6.Keep strong matches models with o_t >= lambda3 * o_max
     // 7.Any seed that is inlier for at least one strong model is marked as inlier
    // 8.Final correspondences = union of neighborhoods of those seeds.

    // RANSAC ranking and local support Ni (number of inlier matches in 
    // that seed's neighborhood after stage 1)

    // number of seeds
    const int numSeeds = static_cast<int>(seedPoints.size());

    // RANSACSampling Parameters
    int minSampleSize = params_.minSampleSize; // 8
    int num_iterations = params_.num_iterations; // 128
    double epsilon = params_.epsilon; // epipolar threshold
    double lambda3 = params_.lambda3; // lamda3 from paper

    // Check if there is enough seeds to estimate fundamental matrix
    if (numSeeds == 0) {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): No seeds provided." << std::endl;
        return {};
    }

    if (matches.empty()) {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): No matches provided." << std::endl;
        return {};
    }

    if (static_cast<int>(neighborhoods.size()) != numSeeds) {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): neighborhoods.size() ("
                  << neighborhoods.size() << ") does not match seedPoints.size() ("
                  << numSeeds << ")." << std::endl;
        return {};
    }

    if (numSeeds < minSampleSize) {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): Not enough seeds for 8-point "
                  << "fundamental matrix (" << numSeeds << " < "
                  << minSampleSize << ")." << std::endl;
        return {};
    }

    // Sample multiple fundamental matrices
    std::vector<int> modelSupports;                  // o_t for each model
    std::vector<std::vector<bool>> modelSeedInliers; // which seeds satisfied r_i(F_t) < threshold

    // pre-allocation 
    modelSupports.reserve(num_iterations);
    modelSeedInliers.reserve(num_iterations);

    // RNG for sampling
    cv::RNG rng((uint64)cv::getTickCount());

    for (int iter = 0; iter < num_iterations; ++iter) {
        // RANSAC sampling of minSampleSize from top poolSize seeds
        std::vector<int> sampleSeedIndices;
        sampleSeedIndices.reserve(minSampleSize);
        std::vector<bool> used(numSeeds, false);

        int attempts = 0;
        while (static_cast<int>(sampleSeedIndices.size()) < minSampleSize && 
                                        attempts < 100 * minSampleSize) {
            int idx = rng.uniform(0, numSeeds); // uniform sampling from [0, numSeeds)
            if (used[idx]) {
                ++attempts;
                continue;
            }
            used[idx] = true;
            sampleSeedIndices.push_back(idx);
        }

        // if we could not sample enough unique seeds, skip this iteration
        if (static_cast<int>(sampleSeedIndices.size()) < minSampleSize) {
            continue;
        }

        // 8-points sample and estimate F_t
        std::vector<cv::Point2f> pts1, pts2;
        pts1.reserve(minSampleSize);
        pts2.reserve(minSampleSize);

        for (int seedIdx : sampleSeedIndices) {
            const ScoredMatch& sm = seedPoints[seedIdx];
            const cv::KeyPoint& kp1 = keypoints1[sm.match.queryIdx];
            const cv::KeyPoint& kp2 = keypoints2[sm.match.trainIdx];
            pts1.push_back(kp1.pt);
            pts2.push_back(kp2.pt);
        }

        // Estimate fundamental matrix from 8-point sample
        cv::Mat F = cv::findFundamentalMat(
            pts1, pts2,
            cv::FM_8POINT
        );

        if (F.empty()) {
            // degenerate configuration
            continue;
        }

        cv::Matx33d Fm;
        F.convertTo(Fm, CV_64F);


        // Evaluate seeds with Equation 7 and 8 and compute o_t
        //   Eq (7): l'_i = F_t x_i
        //   Eq (8): r_i(F_t) = |x'_i^T l'_i| / sqrt(a_i^2 + b_i^2)
        std::vector<bool> seedInlier(numSeeds, false);
        int supportCount = 0; // this is o_t for this model

        for (int i = 0; i < numSeeds; ++i) {
            const ScoredMatch& sm = seedPoints[i];
            const cv::KeyPoint& kp1 = keypoints1[sm.match.queryIdx];
            const cv::KeyPoint& kp2 = keypoints2[sm.match.trainIdx];

            cv::Vec3d x(kp1.pt.x, kp1.pt.y, 1.0);
            cv::Vec3d xp(kp2.pt.x, kp2.pt.y, 1.0);

            // Eq. (7): epipolar line in image 2
            cv::Vec3d l = Fm * x;
            double a = l[0], b = l[1], c = l[2];

            double denom = std::sqrt(a * a + b * b);
            if (denom < 1e-12) {
                // degenerate epipolar line
                continue;
            }

            // Eq. (8): epipolar distance from x' to line l'
            double num = std::fabs(a * xp[0] + b * xp[1] + c);
            double r   = num / denom;

            if (r <= epsilon) {
                seedInlier[i] = true;
                ++supportCount;
            }
        }

        if (supportCount == 0) {
            // F_t explains no seeds
            continue;
        }

        modelSupports.push_back(supportCount);
        modelSeedInliers.push_back(std::move(seedInlier));
    }

    // if no valid models, we should handle fallback to Stage 1 results
    // return stage 1 results instead
    if (modelSupports.empty()) {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): No valid fundamental "
                  << "matrix models were generated." << std::endl;
        return {};
    }

    // find maximum support o_max
    // lambda3 * o_max means model m is strong
    int omax = *std::max_element(modelSupports.begin(), modelSupports.end());
    if (omax <= 0) {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): All models have zero "
                  << "support (omax <= 0)." << std::endl;
        return {};
    }

    // mark strong models
    std::vector<bool> modelIsGood(modelSupports.size(), false);
    for (size_t m = 0; m < modelSupports.size(); ++m) {
        if (modelSupports[m] >= lambda3 * omax) {
            modelIsGood[m] = true;
        }
    }

    // a seed is kept if it is inlier for at least one strong model
    // if lambda3 filtering removes everything, we fallback to best model only
    // if not strong models, fallback to stage 1
    std::vector<bool> keepSeed(numSeeds, false);

    for (size_t m = 0; m < modelSupports.size(); ++m) {
        if (!modelIsGood[m]) continue;
        const auto& seedInlier = modelSeedInliers[m];
        for (int i = 0; i < numSeeds; ++i) {
            if (seedInlier[i]) {
                keepSeed[i] = true;
            }
        }
    }

    bool anySeedKept = false;
    for (int i = 0; i < numSeeds; ++i) {
        if (keepSeed[i]) { 
            anySeedKept = true; 
            break; 
        }
    }

    // if λ3-strong models kept no seeds, we still use the best model.
    if (!anySeedKept) {
        int bestModelIdx = 0;
        for (std::size_t m = 1; m < modelSupports.size(); ++m) {
            if (modelSupports[m] > modelSupports[bestModelIdx]) {
                bestModelIdx = static_cast<int>(m);
            }
        }

        const auto& bestSeedInliers = modelSeedInliers[bestModelIdx];
        for (int i = 0; i < numSeeds; ++i) {
            keepSeed[i] = bestSeedInliers[i];
        }
    }

    // build final correspondences = union of neighborhoods of kept seeds
    std::vector<bool> isGoodMatch(matches.size(), false);
    for (int i = 0; i < numSeeds; ++i) {
        if (!keepSeed[i]) continue;

        // add neighborhood of seed i
        const auto& neigh = neighborhoods[i];
        for (int idx : neigh) {
            if (idx >= 0 && idx < static_cast<int>(matches.size())) {
                isGoodMatch[idx] = true;
            }
        }
    }

    std::vector<cv::DMatch> finalMatches;
    finalMatches.reserve(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        if (isGoodMatch[i]) {
            finalMatches.push_back(matches[i].match);
        }
    }

    std::cout << "STAGE 2 (RANSAC): returning "
              << finalMatches.size()
              << " matches after global geometry detection."
              << std::endl;

    return finalMatches;

}


void GaLAM::localAffineVerification(
    std::vector<cv::KeyPoint>& keypoints1,
    std::vector<cv::KeyPoint>& keypoints2,
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    const cv::Size& imageSize1,
    const cv::Size& imageSize2,
    std::vector<ScoredMatch>& seedPoints,
    std::vector<std::set<int>>& neighborhoods,
    std::vector<ScoredMatch>& matches
) const
{
    // Get seed points
    seedPoints = selectSeedPoints(matches, keypoints1, descriptors1, descriptors2, imageSize1);

    // Get neighborhoods for each seed point
    neighborhoods = localNeighborhoodSelection(matches, seedPoints, keypoints1, keypoints2, imageSize1, imageSize2);

    // Perform affine verification
    affineVerification(matches, seedPoints, keypoints1, keypoints2, neighborhoods, imageSize1, imageSize2);
}

GaLAM::StageResults GaLAM::detectOutliers(
    std::vector<cv::KeyPoint>& keypoints1,
    std::vector<cv::KeyPoint>& keypoints2,
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    const std::vector<cv::DMatch>& candidateMatches,
    const cv::Size& imageSize1,
    const cv::Size& imageSize2
) const
{
    StageResults results;
    std::cout << "GaLAM: Processing " << candidateMatches.size()
              << " candidate matches..." << std::endl;

    // Stage 1: Local affine verification
    std::vector<ScoredMatch> seedPoints;
    std::vector<std::set<int>> neighborhoods;
    std::vector<ScoredMatch> matches;

    localAffineVerification(keypoints1, keypoints2, descriptors1, descriptors2, 
                            imageSize1, imageSize2, seedPoints, neighborhoods, matches);

    // Store seed matches for visualization
    for (const auto& seed : seedPoints) {
        results.seedMatches.push_back(seed.match);
    }

    std::cout << "GaLAM: Built " << neighborhoods.size() 
              << " local neighborhoods" << std::endl;

    // Collect all unique inlier indices from neighborhoods
    std::set<int> inlierIndices;
    for (size_t s = 0; s < neighborhoods.size(); ++s) {
        for (int idx : neighborhoods[s]) {
            inlierIndices.insert(idx);
        }
    }
    
    std::cout << "GaLAM: Stage 1 produced " << inlierIndices.size() 
              << " unique inlier candidates" << std::endl;

    // Convert inlier indices to cv::DMatch
    for (int idx : inlierIndices) {
        results.stage1Matches.push_back(matches[idx].match);
    }

    std::cout << "GaLAM: Returning " << results.stage1Matches.size()
              << " matches after Stage 1 (Local Affine Matching)" << std::endl;

    // Stage 2: Global geometric consistency
    results.finalMatches = globalGeometryVerification(
        matches, seedPoints, keypoints1, keypoints2, neighborhoods
    );

    // If Stage 2 returns empty, fallback to Stage 1 results
    if (results.finalMatches.empty()) {
        std::cout << "GaLAM: Stage 2 returned no matches, using Stage 1 results" << std::endl;
        results.finalMatches = results.stage1Matches;
    }

    std::cout << "GaLAM: Final matches after Stage 2: " << results.finalMatches.size() << std::endl;

    return results;
}

//} // namespace galam
