/*
 * galam.cpp
 * Implements the GaLAM outlier detection algorithm in C++
 * Implementation authors: Yu Dinh, Neha Kotwal, Anoop Prasad
 * Paper title: GaLAM: Two-Stage Outlier Detection Algorithm
 * Paper authors: X. Lu, Z. Yan, Z. Fan
 *
 * This file contains the implementation of the GaLAM class methods.
 *
 * Purpose:
 * - Implement the GaLAM outlier detection algorithm
 * - Provide functions for seed point selection, local neighborhood selection,
 *  affine verification, and global geometry verification
 * - Implement a robust error handling mechanism
 * - Ensure compatibility with OpenCV library
 *
 * High-level overview of GaLAM algorithm:
 * - Stage 1: Local Affine Matching
 *    - Select seed points using bidirectional nearest neighbor matching,
 *      ratio test, and non-maximum suppression
 *    - For each seed point, find local neighborhoods based on spatial,
 *      appearance, and geometric consistency
 *    - Perform affine verification within each neighborhood to filter out outliers
 * - Stage 2: Global Geometric Consistency
 *    - Use RANSAC to fit a global geometric model (fundamental matrix)
 *    - Evaluate the model over all seed points and their neighborhoods
 *    - Select strong models and corresponding inlier matches
 *    - Final RT thresholding to filter matches 0.9 to ensure no bad matches passed through
 *    - Return the final set of inlier matches after both stages
 *
 * Note:
 * - This implementation uses OpenCV for image processing and feature matching
 * - The code is structured to allow easy modification of parameters and integration
 * with other systems
 * - Error handling is implemented to manage potential issues during processing
<<<<<<< Updated upstream
=======
 *
 * Future Work: Implement OpenCV outlier detection interface
>>>>>>> Stashed changes
 */

#include "galam.h"

#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <set>

// Constructor
// Constructs a GaLAM object based on the given InputParameters struct
// Preconditions: params includes valid parameters for GaLAM algorithm
// Postconditions: Initializes GaLAM object using the params
/*
 * Parameters:
 * params: InputParameters struct containing parameters to use for the algorithm
 *
 * Return:
 * GaLAM object
 */
GaLAM::GaLAM(const InputParameters &params)
    : params_(params), radius1(1.0), radius2(1.0) {}

// selectSeedPoints
// Selects seed points to be used throughout the algorithm
// Preconditions: Valid parameters are provided
// Postconditions: Returns a vector of seed points
/*
 * Parameters:
 * matches: Empty vector of ScoredMatch to store all matches found
 * keypoints1: vector of cv::KeyPoint containing the keypoints from the first image
 * descriptors1: cv::Mat containing the descriptors for the first image
 * descriptors2: cv::Mat containing the descriptors for the second image
 * imageSize1: cv::Size representing the size of the first image
 *
 * Return:
 * std::vector<GaLAM::ScoredMatch> of seed points found
 */
std::vector<GaLAM::ScoredMatch> GaLAM::selectSeedPoints(
    std::vector<ScoredMatch> &matches,
    const std::vector<cv::KeyPoint> &keypoints1,
    const cv::Mat &descriptors1,
    const cv::Mat &descriptors2,
    const cv::Size &imageSize1) const
{
    // Perform bidirectional nearest neighbor matching to filter out unreliable matches
    matches = filterBidirectionalNN(descriptors1, descriptors2);

    // If no matches were found, return empty vector
    if (matches.empty())
    {
        std::cout << "GaLAM: No matches after bidirectional NN." << std::endl;
        return std::vector<ScoredMatch>();
    }

    // Assign confidence score to each match (reciprocal of ratio test value)
    assignConfidenceScore(matches);

    // Select highest scoring points within a radius as seed points using non-maximum suppression
    std::vector<ScoredMatch> seeds = selectPoints(matches, keypoints1, imageSize1);

    return seeds;
}

// filterBidirectionalNN
// Filters out unreliable matches using bidirectional nearest neighbor matching
// Preconditions: Descriptors for both images are provided
// Postconditions: Returns a vector of matches
/*
 * Parameters:
 * descriptors1: cv::Mat containing the descriptors for the first image
 * descriptors2: cv::Mat containing the descriptors for the second image
 *
 * Return:
 * std::vector<GaLAM::ScoredMatch> of matches found so far
 */
std::vector<GaLAM::ScoredMatch> GaLAM::filterBidirectionalNN(
    const cv::Mat &descriptors1,
    const cv::Mat &descriptors2) const
{
    std::vector<std::vector<cv::DMatch>> knn12, knn21;


    // Could be made faster by using external match results
    cv::BFMatcher matcher(cv::NORM_L2);
    matcher.knnMatch(descriptors1, descriptors2, knn12, 2);
    matcher.knnMatch(descriptors2, descriptors1, knn21, 2);

    // FLANN -- Not many matches
    //cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
    //matcher->knnMatch(descriptors1, descriptors2, knn12, 2);
    //matcher->knnMatch(descriptors2, descriptors1, knn21, 2);

    std::vector<ScoredMatch> validMatches;

    // Iterate through first direction
    for (int i = 0; i < static_cast<int>(knn12.size()); ++i)
    {
        // If not enough data to compare, continue to next
        if (knn12[i].size() < 2)
            continue;

        // Get best two matches from image 1 to image 2
        const cv::DMatch &bestMatch = knn12[i][0];
        const cv::DMatch &secondBestMatch = knn12[i][1];

        // Apply ratio test -- not explicitly mentioned in the paper but seems to improve accuracy
        if (bestMatch.distance >= params_.rt_threshold * secondBestMatch.distance)
            continue;

        // Get index of best match in each image
        int queryIdx = bestMatch.queryIdx;
        int trainIdx = bestMatch.trainIdx;

        // Check validity of index and match in other image
        if (trainIdx < 0 || trainIdx >= static_cast<int>(knn21.size()))
            continue;
        if (knn21[trainIdx].empty())
            continue;

        // Perform bidirectional check from image 2 to image 1
        const cv::DMatch &reverseMatch = knn21[trainIdx][0];
        if (reverseMatch.trainIdx != queryIdx)
            continue;

        // If all conditions satisfied, add this match to return vector
        ScoredMatch scored;
        scored.match = bestMatch;
        scored.secondMatch = secondBestMatch;
        scored.confidence = 0.0;
        validMatches.push_back(scored);
    }

    std::cout << "GaLAM: Bidirectional NN matches = " << validMatches.size() << std::endl;
    return validMatches;
}

// assignConfidenceScore
// Assign confidence score to each match, defined as the reciprocal of the ratio test value
// Preconditions: vector of matches is provided
// Postconditions: The score for each ScoredMatch is assigned
/*
 * Parameters:
 * matches: vector of ScoredMatch to store all matches found
 *
 * Return:
 * None (outputs are via reference parameter)
 */
void GaLAM::assignConfidenceScore(
    std::vector<ScoredMatch> &matches) const
{
    // Assign a score of the reciprocal of the ratio test score to each match
    for (auto &scored : matches)
    {
        double distance = std::max(static_cast<double>(scored.match.distance), 1e-6);
        double secondDistance = std::max(static_cast<double>(scored.secondMatch.distance), 1e-6);
        scored.confidence = 1.0 / (distance / secondDistance);
    }
}

// selectPoints
// Selects high-scoring points as seed points using non-maximum suppression
// Preconditions: vector of matches with initialized scores and keypoints and size for first image are provided
// Postconditions: Returns the vector of seed points found
/*
 * Parameters:
 * matches: vector of ScoredMatch to store all matches found, along with their scores
 * keypoints1: vector of cv::KeyPoint containing the keypoints from the first image
 * imageSize1: cv::Size representing the size of the first image
 *
 * Return:
 * std::vector<GaLAM::ScoredMatch> of seed points found
 */
std::vector<GaLAM::ScoredMatch> GaLAM::selectPoints(
    const std::vector<ScoredMatch> &matches,
    const std::vector<cv::KeyPoint> &keypoints1,
    const cv::Size &imageSize1) const
{
    // Return empty vector if we found no matches
    if (matches.empty())
        return {};

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

    // Calculate the radius for non-maximum suppression
    double imageArea = static_cast<double>(imageSize1.width) *
                       static_cast<double>(imageSize1.height);
    double globalSeedRadius_R = std::sqrt(imageArea / (CV_PI * params_.ratio));

    // Sort indices by confidence, descending
    std::vector<int> sortedIndices(matches.size());
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
    std::sort(sortedIndices.begin(), sortedIndices.end(),
              [&](int a, int b)
              {
                  return matches[a].confidence > matches[b].confidence;
              });

    // ----------------------------------------------------------------------
    // Non-maximum suppression:
    // Keep the highest-confidence match, suppress all within radius R.
    // ----------------------------------------------------------------------
    std::vector<bool> isSuppressed(matches.size(), false);
    std::vector<ScoredMatch> seedPoints;
    seedPoints.reserve(matches.size() / 4);

    // Iterate through matches in order of confidence
    for (int idx : sortedIndices)
    {
        if (isSuppressed[idx])
            continue;

        const auto &seedMatch = matches[idx];
        const cv::Point2f &seedPoint = keypoints1[seedMatch.match.queryIdx].pt;

        // Suppress all matches inside the NMS radius
        for (size_t j = 0; j < matches.size(); ++j)
        {
            if (isSuppressed[j])
                continue;

            const auto &otherMatch = matches[j];
            const cv::Point2f &otherPoint = keypoints1[otherMatch.match.queryIdx].pt;

            // Compute distance
            double dx = seedPoint.x - otherPoint.x;
            double dy = seedPoint.y - otherPoint.y;
            if (std::sqrt(dx * dx + dy * dy) <= globalSeedRadius_R)
            {
                isSuppressed[j] = true;
            }
        }

        // Keep track of the seed point we selected
        seedPoints.push_back(seedMatch);
    }

    std::cout << "GaLAM: Seed points selected = " << seedPoints.size() << std::endl;
    return seedPoints;
}

// normalizeAngle
// Helper function to normalize angle difference (in degrees) to the range [-180, 180)
// Preconditions: angle to normalize is provided as parameter
// Postconditions: Returns the normalized angle
/*
 * Parameters:
 * angle: double of the angle to be normalized
 *
 * Return:
 * double of the normalized angle
 */
static double normalizeAngle(double angle)
{
    while (angle <= -180.0)
        angle += 360.0;
    while (angle > 180.0)
        angle -= 360.0;
    return angle;
}

// computeBaseRadius
// Helper function to compute the radius R based on image size
// Preconditions: Image size is provided
// Postconditions: Returns the radius for the image as a double
/*
 * Parameters:
 * imageSize: cv::Size for the image is provided
 *
 * Return:
 * double of the radius for that image
 */
double GaLAM::computeBaseRadius(const cv::Size &imageSize) const
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

// localNeighborhoodSelection
// Finds neighborhoods for each seed point
// Preconditions: Valid parameters are provided
// Postconditions: Returns the neighborhoods as a vector of sets
/*
 * Parameters:
 * matches: vector of ScoredMatch to store all matches found
 * seedPoints: vector of ScoredMatch of the seed points
 * keypoints1: vector of cv::KeyPoint containing the keypoints from the first image
 * keypoints2: vector of cv::KeyPoint containing the keypoints from the second image
 * imageSize1: cv::Size representing the size of the first image
 * imageSize2: cv::Size representing the size of the second image
 *
 * Return:
 * std::vector<std::set<int>> where set n contains all the neighborhood matches for seed point n in seedPoints
 */
std::vector<std::set<int>> GaLAM::localNeighborhoodSelection(
    const std::vector<ScoredMatch> &matches,
    const std::vector<ScoredMatch> &seedPoints,
    const std::vector<cv::KeyPoint> &keypoints1,
    const std::vector<cv::KeyPoint> &keypoints2,
    const cv::Size &imageSize1,
    const cv::Size &imageSize2) const
{
    std::vector<std::set<int>> neighborhoods;

    // Early exit if no matches or no seed points
    if (matches.empty() || seedPoints.empty())
    {
        return neighborhoods;
    }

    // Base radius R1 from image 1 (eq. for R1 in the paper)
    double R1 = computeBaseRadius(imageSize1);

    // Allocate one neighborhood per seed
    neighborhoods.resize(seedPoints.size());

    double lambda1 = params_.lambda1; // spatial multiplier
    double tAlpha = params_.tAlpha;   // max allowed rotation difference (deg)
    double tSigma = params_.tSigma;   // max allowed log-scale difference

    // Iterate through seed points
    for (size_t s = 0; s < seedPoints.size(); ++s)
    {
        // Get keypoint data for the seed point
        const ScoredMatch &seed = seedPoints[s];
        const cv::KeyPoint &kp1Seed = keypoints1[seed.match.queryIdx];
        const cv::KeyPoint &kp2Seed = keypoints2[seed.match.trainIdx];

        // Seed scale ratio σ_S = σ2 / σ1
        double sigma1Seed = std::max(static_cast<double>(kp1Seed.size), 1e-6);
        double sigma2Seed = std::max(static_cast<double>(kp2Seed.size), 1e-6);
        double sigmaSeed = sigma2Seed / sigma1Seed;

        // Per-seed R2 (paper: R2 = R1 / σ_S)
        double R2_seed = R1 / sigmaSeed;
        if (R2_seed <= 1e-6)
            continue; // degenerate, skip this seed

        // Seed rotation α_S = angle2 − angle1 (normalized)
        double alphaSeed = normalizeAngle(kp2Seed.angle - kp1Seed.angle);

        std::set<int> &neigh = neighborhoods[s];

        // Iterate through each of the matches to determine what neighborhood, if any, they should go to
        for (size_t i = 0; i < matches.size(); ++i)
        {
            const ScoredMatch &m = matches[i];
            const cv::KeyPoint &kp1 = keypoints1[m.match.queryIdx];
            const cv::KeyPoint &kp2 = keypoints2[m.match.trainIdx];

            // Skip if this is exactly the same correspondence as the seed
            if (seed.match.queryIdx == m.match.queryIdx &&
                seed.match.trainIdx == m.match.trainIdx)
            {
                continue;
            }

            // ----------------------------------------------------------------------
            // 1) Distance constraint (Eq. 1)
            // ----------------------------------------------------------------------
            // Image 1
            double dx1 = kp1.pt.x - kp1Seed.pt.x;
            double dy1 = kp1.pt.y - kp1Seed.pt.y;
            double d1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
            if (d1 > lambda1 * R1)
            {
                continue;
            }

            // Image 2 (uses per-seed R2)
            double dx2 = kp2.pt.x - kp2Seed.pt.x;
            double dy2 = kp2.pt.y - kp2Seed.pt.y;
            double d2 = std::sqrt(dx2 * dx2 + dy2 * dy2);
            if (d2 > lambda1 * R2_seed)
            {
                continue;
            }

            // ----------------------------------------------------------------------
            // 2) Rotation + scale in log-domain (Eq. 2)
            // ----------------------------------------------------------------------

            // rotation difference
            double alphaCand = normalizeAngle(kp2.angle - kp1.angle);
            double dAlpha = std::fabs(normalizeAngle(alphaCand - alphaSeed));
            if (dAlpha > tAlpha)
            {
                continue;
            }

            // scale ratio in candidate
            double sigma1C = std::max(static_cast<double>(kp1.size), 1e-6);
            double sigma2C = std::max(static_cast<double>(kp2.size), 1e-6);
            double sigmaCand = sigma2C / sigma1C;
            double ratio = sigmaCand / sigmaSeed;

            if (std::fabs(std::log(ratio)) > tSigma)
            {
                continue;
            }

            // ----------------------------------------------------------------------
            // (Eq. 3 is omitted here; Eq. 1 + Eq. 2 already enforce locality and Eq. 3 is used to obtain R2)
            // ----------------------------------------------------------------------

            // passes all constraints → add to neighborhood
            neigh.insert(static_cast<int>(i));
        }

         std::cout << "GaLAM: Neighborhood " << s
                   << " size = " << neigh.size() << std::endl;
    }

    return neighborhoods;
}

// affineVerification
// Fits an affine transformation matrix and filters out points which do not match the transformation
// Preconditions: Valid parameters are provided
// Postconditions: Neighborhood points which do not fit the affine transformation are removed
/*
 * Parameters:
 * matches: vector of ScoredMatch to store all matches found
 * seedPoints: vector of ScoredMatch of the seed points
 * keypoints1: vector of cv::KeyPoint containing the keypoints from the first image
 * keypoints2: vector of cv::KeyPoint containing the keypoints from the second image
 * neighborhoods: std::vector<std::set<int>> of neighborhoods corresponding to seedPoints
 * imageSize1: cv::Size representing the size of the first image
 * imageSize2: cv::Size representing the size of the second image
 *
 * Return:
 * std::vector<std::set<int>> where set n contains all the neighborhood matches for seed point n in seedPoints
 */
void GaLAM::affineVerification(
    std::vector<ScoredMatch> &matches,
    std::vector<ScoredMatch> &seedPoints,
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2,
    std::vector<std::set<int>> &neighborhoods,
    const cv::Size &imageSize1,
    const cv::Size &imageSize2) const
{
    // Standardize coordinates
    std::vector<cv::Point2f> normalizedKeypoints1(keypoints1.size());
    std::vector<cv::Point2f> normalizedKeypoints2(keypoints2.size());

    preprocessSets(matches, seedPoints, keypoints1, keypoints2, neighborhoods, imageSize1, imageSize2, normalizedKeypoints1, normalizedKeypoints2);

    // Get affine transformations for each neighborhood and filter out points
    std::vector<cv::Mat> transformations = fitTransformationMatrix(matches, seedPoints, keypoints1, keypoints2, neighborhoods, imageSize1,
                                                                   imageSize2, normalizedKeypoints1, normalizedKeypoints2);
}

// preprocessSets
// Normalizes the coordinates for each neighborhood based on the seed point
// Preconditions: Valid parameters are provided
// Postconditions: normalizedKeypoints1 and normalizedKeypoints2 are populated with normalized keypoints
/*
 * Parameters:
 * matches: vector of ScoredMatch to store all matches found
 * seedPoints: vector of ScoredMatch of the seed points
 * keypoints1: vector of cv::KeyPoint containing the keypoints from the first image
 * keypoints2: vector of cv::KeyPoint containing the keypoints from the second image
 * neighborhoods: std::vector<std::set<int>> of neighborhoods corresponding to seedPoints
 * imageSize1: cv::Size representing the size of the first image
 * imageSize2: cv::Size representing the size of the second image
 * normalizedKeypoints1: Empty vector for storing normalized keypoints from the first image
 * normalizedKeypoints2: Empty vector for storing normalized keypoints from the second image
 */
void GaLAM::preprocessSets(
    const std::vector<ScoredMatch> &matches,
    const std::vector<ScoredMatch> &seedPoints,
    const std::vector<cv::KeyPoint> &keypoints1,
    const std::vector<cv::KeyPoint> &keypoints2,
    std::vector<std::set<int>> &neighborhoods,
    const cv::Size &imageSize1,
    const cv::Size &imageSize2,
    std::vector<cv::Point2f> &normalizedKeypoints1,
    std::vector<cv::Point2f> &normalizedKeypoints2) const
{
    // Base radius R1 from image 1 (eq. for R1 in the paper)
    double R1 = computeBaseRadius(imageSize1);

    // Iterate through each seed point
    for (size_t s = 0; s < neighborhoods.size(); ++s)
    {
        const ScoredMatch &seedPoint = seedPoints[s];
        const cv::DMatch &seedMatch = seedPoint.match;
        int seed_q = seedMatch.queryIdx;
        int seed_t = seedMatch.trainIdx;

        const cv::KeyPoint &seedPoint1 = keypoints1[seed_q];
        const cv::KeyPoint &seedPoint2 = keypoints2[seed_t];

        // Compute per-seed R2 based on seed's scale ratio
        double sigma1Seed = std::max(static_cast<double>(seedPoint1.size), 1e-6);
        double sigma2Seed = std::max(static_cast<double>(seedPoint2.size), 1e-6);
        double sigmaSeed = sigma2Seed / sigma1Seed;
        double R2_seed = R1 / sigmaSeed;

        // For each match in this neighborhood, normalize its coordinates
        for (int matchIdx : neighborhoods[s])
        {
            const ScoredMatch &match = matches[matchIdx];
            const cv::DMatch &dmatch = match.match;

            int q = dmatch.queryIdx;
            int t = dmatch.trainIdx;

            const cv::KeyPoint &keypoint1 = keypoints1[q];
            const cv::KeyPoint &keypoint2 = keypoints2[t];

            // Normalize coordinates based on seed points
            normalizedKeypoints1[q].x =
                (keypoint1.pt.x - seedPoint1.pt.x) / (params_.lambda1 * R1);
            normalizedKeypoints1[q].y =
                (keypoint1.pt.y - seedPoint1.pt.y) / (params_.lambda1 * R1);

            // IMPORTANT: index by 't' (image 2), not 'q'
            normalizedKeypoints2[t].x =
                (keypoint2.pt.x - seedPoint2.pt.x) / (params_.lambda1 * R2_seed);
            normalizedKeypoints2[t].y =
                (keypoint2.pt.y - seedPoint2.pt.y) / (params_.lambda1 * R2_seed);
        }
    }
}

// fitTransformationMatrix
// Fits an affine transformation matrix to each neighborhood using RANSAC and filters out points that don't fit it
// Preconditions: Valid parameters are provided
// Postconditions: Points which do not fit the best affine transformation are filtered out from neighborhoods
/*
 * Parameters:
 * matches: vector of ScoredMatch to store all matches found
 * seedPoints: vector of ScoredMatch of the seed points
 * keypoints1: vector of cv::KeyPoint containing the keypoints from the first image
 * keypoints2: vector of cv::KeyPoint containing the keypoints from the second image
 * neighborhoods: std::vector<std::set<int>> of neighborhoods corresponding to seedPoints
 * imageSize1: cv::Size representing the size of the first image
 * imageSize2: cv::Size representing the size of the second image
 * normalizedKeypoints1: vector of normalized keypoints from the first image
 * normalizedKeypoints2: vector of normalized keypoints from the second image
 */
std::vector<cv::Mat> GaLAM::fitTransformationMatrix(
    std::vector<ScoredMatch> &matches,
    std::vector<ScoredMatch> &seedPoints,
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2,
    std::vector<std::set<int>> &neighborhoods,
    const cv::Size &imageSize1,
    const cv::Size &imageSize2,
    std::vector<cv::Point2f> &normalizedKeypoints1,
    std::vector<cv::Point2f> &normalizedKeypoints2) const
{
    // Check that we have at least 2 points; if we don't, remove this seed point and neighborhood
    std::vector<ScoredMatch> newSeeds;
    std::vector<std::set<int>> newNeighborhoods;

    // Iterate through existing neighborhoods and only keep those with at least 2 points
    size_t limit = std::min(neighborhoods.size(), seedPoints.size());
    for (size_t i = 0; i < limit; ++i)
    {
        if (neighborhoods[i].size() >= 2)
        {
            newSeeds.push_back(seedPoints[i]);
            newNeighborhoods.push_back(neighborhoods[i]);
        }
    }

    // Update seedPoints and neighborhoods to only include those with at least 2 points
    seedPoints = std::move(newSeeds);
    neighborhoods = std::move(newNeighborhoods);

    // Create vector of affine transformations
    std::vector<cv::Mat> transforms;

    double R1 = computeBaseRadius(imageSize1);

    // Neighborhoods to remove
    std::vector<size_t> removeNeighborhood;

    // Iterate through each seed point's neighborhood
    for (size_t neighborhood = 0; neighborhood < neighborhoods.size(); neighborhood++)
    {
        // Compute per-seed R2 for threshold calculation
        const ScoredMatch &seedPoint = seedPoints[neighborhood];
        const cv::KeyPoint &kp1Seed = keypoints1[seedPoint.match.queryIdx];
        const cv::KeyPoint &kp2Seed = keypoints2[seedPoint.match.trainIdx];

        // Compute per-seed R2 based on seed's scale ratio
        double sigma1Seed = std::max(static_cast<double>(kp1Seed.size), 1e-6);
        double sigma2Seed = std::max(static_cast<double>(kp2Seed.size), 1e-6);
        double sigmaSeed = sigma2Seed / sigma1Seed;
        double R2_seed = R1 / sigmaSeed;

        double threshold = params_.lambda2 / (params_.lambda1 * R2_seed);

        // Build the vectors to use for fitting
        std::vector<cv::Point2f> points1;
        std::vector<cv::Point2f> points2;

        // Populate points1 and points2 from the neighborhood
        for (int match : neighborhoods[neighborhood])
        {
            points1.push_back(normalizedKeypoints1[matches[match].match.queryIdx]);
            points2.push_back(normalizedKeypoints2[matches[match].match.trainIdx]);
        }

        // Use RANSAC to fit the affine transformation matrix
        // Note: Paper runs PROSAC with 128 interations
        // Here, we use RANSAC with num_iterations from params
        // Calls estimateAffinePartial2D() repeatedly, which internally does its own RANSAC on all points
        cv::Mat optimalTransformation;
        int bestScore = -1;
        for (int j = 0; j < params_.num_iterations; j++)
        {
            // Carries out 1 RANSAC iteration to be evaluated
            cv::Mat transformation = cv::estimateAffinePartial2D(points1, points2, cv::noArray(), cv::RANSAC, 3, 1, 0.99, 10);

            // Find the residual rk for each correspondence point pair in the neighborhood if we found a transformation
            if (!transformation.empty())
            {
                int score = 0;
                for (int match : neighborhoods[neighborhood])
                {
                    double rk = measureAffineResidual(transformation, matches[match], normalizedKeypoints1, normalizedKeypoints2);

                    // Compare rk against threshold and count the number of rk below threshold
                    if (rk <= threshold)
                        ++score;
                }

                // If this transformation has more rk below threshold, select it as the optimal transformation
                if (score > bestScore)
                {
                    optimalTransformation = transformation;
                    bestScore = score;
                }
            }
        }

        // If we didn't find any optimal transformation, we should get rid of this neighborhood
        if (optimalTransformation.empty())
        {
            removeNeighborhood.push_back(neighborhood);
            continue;
        }

        // Keep best affine transformation
        transforms.push_back(optimalTransformation);

        // Candidate correspondences with residuals below threshold are kept, others removed
        std::vector<int> toRemove;
        for (int match : neighborhoods[neighborhood])
        {
            double rk = measureAffineResidual(optimalTransformation, matches[match], normalizedKeypoints1, normalizedKeypoints2);

            // Compare rk against threshold and remove the match if above threshold
            if (rk > threshold)
            {
                toRemove.push_back(match);
            }
        }

        // Remove points from neighborhood
        for (int idx : toRemove)
        {
            neighborhoods[neighborhood].erase(idx);
        }
    }

    // Remove neighborhoods for which we could not fit an affine transformation
    // sort indices in reverse order and then erase to ensure we remove the right indices and the removal does not affect the index
    // seedpoints and neighbours should match for stage 2 so remove from seedPoints as well
    std::sort(removeNeighborhood.rbegin(), removeNeighborhood.rend());
    for (size_t idx : removeNeighborhood)
    {
        neighborhoods.erase(neighborhoods.begin() + idx);
        seedPoints.erase(seedPoints.begin() + idx);
    }

    return transforms;
}

// measureAffineResidual
// Calculates the residual for an affine transformation and correspondence pair
// Preconditions: Valid parameters are provided
// Postconditions: Returns the residual as a double
/*
 * Parameters:
 * transformation: cv::Mat of the affine transformation to be evaluated
 * correspondence: ScoredMatch representing the correspondence pair we are using for evaluation
 * normalizedKeypoints1: vector of normalized keypoints from the first image
 * normalizedKeypoints2: vector of normalized keypoints from the second image
 */
double GaLAM::measureAffineResidual(
    const cv::Mat &transformation,
    const ScoredMatch &correspondence,
    const std::vector<cv::Point2f> &normalizedKeypoints1,
    const std::vector<cv::Point2f> &normalizedKeypoints2) const
{
    // Get the normalized points
    const cv::Point2f &point1 = normalizedKeypoints1[correspondence.match.queryIdx];
    const cv::Point2f &point2 = normalizedKeypoints2[correspondence.match.trainIdx];

    // Convert points to Mat for matrix multiplication
    cv::Mat_<double> matPoint1(3, 1);
    matPoint1(0, 0) = point1.x;
    matPoint1(1, 0) = point1.y;
    matPoint1(2, 0) = 1.0;

    // Convert point2 to Mat
    cv::Mat_<double> matPoint2(2, 1);
    matPoint2(0, 0) = point2.x;
    matPoint2(1, 0) = point2.y;

    // Compute matrix multiplication
    cv::Mat result = (transformation * matPoint1) - matPoint2;

    // Get norm of resulting vector
    double residual = cv::norm(result, cv::NORM_L2);

    return residual;
}

//================================================================
// Stage 2: Global Geometry Verification
//================================================================

// GlobalGeometryVerification
// RANSAC to fit fundamental matrix over seed points
// Evaluate models over all seed points
// Select strong models and seeds
// Return inlier matches
// Precondition: matches and seedPoints are non-empty
// Postcondition: returns inlier matches after global geometry verification
/*
 * Parameters:
 * matches: vector of ScoredMatch to store all matches found
 * seedPoints: vector of ScoredMatch of the seed points
 * keypoints1: vector of cv::KeyPoint containing the keypoints from the first image
 * keypoints2: vector of cv::KeyPoint containing the keypoints from the second image
 * neighborhoods: std::vector<std::set<int>> of neighborhoods corresponding to seedPoints
 *
 * Return:
 * std::vector<cv::DMatch> of inlier matches after global geometry verification
 */
std::vector<cv::DMatch> GaLAM::globalGeometryVerification(
    const std::vector<ScoredMatch> &matches,
    const std::vector<ScoredMatch> &seedPoints,
    const std::vector<cv::KeyPoint> &keypoints1,
    const std::vector<cv::KeyPoint> &keypoints2,
    const std::vector<std::set<int>> &neighborhoods) const
{
    // Number of seed points
    const int numSeeds = static_cast<int>(seedPoints.size());

    int minSampleSize = params_.minSampleSize;   // 8
    int num_iterations = params_.num_iterations; // 128
    double epsilon = params_.epsilon;            // epipolar threshold
    double lambda3 = params_.lambda3;            // lambda3 from paper

    // Basic sanity checks
    // Precondition: matches and seedPoints are non-empty
    if (numSeeds == 0)
    {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): No seeds provided." << std::endl;
        return {};
    }

    // Precondition: matches and seedPoints are non-empty
    if (matches.empty())
    {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): No matches provided." << std::endl;
        return {};
    }

    // Precondition: neighborhoods.size() == seedPoints.size()
    if (static_cast<int>(neighborhoods.size()) != numSeeds)
    {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): neighborhoods.size() ("
                  << neighborhoods.size() << ") does not match seedPoints.size() ("
                  << numSeeds << ")." << std::endl;
        return {};
    }

    //  Precondition: enough seeds for 8-point fundamental matrix
    if (numSeeds < minSampleSize)
    {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): Not enough seeds for 8-point "
                  << "fundamental matrix (" << numSeeds << " < "
                  << minSampleSize << ")." << std::endl;
        return {};
    }

    // Store model supports and which seeds are inliers per model
    std::vector<int> modelSupports;
    std::vector<std::vector<bool>> modelSeedInliers;
    // Pre-allocate space
    modelSupports.reserve(num_iterations);
    modelSeedInliers.reserve(num_iterations);

    // Random number generator
    cv::RNG rng((uint64)cv::getTickCount());

    // The RANSAC loop
    // For each iteration:
    //   1) Randomly sample 8 seed points
    //   2) Fit fundamental matrix using 8-point algorithm
    //   3) Evaluate model over all seed points
    //   4) Record model support and which seeds are inliers
    for (int iter = 0; iter < num_iterations; ++iter)
    {
        std::vector<int> sampleSeedIndices;
        sampleSeedIndices.reserve(minSampleSize);
        std::vector<bool> used(numSeeds, false);

        // Randomly sample minSampleSize unique seed points
        int attempts = 0;
        while (static_cast<int>(sampleSeedIndices.size()) < minSampleSize &&
               attempts < 100 * minSampleSize)
        {
            // Random index in [0, numSeeds)
            int idx = rng.uniform(0, numSeeds);
            if (used[idx])
            {
                ++attempts;
                continue;
            }
            used[idx] = true;
            sampleSeedIndices.push_back(idx);
        }

        if (static_cast<int>(sampleSeedIndices.size()) < minSampleSize)
            continue;

        // Build 8-point sample
        std::vector<cv::Point2f> pts1, pts2;
        pts1.reserve(minSampleSize);
        pts2.reserve(minSampleSize);

        // Gather the sampled seed points
        for (int seedIdx : sampleSeedIndices)
        {
            const ScoredMatch &sm = seedPoints[seedIdx];
            const cv::KeyPoint &kp1 = keypoints1[sm.match.queryIdx];
            const cv::KeyPoint &kp2 = keypoints2[sm.match.trainIdx];
            pts1.push_back(kp1.pt);
            pts2.push_back(kp2.pt);
        }

        // Fit fundamental matrix using 8-point algorithm
        cv::Mat fundamental = cv::findFundamentalMat(pts1, pts2, cv::FM_8POINT);
        if (fundamental.empty())
            continue;

        // Evaluate model over all seed points
        cv::Matx33d fundM;
        fundamental.convertTo(fundM, CV_64F);

        // Count inliers among all seed points
        std::vector<bool> seedInlier(numSeeds, false);
        int supportCount = 0;

        // For each seed point, compute the epipolar constraint error
        for (int i = 0; i < numSeeds; ++i)
        {
            // Get seed point
            const ScoredMatch &sm = seedPoints[i];
            const cv::KeyPoint &kp1 = keypoints1[sm.match.queryIdx];
            const cv::KeyPoint &kp2 = keypoints2[sm.match.trainIdx];

            // Homogeneous coordinates
            cv::Vec3d x(kp1.pt.x, kp1.pt.y, 1.0);
            cv::Vec3d xp(kp2.pt.x, kp2.pt.y, 1.0);

            // Epipolar line in image 2: l' = F * x
            cv::Vec3d l = fundM * x;
            double a = l[0], b = l[1], c = l[2];

            // Compute distance from point to epipolar line
            double denom = std::sqrt(a * a + b * b);
            if (denom < 1e-12)
                continue;

            // Distance r = |a*x' + b*y' + c| / sqrt(a^2 + b^2)
            double num = std::fabs(a * xp[0] + b * xp[1] + c);
            double residual = num / denom;

            // Check inlier condition
            if (residual <= epsilon)
            {
                seedInlier[i] = true;
                ++supportCount;
            }
        }

        // Record model support and seed inliers
        if (supportCount == 0)
            continue;

        // Store results
        modelSupports.push_back(supportCount);
        modelSeedInliers.push_back(std::move(seedInlier));
    }

    // After RANSAC, check if any models were generated and find the maximum support
    if (modelSupports.empty())
    {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): No valid fundamental "
                  << "matrix models were generated." << std::endl;
        return {};
    }

    // Find maximum model support
    int omax = *std::max_element(modelSupports.begin(), modelSupports.end());
    if (omax <= 0)
    {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): All models have zero support (omax <= 0)."
                  << std::endl;
        return {};
    }

    // Mark strong models and keep seeds that are inliers to any strong model
    std::vector<bool> keepSeed(numSeeds, false);
    for (size_t m = 0; m < modelSupports.size(); ++m)
    {
        // Check if this model is "strong" (support >= lambda3 * omax)
        if (modelSupports[m] >= lambda3 * omax)
        {
            // Keep seeds that are inliers to this model
            const auto &seedInlier = modelSeedInliers[m];
            for (int i = 0; i < numSeeds; ++i)
            {
                // Check if this seed is an inlier to the current model
                if (seedInlier[i])
                    keepSeed[i] = true;
            }
        }
    }

    // Check if any seeds survived
    bool anySeedKept = false;
    for (int i = 0; i < numSeeds; ++i)
    {
        if (keepSeed[i])
        {
            anySeedKept = true;
            break;
        }
    }

    // If λ3 filtering removed everything, fall back to best single model
    if (!anySeedKept)
    {
        std::cout << "[WARNING] GaLAM Stage 2: Lambda3 filtering removed all seeds, using best model." << std::endl;
        int bestModelIdx = 0;
        // Find model with maximum support
        for (size_t m = 1; m < modelSupports.size(); ++m)
        {
            // Check if this model has higher support
            if (modelSupports[m] > modelSupports[bestModelIdx])
                bestModelIdx = static_cast<int>(m);
        }

        // Keep seeds that are inliers to the best model
        const auto &bestSeedInliers = modelSeedInliers[bestModelIdx];
        for (int i = 0; i < numSeeds; ++i)
            keepSeed[i] = bestSeedInliers[i];
    }

    // Union of neighborhoods of kept seeds → final matches
    std::vector<bool> isGoodMatch(matches.size(), false);
    for (int i = 0; i < numSeeds; ++i)
    {
        if (!keepSeed[i])
            continue;

        // Add all matches in this seed's neighborhood
        const auto &neigh = neighborhoods[i];
        // Mark matches as good
        for (int idx : neigh)
        {
            if (idx >= 0 && idx < static_cast<int>(matches.size()))
                isGoodMatch[idx] = true;
        }
    }

    // Report statistics
    int keptSeeds = 0;
    for (int i = 0; i < numSeeds; ++i)
        if (keepSeed[i])
            ++keptSeeds;

    std::cout << "GaLAM: Stage 2: kept " << keptSeeds << " / " << numSeeds << " seeds" << std::endl;

    // Collect final matches
    // Return only matches marked as good
    std::vector<cv::DMatch> finalMatches;
    finalMatches.reserve(matches.size());
    // Iterate through matches and add those marked as good
    for (size_t i = 0; i < matches.size(); ++i)
    {
        if (isGoodMatch[i])
            finalMatches.push_back(matches[i].match);
    }

    std::cout << "GaLAM: Stage 2: returning " << finalMatches.size()
              << " matches after global geometry detection." << std::endl;

    return finalMatches;
}

// localAffineVerification
//  Combines seed point selection, neighborhood selection, and affine verification
//  Preconditions: Valid parameters are provided
//  Postconditions: seedPoints, neighborhoods, and matches are populated after local affine verification
/*
 * Parameters:
 * keypoints1: vector of cv::KeyPoint containing the keypoints from the first image
 * keypoints2: vector of cv::KeyPoint containing the keypoints from the second image
 * descriptors1: cv::Mat containing the descriptors from the first image
 * descriptors2: cv::Mat containing the descriptors from the second image
 * imageSize1: cv::Size representing the size of the first image
 * imageSize2: cv::Size representing the size of the second image
 * seedPoints: vector of ScoredMatch to store the selected seed points
 * neighborhoods: std::vector<std::set<int>> to store the neighborhoods corresponding to seedPoints
 * matches: vector of ScoredMatch to store all matches found after affine verification
 *
 * Return:
 * None (outputs are via reference parameters)
 */
void GaLAM::localAffineVerification(
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2,
    const cv::Mat &descriptors1,
    const cv::Mat &descriptors2,
    const cv::Size &imageSize1,
    const cv::Size &imageSize2,
    std::vector<ScoredMatch> &seedPoints,
    std::vector<std::set<int>> &neighborhoods,
    std::vector<ScoredMatch> &matches) const
{
    // Get seed points
    seedPoints = selectSeedPoints(matches, keypoints1, descriptors1, descriptors2, imageSize1);

    // Get neighborhoods for each seed point
    neighborhoods = localNeighborhoodSelection(matches, seedPoints, keypoints1, keypoints2, imageSize1, imageSize2);

    // Perform affine verification
    affineVerification(matches, seedPoints, keypoints1, keypoints2, neighborhoods, imageSize1, imageSize2);
}

// detectOutliers
// Main function to detect outliers using GaLAM algorithm
// Preconditions: Valid parameters are provided
// Postconditions: returns StageResults containing seedMatches, stage1Matches, and finalMatches
/*
 * Parameters:
 * keypoints1: vector of cv::KeyPoint containing the keypoints from the first image
 * keypoints2: vector of cv::KeyPoint containing the keypoints from the second image
 * descriptors1: cv::Mat containing the descriptors from the first image
 * descriptors2: cv::Mat containing the descriptors from the second image
 * imageSize1: cv::Size representing the size of the first image
 * imageSize2: cv::Size representing the size of the second image
 *
 * Return:
 * GaLAM::StageResults containing seedMatches, stage1Matches, and finalMatches
 */
GaLAM::StageResults GaLAM::detectOutliers(
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2,
    const cv::Mat &descriptors1,
    const cv::Mat &descriptors2,
    const std::vector<cv::DMatch> &candidateMatches,
    const cv::Size &imageSize1,
    const cv::Size &imageSize2) const
{
    // Initialize results
    StageResults results;
    std::cout << "GaLAM: Processing " << candidateMatches.size()
              << " candidate matches..." << std::endl;

    // Stage 1: Local affine verification
    std::vector<ScoredMatch> seedPoints;
    std::vector<std::set<int>> neighborhoods;
    std::vector<ScoredMatch> matches;

    // Perform local affine verification
    localAffineVerification(keypoints1, keypoints2, descriptors1, descriptors2,
                            imageSize1, imageSize2, seedPoints, neighborhoods, matches);

    // Store seed matches for visualization
    for (const auto &seed : seedPoints)
    {
        results.seedMatches.push_back(seed.match);
    }

    std::cout << "GaLAM: Built " << neighborhoods.size()
              << " local neighborhoods" << std::endl;

    // Collect all unique inlier indices from neighborhoods
    std::set<int> inlierIndices;
    for (size_t s = 0; s < neighborhoods.size(); ++s)
    {
        // Add all indices from this neighborhood
        for (int idx : neighborhoods[s])
        {
            inlierIndices.insert(idx);
        }
    }

    std::cout << "GaLAM: Stage 1 produced " << inlierIndices.size()
              << " unique inlier candidates" << std::endl;

    // Convert inlier indices to cv::DMatch
    for (int idx : inlierIndices)
    {
        results.stage1Matches.push_back(matches[idx].match);
    }

    std::cout << "GaLAM: Returning " << results.stage1Matches.size()
              << " matches after Stage 1 (Local Affine Matching)" << std::endl;

    // Initialize final matches with Stage 1 results
    results.finalMatches = results.stage1Matches;

    // Stage 2: Global geometric consistency
    results.finalMatches = globalGeometryVerification(
        matches, seedPoints, keypoints1, keypoints2, neighborhoods);

    // If Stage 2 returns empty, fallback to Stage 1 results
    if (results.finalMatches.empty())
    {
        std::cout << "GaLAM: Stage 2 returned no matches, using Stage 1 results" << std::endl;
        results.finalMatches = results.stage1Matches;
    }

    std::cout << "GaLAM: Final matches after Stage 2: " << results.finalMatches.size() << std::endl;

    // Final RT-based thresholding
    // If the number of matches exceeds 200, filter matches using
    // a ratio test (RT < 0.9).
    if (results.finalMatches.size() > 200)
    {
        const double finalRTThreshold = params_.finalRTThreshold; // 0.9

        std::vector<cv::DMatch> filtered;
        filtered.reserve(results.finalMatches.size());

        // debuging logs
        int totalChecked = 0;
        int totalFound = 0;
        int totalBad = 0;

        // For each final match, recover its ratio test value from the original matches
        for (const auto &dm : results.finalMatches)
        {
            ++totalChecked;

            bool found = false;
            double ratio = 0.0;

            // Recover the ratio test using original matches vector
            for (const auto &sm : matches)
            {
                if (sm.match.queryIdx == dm.queryIdx &&
                    sm.match.trainIdx == dm.trainIdx)
                {
                    double dist1 = std::max((double)sm.match.distance, 1e-6);
                    double dist2 = std::max((double)sm.secondMatch.distance, 1e-6);
                    ratio = dist1 / dist2;

                    found = true;
                    ++totalFound;

                    if (ratio >= finalRTThreshold)
                    {
                        ++totalBad; // This one would be removed
                    }

                    break;
                }
            }

            // If no ScoredMatch was found, keep the match to preserve original logic
            if (!found)
            {
                filtered.push_back(dm);
                continue;
            }

            // Keep only matches that pass RT threshold
            if (ratio < finalRTThreshold)
            {
                filtered.push_back(dm);
            }
        }

        // Debug output for verification
        std::cout << "RT filter stats: checked=" << totalChecked
                  << ", found=" << totalFound
                  << ", bad=" << totalBad << std::endl;

        // Report filtering results
        std::cout << "GaLAM: Final RT filter (" << finalRTThreshold
                  << ") reduced matches from "
                  << results.finalMatches.size() << " to "
                  << filtered.size() << std::endl;

        // Update final matches
        results.finalMatches.swap(filtered);
    }

    return results;
}