#include "galam.h"
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <set>

// namespace galam {

GaLAM::GaLAM(const InputParameters &params)
    : params_(params) {}

std::vector<GaLAM::ScoredMatch> GaLAM::selectSeedPoints(
    std::vector<ScoredMatch> &matches,
    const std::vector<cv::KeyPoint> &keypoints1,
    const cv::Mat &descriptors1,
    const cv::Mat &descriptors2,
    const cv::Size &imageSize1) const
{
    // 1) Bidirectional NN + ratio test
    matches = filterBidirectionalNN(descriptors1, descriptors2);
    if (matches.empty())
    {
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
    const cv::Mat &descriptors1,
    const cv::Mat &descriptors2) const
{
    std::vector<std::vector<cv::DMatch>> knn12, knn21;

    // If slow, FLANN instead
    cv::BFMatcher matcher(cv::NORM_L2);
    matcher.knnMatch(descriptors1, descriptors2, knn12, 2);
    matcher.knnMatch(descriptors2, descriptors1, knn21, 2);

    std::vector<ScoredMatch> validMatches;

    for (int i = 0; i < static_cast<int>(knn12.size()); ++i)
    {
        if (knn12[i].size() < 2)
            continue;

        const cv::DMatch &bestMatch = knn12[i][0];
        const cv::DMatch &secondBestMatch = knn12[i][1];

        if (bestMatch.distance >= params_.rt_threshold * secondBestMatch.distance)
            continue;

        int queryIdx = bestMatch.queryIdx;
        int trainIdx = bestMatch.trainIdx;

        if (trainIdx < 0 || trainIdx >= static_cast<int>(knn21.size()))
            continue;
        if (knn21[trainIdx].empty())
            continue;

        const cv::DMatch &reverseMatch = knn21[trainIdx][0];
        if (reverseMatch.trainIdx != queryIdx)
            continue;

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
    std::vector<ScoredMatch> &matches) const
{
    for (auto &scored : matches)
    {
        double distance = std::max(static_cast<double>(scored.match.distance), 1e-6);
        scored.confidence = 1.0 / distance;
    }
}

// 3) Select seed points using non-maximum suppression
// This R is global for the whole image pair.
// It is used ONLY for NMS—NOT the local neighborhood radius (R1, R2).
std::vector<GaLAM::ScoredMatch> GaLAM::selectPoints(
    const std::vector<ScoredMatch> &matches,
    const std::vector<cv::KeyPoint> &keypoints1,
    const cv::Size &imageSize1) const
{
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
              [&](int a, int b)
              {
                  return matches[a].confidence > matches[b].confidence;
              });

    std::vector<bool> isSuppressed(matches.size(), false);
    std::vector<ScoredMatch> seedPoints;
    seedPoints.reserve(matches.size() / 4);

    // ----------------------------------------------------------------------
    // Non-maximum suppression:
    // Keep the highest-confidence match, suppress all within radius R.
    // ----------------------------------------------------------------------
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

            double dx = seedPoint.x - otherPoint.x;
            double dy = seedPoint.y - otherPoint.y;
            if (std::sqrt(dx * dx + dy * dy) <= globalSeedRadius_R)
            {
                isSuppressed[j] = true;
            }
        }

        seedPoints.push_back(seedMatch);
    }

    std::cout << "GaLAM: Seed points selected = " << seedPoints.size() << std::endl;
    return seedPoints;
}

// helper: normalize angle difference (in degrees) to range [-180, 180)
static double normalizeAngle(double angle)
{
    while (angle <= -180.0)
        angle += 360.0;
    while (angle > 180.0)
        angle -= 360.0;
    return angle;
}

// Compute base radius R1 from image size
// Larger images --> larger R1
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

// Local neighborhood selection
// for each seed point, find matches in its neighborhood
// according to distance, rotation, scale constraints
// R2 is computed per-seed based on its scale ratio
std::vector<std::set<int>> GaLAM::localNeighborhoodSelection(
    const std::vector<ScoredMatch> &matches,
    const std::vector<ScoredMatch> &seedPoints,
    const std::vector<cv::KeyPoint> &keypoints1,
    const std::vector<cv::KeyPoint> &keypoints2,
    const cv::Size &imageSize1,
    const cv::Size &imageSize2) const
{
    std::vector<std::set<int>> neighborhoods;

    // early exit if no matches or no seed points
    if (matches.empty() || seedPoints.empty())
    {
        return neighborhoods;
    }

    // base radius R1 from image 1 (eq. for R1 in the paper)
    double R1 = computeBaseRadius(imageSize1);

    // allocate one neighborhood per seed
    neighborhoods.resize(seedPoints.size());

    double lambda1 = params_.lambda1; // spatial multiplier
    double tAlpha = params_.tAlpha;   // max allowed rotation difference (deg)
    double tSigma = params_.tSigma;   // max allowed log-scale difference

    for (size_t s = 0; s < seedPoints.size(); ++s)
    {
        const ScoredMatch &seed = seedPoints[s];
        const cv::KeyPoint &kp1Seed = keypoints1[seed.match.queryIdx];
        const cv::KeyPoint &kp2Seed = keypoints2[seed.match.trainIdx];

        // seed scale ratio σ_S = σ2 / σ1
        double sigma1Seed = std::max(static_cast<double>(kp1Seed.size), 1e-6);
        double sigma2Seed = std::max(static_cast<double>(kp2Seed.size), 1e-6);
        double sigmaSeed = sigma2Seed / sigma1Seed;

        // Per-seed R2 (paper: R2 = R1 / σ_S)
        double R2_seed = R1 / sigmaSeed;
        if (R2_seed <= 1e-6)
            continue; // degenerate, skip this seed

        // seed rotation α_S = angle2 − angle1 (normalized)
        double alphaSeed = normalizeAngle(kp2Seed.angle - kp1Seed.angle);

        std::set<int> &neigh = neighborhoods[s];

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

            // 1) Distance constraint (Eq. 1)

            // Image 1
            double dx1 = kp1.pt.x - kp1Seed.pt.x;
            double dy1 = kp1.pt.y - kp1Seed.pt.y;
            double d1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
            if (d1 > lambda1 * R1)
                continue;

            // Image 2 (uses per-seed R2)
            double dx2 = kp2.pt.x - kp2Seed.pt.x;
            double dy2 = kp2.pt.y - kp2Seed.pt.y;
            double d2 = std::sqrt(dx2 * dx2 + dy2 * dy2);
            if (d2 > lambda1 * R2_seed)
                continue;

            // 2) Rotation + scale in log-domain (Eq. 2)

            // rotation difference
            double alphaCand = normalizeAngle(kp2.angle - kp1.angle);
            double dAlpha = std::fabs(normalizeAngle(alphaCand - alphaSeed));
            if (dAlpha > tAlpha)
                continue;

            // scale ratio in candidate
            double sigma1C = std::max(static_cast<double>(kp1.size), 1e-6);
            double sigma2C = std::max(static_cast<double>(kp2.size), 1e-6);
            double sigmaCand = sigma2C / sigma1C;
            double ratio = sigmaCand / sigmaSeed;

            if (std::fabs(std::log(ratio)) > tSigma)
                continue;

            // (Optional Eq. 3 is omitted here; Eq. 1 + Eq. 2 already enforce locality)

            // passes all constraints → add to neighborhood
            neigh.insert(static_cast<int>(i));
        }

        std::cout << "GaLAM: Neighborhood " << s
                  << " size = " << neigh.size() << std::endl;
    }

    return neighborhoods;
}

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
    // NEW: add image sizes 1 into fitTransformationMatrix
    std::vector<cv::Mat> transformations = fitTransformationMatrix(matches, seedPoints, keypoints1, keypoints2, neighborhoods, imageSize1,
                                                                   imageSize2, normalizedKeypoints1, normalizedKeypoints2);
}

// Assuming that R1 and R2 are the same R1 and R2 from earlier and that we don't normalize the coordinates of the seed points
// New R1 from image size 1
// New R2 per-seed from its scale ratio
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
    double R1 = computeBaseRadius(imageSize1);
    // double R2 = computeBaseRadius(imageSize2);

    for (size_t s = 0; s < neighborhoods.size(); ++s)
    {
        const ScoredMatch &seedPoint = seedPoints[s];
        const cv::DMatch &seedMatch = seedPoint.match;
        int seed_q = seedMatch.queryIdx;
        int seed_t = seedMatch.trainIdx;

        const cv::KeyPoint &seedPoint1 = keypoints1[seed_q];
        const cv::KeyPoint &seedPoint2 = keypoints2[seed_t];

        // NEW: Compute per-seed R2 based on seed's scale ratio
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

// Assuming that we need TWO, not THREE and that we should remove if not
// Might be selecting one with RANSAC
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
    for (size_t i = 0; i < neighborhoods.size(); i++)
    {
        if (neighborhoods[i].size() < 2)
        {
            neighborhoods.erase(neighborhoods.begin() + i);
            seedPoints.erase(seedPoints.begin() + i);
            --i; // decrement index if we removed an item
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
    // double R2 = computeBaseRadius(imageSize2);
    // double threshold = params_.lambda2 / (params_.lambda1 * R2);
    double R1 = computeBaseRadius(imageSize1);

    // Iterate through each seed point's neighborhood
    for (size_t neighborhood = 0; neighborhood < neighborhoods.size(); neighborhood++)
    {
        // NEW: Compute per-seed R2 for threshold calculation
        const ScoredMatch &seedPoint = seedPoints[neighborhood];
        const cv::KeyPoint &kp1Seed = keypoints1[seedPoint.match.queryIdx];
        const cv::KeyPoint &kp2Seed = keypoints2[seedPoint.match.trainIdx];

        double sigma1Seed = std::max(static_cast<double>(kp1Seed.size), 1e-6);
        double sigma2Seed = std::max(static_cast<double>(kp2Seed.size), 1e-6);
        double sigmaSeed = sigma2Seed / sigma1Seed;
        double R2_seed = R1 / sigmaSeed;

        double threshold = params_.lambda2 / (params_.lambda1 * R2_seed);

        // Build the vectors to use for fitting
        std::vector<cv::Point2f> points1;
        std::vector<cv::Point2f> points2;

        for (int match : neighborhoods[neighborhood])
        {
            points1.push_back(normalizedKeypoints1[matches[match].match.queryIdx]);
            points2.push_back(normalizedKeypoints2[matches[match].match.trainIdx]);
            // points1.push_back(keypoints1[matches[match].match.queryIdx].pt);
            // points2.push_back(keypoints2[matches[match].match.trainIdx].pt);
        }

        // Use RANSAC to fit the affine transformation matrix
        // TODO: Verify that this approach is consistent with the paper
        cv::Mat optimalTransformation;
        int bestScore = -1;

        for (int j = 0; j < params_.num_iterations; j++)
        {
            cv::Mat transformation = cv::estimateAffinePartial2D(points1, points2, cv::noArray(), cv::RANSAC, 3, 1, 0.99, 10);

            // Find the residual rk for each correspondence point pair in the neighborhood
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

        // Keep best affine transformation
        transforms.push_back(optimalTransformation);

        // Candidate correspondences with residuals below threshold are kept, others removed
        // need to iterate through matches, check if their residual was less than threshold, and keep if so
        std::vector<int> toRemove;
        for (int match : neighborhoods[neighborhood])
        {
            double rk = measureAffineResidual(optimalTransformation, matches[match], normalizedKeypoints1, normalizedKeypoints2);

            // Compare rk against threshold and remove the match if above threshold
            if (rk > threshold)
            {
                toRemove.push_back(match); // TODO: Make sure this actually erases the correct element
            }
        }
        for (int idx : toRemove)
        {
            neighborhoods[neighborhood].erase(idx);
        }
    }

    return transforms;
}

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
    // TODO: Do we need to transpose?
    cv::Mat_<double> matPoint1(3, 1);
    matPoint1(0, 0) = point1.x;
    matPoint1(1, 0) = point1.y;
    matPoint1(2, 0) = 1.0;

    cv::Mat_<double> matPoint2(2, 1);
    matPoint2(0, 0) = point2.x;
    matPoint2(1, 0) = point2.y;

    // Compute matrix multiplication
    cv::Mat result = (transformation * matPoint1) - matPoint2;

    // Get norm of resulting vector
    // Should this be L2 norm or something else?
    double residual = cv::norm(result, cv::NORM_L2);

    return residual;
}

// Stage 2
std::vector<cv::DMatch> GaLAM::globalGeometryVerification(
    const std::vector<ScoredMatch> &matches,
    const std::vector<ScoredMatch> &seedPoints,
    const std::vector<cv::KeyPoint> &keypoints1,
    const std::vector<cv::KeyPoint> &keypoints2,
    const std::vector<std::set<int>> &neighborhoods) const
{
    const int numSeeds = static_cast<int>(seedPoints.size());

    int minSampleSize = params_.minSampleSize;   // 8
    int num_iterations = params_.num_iterations; // 128
    double epsilon = params_.epsilon;            // epipolar threshold
    double lambda3 = params_.lambda3;            // lambda3 from paper

    // Basic sanity checks
    if (numSeeds == 0)
    {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): No seeds provided." << std::endl;
        return {};
    }

    if (matches.empty())
    {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): No matches provided." << std::endl;
        return {};
    }

    if (static_cast<int>(neighborhoods.size()) != numSeeds)
    {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): neighborhoods.size() ("
                  << neighborhoods.size() << ") does not match seedPoints.size() ("
                  << numSeeds << ")." << std::endl;
        return {};
    }

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
    modelSupports.reserve(num_iterations);
    modelSeedInliers.reserve(num_iterations);

    cv::RNG rng((uint64)cv::getTickCount());

    // --- RANSAC over seeds ---
    for (int iter = 0; iter < num_iterations; ++iter)
    {
        std::vector<int> sampleSeedIndices;
        sampleSeedIndices.reserve(minSampleSize);
        std::vector<bool> used(numSeeds, false);

        int attempts = 0;
        while (static_cast<int>(sampleSeedIndices.size()) < minSampleSize &&
               attempts < 100 * minSampleSize)
        {
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

        for (int seedIdx : sampleSeedIndices)
        {
            const ScoredMatch &sm = seedPoints[seedIdx];
            const cv::KeyPoint &kp1 = keypoints1[sm.match.queryIdx];
            const cv::KeyPoint &kp2 = keypoints2[sm.match.trainIdx];
            pts1.push_back(kp1.pt);
            pts2.push_back(kp2.pt);
        }

        cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_8POINT);
        if (F.empty())
            continue;

        cv::Matx33d Fm;
        F.convertTo(Fm, CV_64F);

        std::vector<bool> seedInlier(numSeeds, false);
        int supportCount = 0;

        for (int i = 0; i < numSeeds; ++i)
        {
            const ScoredMatch &sm = seedPoints[i];
            const cv::KeyPoint &kp1 = keypoints1[sm.match.queryIdx];
            const cv::KeyPoint &kp2 = keypoints2[sm.match.trainIdx];

            cv::Vec3d x(kp1.pt.x, kp1.pt.y, 1.0);
            cv::Vec3d xp(kp2.pt.x, kp2.pt.y, 1.0);

            cv::Vec3d l = Fm * x;
            double a = l[0], b = l[1], c = l[2];

            double denom = std::sqrt(a * a + b * b);
            if (denom < 1e-12)
                continue;

            double num = std::fabs(a * xp[0] + b * xp[1] + c);
            double r = num / denom;

            if (r <= epsilon)
            {
                seedInlier[i] = true;
                ++supportCount;
            }
        }

        if (supportCount == 0)
            continue;

        modelSupports.push_back(supportCount);
        modelSeedInliers.push_back(std::move(seedInlier));
    }

    // --- After RANSAC: select strong models and seeds ---

    if (modelSupports.empty())
    {
        std::cerr << "[ERROR] GaLAM Stage 2 (RANSAC): No valid fundamental "
                  << "matrix models were generated." << std::endl;
        return {};
    }

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
            const auto &seedInlier = modelSeedInliers[m];
            for (int i = 0; i < numSeeds; ++i)
            {
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
        for (size_t m = 1; m < modelSupports.size(); ++m)
        {
            if (modelSupports[m] > modelSupports[bestModelIdx])
                bestModelIdx = static_cast<int>(m);
        }

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

        const auto &neigh = neighborhoods[i];
        for (int idx : neigh)
        {
            if (idx >= 0 && idx < static_cast<int>(matches.size()))
                isGoodMatch[idx] = true;
        }
    }

    int keptSeeds = 0;
    for (int i = 0; i < numSeeds; ++i)
        if (keepSeed[i])
            ++keptSeeds;

    std::cout << "GaLAM: Stage 2: kept " << keptSeeds << " / " << numSeeds << " seeds" << std::endl;

    std::vector<cv::DMatch> finalMatches;
    finalMatches.reserve(matches.size());
    for (size_t i = 0; i < matches.size(); ++i)
    {
        if (isGoodMatch[i])
            finalMatches.push_back(matches[i].match);
    }

    std::cout << "GaLAM: Stage 2: returning " << finalMatches.size()
              << " matches after global geometry detection." << std::endl;

    return finalMatches;
}

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

GaLAM::StageResults GaLAM::detectOutliers(
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2,
    const cv::Mat &descriptors1,
    const cv::Mat &descriptors2,
    const std::vector<cv::DMatch> &candidateMatches,
    const cv::Size &imageSize1,
    const cv::Size &imageSize2) const
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

    results.finalMatches = results.stage1Matches;

    // TODO: Stage 2 - Global geometric consistency
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

    return results;
}
