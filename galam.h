/*
 * galam.h
 * Implements the GaLAM outlier detection algorithm in C++
 * Implementation authors: Yu Dinh, Neha Kotwal, Anoop Prasad
 * Paper title: GaLAM: Two-Stage Outlier Detection Algorithm
 * Paper authors: X. Lu, Z. Yan, Z. Fan
 *
 *  * This file contains the implementation of the GaLAM class methods.
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
 *    - Return the final set of inlier matches after both stages
 *
 * Note:
 * - This implementation uses OpenCV for image processing and feature matching
 * - The code is structured to allow easy modification of parameters and integration
 * with other systems
 * - Error handling is implemented to manage potential issues during processing
 *
 * TODO: Implement OpenCV outlier detection interface if possible
 */

#ifndef GALAM_H
#define GALAM_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
#include <set>

// namespace galam {

// GaLAM class
// Main class implementing the GaLAM outlier detection algorithm
class GaLAM
{
public:
    // InputParameters struct
    // Struct to hold all input parameters for the GaLAM algorithm
    // Default values are set in the constructor
    // Precondition: None
    // Postcondition: Initializes InputParameters with default values
    struct InputParameters
    {
        double ratio;                     // ratio for NMS area
        double rt_threshold;              // ratio test threshold
        double epsilon;                   // epipolar threshold
        double lambda1, lambda2, lambda3; // lambdas from paper
        double tAlpha;                    // max rotation difference
        double tSigma;                    // max scale difference
        int num_iterations;               // iterations
        int minSampleSize;                // 8-points algorithm (8 pairs)
        InputParameters()
            : ratio(100.0),
              rt_threshold(0.8),
              epsilon(1.0),
              // lambdas could be different values
              lambda1(4.0), lambda2(2.0), lambda3(0.8),
              // thresholds
              tAlpha(20.0),
              tSigma(0.5),
              // RANSAC thresholds
              num_iterations(128),
              minSampleSize(8)
        {
        }
    };

    // Constructor
    // Constructs a GaLAM object based on the given InputParameters struct
    // Preconditions: params includes valid parameters for GaLAM algorithm
    // Postconditions: Initializes GaLAM object using the params
    /**
     * Parameter:
     * InputParameters struct containing parameters to use for the algorithm
     *
     * Return:
     * GaLAM object
     */
    explicit GaLAM(const InputParameters &params = InputParameters());

    // StageResults struct
    // Struct to hold the results of each stage of the GaLAM algorithms
    struct StageResults
    {
        std::vector<cv::DMatch> seedMatches;
        std::vector<cv::DMatch> stage1Matches;
        std::vector<cv::DMatch> finalMatches;
    };

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
     * Return:
     * GaLAM::StageResults containing seedMatches, stage1Matches, and finalMatches
     */
    StageResults detectOutliers(
        std::vector<cv::KeyPoint> &keypoints1,
        std::vector<cv::KeyPoint> &keypoints2,
        const cv::Mat &descriptors1,
        const cv::Mat &descriptors2,
        const std::vector<cv::DMatch> &candidateMatches,
        const cv::Size &imageSize1,
        const cv::Size &imageSize2) const;

private:
    InputParameters params_;
    double radius1, radius2;

    // TODO: If time, refactor into an object so that we can easily have get/setX and get/setY
    struct ScoredMatch
    {
        cv::DMatch match;
        cv::DMatch secondMatch;
        double confidence;
    };

    // selectSeedPoints
    // Seed point selection
    // Select seed points from initial matches
    // using bidirectional NN + ratio test + non-maximum suppression
    // Returns selected seed points
    // Precondition: matches is non-empty
    // Postcondition: matches is filtered to only include bidirectional NN matches
    /*
     * Parameters:
     * matches: vector of ScoredMatch to store all matches found
     * keypoints1: vector of cv::KeyPoint containing the keypoints from the first image
     * imageSize1: cv::Size representing the size of the first image
     *
     * Return:
     * std::vector<GaLAM::ScoredMatch> of seed points found
     */
    std::vector<ScoredMatch> selectSeedPoints(
        std::vector<ScoredMatch> &matches,
        const std::vector<cv::KeyPoint> &keypoints1,
        const cv::Mat &descriptors1,
        const cv::Mat &descriptors2,
        const cv::Size &imageSize1) const;

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
     * std::vector<ScoredMatch> of reliable matches after bidirectional NN
     */
    std::vector<ScoredMatch> filterBidirectionalNN(
        const cv::Mat &descriptors1,
        const cv::Mat &descriptors2) const;

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
    void assignConfidenceScore(
        std::vector<ScoredMatch> &matches) const;

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
    std::vector<ScoredMatch> selectPoints(
        const std::vector<ScoredMatch> &matches,
        const std::vector<cv::KeyPoint> &keypoints1,
        const cv::Size &imageSize1) const;

    // computeBaseRadius
    // Helper function for Local Neighborhood Selection
    // Computes the base radius R1 based on the image size
    // Preconditions: Image size is provided
    // Postconditions: Returns the radius for the image as a double
    /*
     * Parameters:
     * imageSize: cv::Size for the image is provided
     *
     * Return:
     * double of the radius for that image
     */
    double computeBaseRadius(const cv::Size &imageSize) const;

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
     * std::vector<std::set<int>> of neighborhoods corresponding to each seed point
     */
    std::vector<std::set<int>> localNeighborhoodSelection(
        const std::vector<ScoredMatch> &matches,
        const std::vector<ScoredMatch> &seedPoints,
        const std::vector<cv::KeyPoint> &keypoints1,
        const std::vector<cv::KeyPoint> &keypoints2,
        const cv::Size &imageSize1,
        const cv::Size &imageSize2) const;

    // affineVerification
    // Performs affine verification for each neighborhood and filters out outliers
    // Preconditions: Valid parameters are provided
    // Postconditions: matches and neighborhoods are updated to only include inliers
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
     * None (outputs are via reference parameters)
     */
    void affineVerification(
        std::vector<ScoredMatch> &matches,
        std::vector<ScoredMatch> &seedPoints,
        std::vector<cv::KeyPoint> &keypoints1,
        std::vector<cv::KeyPoint> &keypoints2,
        std::vector<std::set<int>> &neighborhoods,
        const cv::Size &imageSize1,
        const cv::Size &imageSize2) const;

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
     *
     * Return:
     * None (outputs are via reference parameters)
     */
    void preprocessSets(
        const std::vector<ScoredMatch> &matches,
        const std::vector<ScoredMatch> &seedPoints,
        const std::vector<cv::KeyPoint> &keypoints1,
        const std::vector<cv::KeyPoint> &keypoints2,
        std::vector<std::set<int>> &neighborhoods,
        const cv::Size &imageSize1,
        const cv::Size &imageSize2,
        std::vector<cv::Point2f> &normalizedKeypoints1,
        std::vector<cv::Point2f> &normalizedKeypoints2) const;

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
     *
     * Return:
     * std::vector<cv::Mat> of affine transformations for each neighborhood
     */
    std::vector<cv::Mat> fitTransformationMatrix(
        std::vector<ScoredMatch> &matches,
        std::vector<ScoredMatch> &seedPoints,
        std::vector<cv::KeyPoint> &keypoints1,
        std::vector<cv::KeyPoint> &keypoints2,
        std::vector<std::set<int>> &neighborhoods,
        const cv::Size &imageSize1,
        const cv::Size &imageSize2,
        std::vector<cv::Point2f> &normalizedKeypoints1,
        std::vector<cv::Point2f> &normalizedKeypoints2) const;

    // measureAffineResidual
    // Measures the residual error for a given affine transformation and correspondence
    // Preconditions: Valid parameters are provided
    // Postconditions: Returns the residual as a double
    /*
     * Parameters:
     * transformation: cv::Mat of the affine transformation to be evaluated
     * correspondence: ScoredMatch representing the correspondence pair we are using for evaluation
     * normalizedKeypoints1: vector of normalized keypoints from the first image
     * normalizedKeypoints2: vector of normalized keypoints from the second image
     *
     * Return:
     * double representing the residual error
     */
    double measureAffineResidual(
        const cv::Mat &transformation,
        const ScoredMatch &correspondence,
        const std::vector<cv::Point2f> &normalizedKeypoints1,
        const std::vector<cv::Point2f> &normalizedKeypoints2) const;

    // Stage 1: Local Affine Verification
    // localAffineVerification
    // Performs local affine verification for all seed points and their neighborhoods
    // Preconditions: Valid parameters are provided
    // Postconditions: seedPoints, neighborhoods, and matches are populated after local affine verification
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
    void localAffineVerification(
        std::vector<cv::KeyPoint> &keypoints1,
        std::vector<cv::KeyPoint> &keypoints2,
        const cv::Mat &descriptors1,
        const cv::Mat &descriptors2,
        const cv::Size &imageSize1,
        const cv::Size &imageSize2,
        std::vector<ScoredMatch> &seedPoints,
        std::vector<std::set<int>> &neighborhoods,
        std::vector<ScoredMatch> &matches) const;

    // Stage 2
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
    std::vector<cv::DMatch> globalGeometryVerification(
        const std::vector<ScoredMatch> &matches,
        const std::vector<ScoredMatch> &seedPoints,
        const std::vector<cv::KeyPoint> &keypoints1,
        const std::vector<cv::KeyPoint> &keypoints2,
        const std::vector<std::set<int>> &neighborhoods) const;

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
        const cv::Size &imageSize2) const;
};

#endif // GALAM_H
