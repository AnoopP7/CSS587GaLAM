#ifndef GALAM_H
#define GALAM_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
#include <set>

//namespace galam {

class GaLAM {
public:
    struct InputParameters {
        double ratio;
        double rt_threshold;
        double epsilon;
        double lambda1, lambda2, lambda3;
        double tAlpha;// max rotation difference 
        double tSigma;// max scale difference 
        int num_iterations; // iterations
        int minSampleSize; // 8-points algorithm (8 pairs)
        InputParameters()
        : ratio(100.0),
          rt_threshold(0.8),
          epsilon(1e-6),
          // lamba could be different values
          lambda1(4.0), lambda2(2.0), lambda3(0.8),
          tAlpha(10.0),
          tSigma(0.5),
          num_iterations(128),
          minSampleSize(8)
    {}
    };

    explicit GaLAM(const InputParameters& params = InputParameters());

    struct StageResults {
        std::vector<cv::DMatch> seedMatches;
        std::vector<cv::DMatch> stage1Matches;
        std::vector<cv::DMatch> finalMatches;
    };

    // Main detection pipeline (stub for now
    StageResults detectOutliers(
        std::vector<cv::KeyPoint>& keypoints1,
        std::vector<cv::KeyPoint>& keypoints2,
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2,
        const std::vector<cv::DMatch>& candidateMatches,
        const cv::Size& imageSize1,
        const cv::Size& imageSize2
    ) const;

private:
    InputParameters params_;
    double radius1, radius2;

    // TODO: If time, refactor into an object so that we can easily have get/setX and get/setY
    struct ScoredMatch {
        cv::DMatch match;
        cv::DMatch secondMatch;
        double confidence;
    };

    // Seed point selection
    std::vector<ScoredMatch> selectSeedPoints(
        std::vector<ScoredMatch>& matches,
        const std::vector<cv::KeyPoint>& keypoints1,
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2,
        const cv::Size& imageSize1
    ) const;

    std::vector<ScoredMatch> filterBidirectionalNN(
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2
    ) const;

    void assignConfidenceScore(
        std::vector<ScoredMatch>& matches
    ) const;

    std::vector<ScoredMatch> selectPoints(
        const std::vector<ScoredMatch>& matches,
        const std::vector<cv::KeyPoint>& keypoints1,
        const cv::Size& imageSize1
    ) const;

    // Helper function for Local Neighborhood Selection
    double computeBaseRadius(const cv::Size& imageSize) const;

    // Local neighborhood selection
    std::vector<std::set<int>> localNeighborhoodSelection(
        const std::vector<ScoredMatch>& matches,
        const std::vector<ScoredMatch>& seedPoints,
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const cv::Size& imageSize1,
        const cv::Size& imageSize2
    ) const;

    void affineVerification(
        std::vector<ScoredMatch>& matches,
        std::vector<ScoredMatch>& seedPoints,
        std::vector<cv::KeyPoint>& keypoints1,
        std::vector<cv::KeyPoint>& keypoints2,
        std::vector<std::set<int>>& neighborhoods,
        const cv::Size& imageSize1,
        const cv::Size& imageSize2
    ) const;

    void preprocessSets(
        const std::vector<ScoredMatch>& matches,
        const std::vector<ScoredMatch>& seedPoints,
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        std::vector<std::set<int>>& neighborhoods,
        const cv::Size& imageSize1,
        const cv::Size& imageSize2,
        std::vector<cv::Point2f>& normalizedKeypoints1,
        std::vector<cv::Point2f>& normalizedKeypoints2
    ) const;

    std::vector<cv::Mat> fitTransformationMatrix(
        std::vector<ScoredMatch>& matches,
        std::vector<ScoredMatch>& seedPoints,
        std::vector<cv::KeyPoint>& keypoints1,
        std::vector<cv::KeyPoint>& keypoints2,
        std::vector<std::set<int>>& neighborhoods,
        const cv::Size& imageSize2,
        std::vector<cv::Point2f>& normalizedKeypoints1,
        std::vector<cv::Point2f>& normalizedKeypoints2
    ) const;

    double measureAffineResidual(
        const cv::Mat& transformation,
        const ScoredMatch& correspondence,
        const std::vector<cv::Point2f>& normalizedKeypoints1,
        const std::vector<cv::Point2f>& normalizedKeypoints2
    ) const;

    void localAffineVerification(
        std::vector<cv::KeyPoint>& keypoints1,
        std::vector<cv::KeyPoint>& keypoints2,
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2,
        const cv::Size& imageSize1,
        const cv::Size& imageSize2,
        std::vector<ScoredMatch>& seedPoints,
        std::vector<std::set<int>>& neighborhoods,
        std::vector<ScoredMatch>& matches
    ) const;

    // Stage 2
    std::vector<cv::DMatch> globalGeometryVerification(
        const std::vector<ScoredMatch>& matches, 
        const std::vector<ScoredMatch>& seedPoints, 
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const std::vector<std::set<int>>& neighborhoods
    ) const;
    //void filterByAffineResidual(
    //    const std::vector<ScoredMatch>& matches,
    //    const std::vector<ScoredMatch>& seedPoints,
    //    std::vector<std::set<int>>& neighborhoods,
    //    std::vector<cv::Mat> transformations
    //) const;

    /*cv::Mat sampleSeedPoints(
        const std::vector<ScoredMatch>& matches,
        const std::vector<ScoredMatch>& seedPoints,
        std::vector<std::set<int>>& neighborhoods
    ) const;*/



    // std::vector<ScoredMatch> filterByDistance(
    //     const std::vector<ScoredMatch>& matches,
    //     const std::vector<ScoredMatch>& seedPoints,
    //     const std::vector<cv::KeyPoint>& keypoints1,
    //     const std::vector<cv::KeyPoint>& keypoints2,
    //     const double radius1,
    //     const double radius2
    // ) const;

    // std::vector<ScoredMatch> filterByScaleRotation(
    //     const std::vector<ScoredMatch>& matches,
    //     const std::vector<ScoredMatch>& seedPoints,
    //     const std::vector<cv::KeyPoint>& keypoints1,
    //     const std::vector<cv::KeyPoint>& keypoints2
    // ) const;

    // For now, proceeding under the assumption that R1 is the same for all seed points and
    // calculated the same way as R in previous step but R2 is not

    // Actually assuming that R1 and R2 are arbitrary, and different for each correspondence
    // std::vector<GaLAM::ScoredMatch> filterByImageScale(
    //     const std::vector<ScoredMatch>& matches,
    //     const std::vector<ScoredMatch>& seedPoints,
    //     const std::vector<cv::KeyPoint>& keypoints1,
    //     const std::vector<cv::KeyPoint>& keypoints2,
    //     const cv::Size& imageSize1,
    //     const cv::Size& imageSize2,
    //     const double radius1,
    //     const double radius2
    // ) const;
};

//} // namespace galam

#endif // GALAM_H
