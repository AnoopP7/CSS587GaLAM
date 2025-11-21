#ifndef GALAM_H
#define GALAM_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>

//namespace galam {

class GaLAM {
public:
    struct InputParameters {
        double ratio;
        double rt_threshold;
        double epsilon;
        InputParameters()
        : ratio(100.0),
          rt_threshold(0.8),
          epsilon(1e-6)
    {}
    };

    explicit GaLAM(const InputParameters& params = InputParameters());

    // Main detection pipeline (stub for now
    std::vector<cv::DMatch> detectOutliers(
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2,
        const std::vector<cv::DMatch>& candidateMatches,
        const cv::Size& imageSize1,
        const cv::Size& imageSize2
    ) const;

private:
    InputParameters params_;
    double radius1, radius2;

    struct ScoredMatch {
        cv::DMatch match;
        double confidence;
    };

    // Seed point selection
    std::vector<ScoredMatch> selectSeedPoints(
        const std::vector<ScoredMatch>& matches,
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

    // Local neighborhood selection
    std::vector<ScoredMatch> localNeighborhoodSelection(
        const std::vector<ScoredMatch>& matches,
        const std::vector<ScoredMatch>& seedPoints,
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const cv::Size& imageSize1,
        const cv::Size& imageSize2
    ) const;

    std::vector<ScoredMatch> filterByDistance(
        const std::vector<ScoredMatch>& matches,
        const std::vector<ScoredMatch>& seedPoints,
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2
    ) const;

    std::vector<ScoredMatch> filterByScaleRotation(
        const std::vector<ScoredMatch>& matches,
        const std::vector<ScoredMatch>& seedPoints,
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2
    ) const;

    // For now, proceeding under the assumption that R1 is the same for all seed points and
    // calculated the same way as R in previous step but R2 is not

    // Actually assuming that R1 and R2 are arbitrary, and different for each correspondence
    std::vector<GaLAM::ScoredMatch> filterByImageScale(
        const std::vector<ScoredMatch>& matches,
        const std::vector<ScoredMatch>& seedPoints,
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const cv::Size& imageSize1,
        const cv::Size& imageSize2
    ) const;
};

//} // namespace galam

#endif // GALAM_H
