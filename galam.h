#ifndef GALAM_H
#define GALAM_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>

namespace galam {

class GaLAM {
public:
    struct InputParameters {
        double ra;
        double rt_threshold;
        double radius;
        double epsilon;
        InputParameters()
        : ra(100.0),
          rt_threshold(0.8),
          radius(50.0),
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

    struct ScoredMatch {
        cv::DMatch match;
        double confidence;
    };

    // Seed point selection
    std::vector<ScoredMatch> filterBidirectionalNN(
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2
    ) const;

    void assignConfidenceScore(
        std::vector<ScoredMatch>& matches
    ) const;

    std::vector<ScoredMatch> selectSeedPoints(
        const std::vector<ScoredMatch>& matches,
        const std::vector<cv::KeyPoint>& keypoints1,
        const cv::Size& imageSize1
    ) const;
};

} // namespace galam

#endif // GALAM_H
