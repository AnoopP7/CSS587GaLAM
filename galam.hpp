#ifndef GALAM_H
#define GALAM_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace galam {

class GaLAM {
public:
    struct Parameters {
        // TODO: add tunable parameters here (thresholds, RANSAC settings)
    };

    explicit GaLAM(const Parameters& params = Parameters());

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
    Parameters params_;
};

} // namespace galam

#endif // GALAM_H
