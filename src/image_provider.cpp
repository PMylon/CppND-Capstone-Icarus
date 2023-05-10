#include "image_provider.h"
#include <opencv2/dnn/dnn.hpp>
#include <cmath>
#include <algorithm>

Image ImageProvider::GetImage()
{
    const std::string& absImagePath = (*imageIt_).string();

    cv::Mat imageBGR = cv::imread(absImagePath, cv::ImreadModes::IMREAD_COLOR);
    imageIt_++;

    return Image{imageBGR, imageBGR.rows, imageBGR.cols, ColorFormat::BGR, MemoryLayout::HWC, absImagePath};
}