#include "image_preprocessor.h"
#include <opencv2/dnn/dnn.hpp>
#include <cmath>
#include <algorithm>

void ResizeTransformation::resize(Image& image, int height, int width) const
{
    if ((height > image.height && width < image.width) || (height < image.height && width > image.width))
    {
        std::cerr << "Shrinking/Enlarging per dimension is not supported!\n";

        std::exit(EXIT_FAILURE);
    }

    if (height == image.height && width == image.width)
    {
        return;
    }
    else if (height > image.height)
    {
        int delta_height = height - image.height;
        int delta_width = width - image.width;

        int borderTop = static_cast<int>(std::round(delta_height / 2.0f));
        int borderBottom = delta_height - borderTop;

        int borderLeft = static_cast<int>(std::round(delta_width / 2.0f));
        int borderRight = delta_width - borderLeft;

        cv::copyMakeBorder(image.matrix, image.matrix, borderTop, borderBottom, borderLeft, borderRight, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar{0, 0, 0});
    }
    else // Shrink
    {
        cv::resize(image.matrix, image.matrix, cv::Size(width, height), cv::InterpolationFlags::INTER_AREA);
    }

    image.height = image.matrix.rows;
    image.width = image.matrix.cols;
}

void ConvertColorTransformation::convertRGB(Image& image, ColorFormat colorFmt) const
{
    if (image.fmt == ColorFormat::BGR)
    {
        cv::cvtColor(image.matrix, image.matrix, cv::ColorConversionCodes::COLOR_BGR2RGB);
        image.fmt = ColorFormat::RGB;
    }
}

void ConvertMemoryLayoutTransformation::convertCHW(Image& image, MemoryLayout memLayout_) const
{
    if (image.layout == MemoryLayout::HWC)
    {
        cv::dnn::blobFromImage(image.matrix, image.matrix);
        image.layout = MemoryLayout::CHW;
    }
}

void NormalizeTransformation::normalize(Image& image, const std::array<ChannelNormParams, 3>& normParams) const
{
    image.matrix.convertTo(image.matrix, CV_32F, 1.0 / 255);

    std::array<cv::Mat, 3> imageChannels;

    cv::split(image.matrix, imageChannels);

    auto norm_fun = [](const cv::Mat& channel, const ChannelNormParams& normParams)
    {
        return (channel - normParams.mean) / normParams.std;
    };

    std::transform(imageChannels.cbegin(), imageChannels.cend(), normParams.cbegin(), imageChannels.begin(), norm_fun);
    
    cv::merge(imageChannels.data(), 3, image.matrix);
}