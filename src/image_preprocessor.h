#ifndef IMAGE_PREPROCESSOR_H_
#define IMAGE_PREPROCESSOR_H_

#include "image_provider.h"
#include <memory>
#include <vector>

struct ChannelNormParams
{
    float mean;
    float std;
};

class ImageTransformation
{
    public:
    virtual void apply(Image& image) const = 0;
    virtual ~ImageTransformation() = default;
};

class ResizeTransformation final : public ImageTransformation
{
    public:
    ResizeTransformation(int height, int width) :  height_{height}, width_{width} {}
    void apply(Image& image) const override
    {
        resize(image, height_, width_);
    }

    private:
    void resize(Image& image, int height, int width) const;
    int height_;
    int width_;
};

class ConvertColorTransformation final : public ImageTransformation
{
    public:
    ConvertColorTransformation(ColorFormat colorFmt) : colorFmt_{colorFmt} {}
    void apply(Image& image) const override
    {
        switch (colorFmt_)
        {
            case ColorFormat::RGB:
            {
                convertRGB(image, colorFmt_);
                break;
            }
            case ColorFormat::BGR:
            default:
            {
                // do nothing
            }
        }
    }

    private:
    void convertRGB(Image& image, ColorFormat colorFmt) const;
    ColorFormat colorFmt_;
};

class ConvertMemoryLayoutTransformation final : public ImageTransformation
{
    public:
    ConvertMemoryLayoutTransformation(MemoryLayout memLayout) : memLayout_{memLayout} {}
    void apply(Image& image) const override
    {
        convertCHW(image, memLayout_);
    }

    private:
    void convertCHW(Image& image, MemoryLayout memLayout_) const;
    MemoryLayout memLayout_;
};

class NormalizeTransformation final : public ImageTransformation
{
    public:
    NormalizeTransformation(const std::array<ChannelNormParams, 3>& normParams) : normParams_{normParams} {}
    void apply(Image& image) const override
    {
        normalize(image, normParams_);
    }

    private:
    void normalize(Image& image, const std::array<ChannelNormParams, 3>& normParams) const;
    const std::array<ChannelNormParams, 3>& normParams_;
};


class ImagePreprocessingPipeline
{
    public:
    void add(std::unique_ptr<ImageTransformation> imageTransformation)
    {
        imageTransformations_.push_back(std::move(imageTransformation));
    }
    void apply(Image& image)
    {
        for (const auto& transform : imageTransformations_)
        {
            transform->apply(image);
        }
    }

    private:
    std::vector<std::unique_ptr<ImageTransformation>> imageTransformations_;
};

class ImagePreprocessingPipelineBuilder
{
    public:
    ImagePreprocessingPipelineBuilder() : pipeline_{std::make_unique<ImagePreprocessingPipeline>()} {}
    void addResize(int height, int width)
    {
        pipeline_->add(std::make_unique<ResizeTransformation>(height, width));
    }
    void addConvertColor(ColorFormat colorFmt)
    {
        pipeline_->add(std::make_unique<ConvertColorTransformation>(colorFmt));
    }
    void addConvertMemLayout(MemoryLayout memLayout)
    {
        pipeline_->add(std::make_unique<ConvertMemoryLayoutTransformation>(memLayout));
    }

    void addNormalize(const std::array<ChannelNormParams, 3>& normParams)
    {
        pipeline_->add(std::make_unique<NormalizeTransformation>(normParams));
    }

    std::unique_ptr<ImagePreprocessingPipeline> build()
    {
        return std::move(pipeline_);
    }

    private:
    std::unique_ptr<ImagePreprocessingPipeline> pipeline_;
};

#endif // #ifndef IMAGE_PREPROCESSOR_H_