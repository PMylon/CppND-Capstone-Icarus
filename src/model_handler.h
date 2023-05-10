#ifndef MODEL_HANDLER_H_
#define MODEL_HANDLER_H_

#include "image_preprocessor.h"
#include <array>
#include <memory>
#include <fstream>
#include <string>

class ModelHandler
{
    public:
    virtual const char* const getModelPath() const = 0;
    virtual std::vector<float>& getInputBuffer() = 0;
    virtual std::vector<float>& getOutputBuffer() = 0;
    virtual int64_t getInputHeight() const = 0;
    virtual int64_t getInputWidth() const = 0;
    virtual int64_t getInputChannels() const = 0;
    virtual int64_t getInputBatches() const = 0;
    virtual int64_t getNrOfClasses() const = 0;
    virtual const char* const getLabels() const = 0;
    virtual std::string extractClassLabel(const std::string& labelStr) const = 0;
    virtual void BuildPreprocessPipeline() = 0;
    void Preprocess(Image& img)
    {
        if (preprocessingPipeline_ != nullptr)
        {
            preprocessingPipeline_->apply(img);
        }
    }
    virtual std::string Postprocess();
    virtual ~ModelHandler() = default;

    protected:
    std::unique_ptr<ImagePreprocessingPipeline> preprocessingPipeline_;
};

class MobileNetV2ModelHandler final : public ModelHandler
{
    public:
    MobileNetV2ModelHandler() : inputBuffer_(inputBatches_ * inputHeight_ * inputWidth_ * inputChannels_),
                                outputBuffer_(inputBatches_ * kClasses_) {}
    const char* const getModelPath() const override { return modelPath_; }
    std::vector<float>& getInputBuffer() override { return inputBuffer_; }
    std::vector<float>& getOutputBuffer() override { return outputBuffer_; }
    int64_t getInputHeight() const override { return inputHeight_; }
    int64_t getInputWidth() const override { return inputWidth_; }
    int64_t getInputChannels() const override { return inputChannels_; }
    int64_t getInputBatches() const override { return inputBatches_; }
    int64_t getNrOfClasses() const override { return kClasses_; }
    const char* const getLabels() const override { return labelsPath_; }
    std::string extractClassLabel(const std::string& labelStr) const override { return labelStr.substr(labelStr.find(' ') + 1, std::string::npos); }
    void BuildPreprocessPipeline() override;
    virtual std::string Postprocess() override { return extractClassLabel(ModelHandler::Postprocess()); }
    constexpr const std::array<ChannelNormParams, 3>& getChannelNormParams() const { return channelNormParams_; }

    private:
    static constexpr const char* const modelPath_{"/assets/model/mobilenetv2-12.onnx"};
    static constexpr const char* const labelsPath_{"/assets/labels/synset.txt"};
    static constexpr int64_t inputHeight_{224};
    static constexpr int64_t inputWidth_{224};
    static constexpr int64_t inputChannels_{3};
    static constexpr int64_t inputBatches_{1};
    static constexpr int64_t kClasses_{1000};
    static constexpr std::array<ChannelNormParams, 3> channelNormParams_
    {
        // R-channel
        ChannelNormParams{0.485, 0.229},
        // G-channel
        ChannelNormParams{0.456, 0.224},
        // B-channel
        ChannelNormParams{0.406, 0.225}
    };
    std::vector<float> inputBuffer_;
    std::vector<float> outputBuffer_;
};

#endif // #ifndef MODEL_HANDLER_H_