#include "model_handler.h"

static std::ifstream& GoToLine(std::ifstream& ifstrm, size_t line_num)
{
    ifstrm.seekg(std::ios::beg);

    for (size_t line_idx = 0; line_idx < line_num; line_idx++)
    {
        ifstrm.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    return ifstrm;
}

std::string ModelHandler::Postprocess()
{
    const std::vector<float>& outputValues = getOutputBuffer();

    auto maxScoreItr = std::max_element(outputValues.begin(), outputValues.end());
    auto predictedClassIdx = std::distance(outputValues.begin(), maxScoreItr);

    std::filesystem::path cwd = std::filesystem::current_path();
    const std::string absLabelsPath = cwd.string() + getLabels();

    std::ifstream ifstrm{absLabelsPath, std::ios::in};

    if (!ifstrm.is_open())
    {
        std::cerr << "Could not open file: " << absLabelsPath << std::endl;

        std::exit(EXIT_FAILURE);
    }

    (void)GoToLine(ifstrm, predictedClassIdx);

    std::string labelStr;

    std::getline(ifstrm, labelStr);

    return labelStr;
}

void MobileNetV2ModelHandler::BuildPreprocessPipeline()
{
    ImagePreprocessingPipelineBuilder builder;
    builder.addResize(inputHeight_, inputWidth_);
    builder.addConvertColor(ColorFormat::RGB);
    builder.addNormalize(channelNormParams_);
    builder.addConvertMemLayout(MemoryLayout::CHW);
    preprocessingPipeline_ = std::move(builder.build());
}