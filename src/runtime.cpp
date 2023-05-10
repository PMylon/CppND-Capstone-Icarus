#include "runtime.h"
#include <iostream>
#include <string>
#include <filesystem>

Runtime& Runtime::Instance()
{
    static Runtime instance;

    return instance;
}

void Runtime::Prepare(const char* const modelPath, std::vector<float>& inputData, std::vector<float>& outputData, int64_t batchSize = -1)
{
    Ort::Env env;

    std::filesystem::path cwd = std::filesystem::current_path();

    std::string absModelPath = cwd.string() + modelPath;

    Ort::Session session{env, absModelPath.c_str(), Ort::SessionOptions{}};

    std::vector<int64_t> inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> outputShape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    if (inputShape[0] == -1)
    {
        // Dynamic batch size dimension
        inputShape[0] = batchSize;
    }

    if (outputShape[0] == -1)
    {
        // Dynamic batch size dimension
        outputShape[0] = batchSize;
    }

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memoryInfo, outputData.data(), outputData.size(), outputShape.data(), outputShape.size());

    session_ = std::move(session);
    memoryInfo_ = std::move(memoryInfo);
    inputTensor_ = std::move(inputTensor);
    outputTensor_ = std::move(outputTensor);
    inputShape_ = std::move(inputShape);
    outputShape_ = std::move(outputShape);
}

void Runtime::Execute()
{
    Ort::AllocatorWithDefaultOptions allocator;

    const auto inputName = session_.GetInputNameAllocated(0, allocator);
    const auto outputName = session_.GetOutputNameAllocated(0, allocator);

    std::array<const char*, 1> inputNames{inputName.get()};
    std::array<const char*, 1> outputNames{outputName.get()};

    session_.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor_, session_.GetInputCount(), outputNames.data(), &outputTensor_, session_.GetOutputCount());
}

void Runtime::PrintModelInfo()
{
    std::cout << "=====================================================================" << std::endl;
    std::cout << "Model Info: " << std::endl;

    std::cout << "Nr of model inputs: " << session_.GetInputCount() << std::endl;
    std::cout << "Nr of model outputs: " << session_.GetOutputCount() << std::endl;

    std::cout << "Input shape: " << std::endl;

    for (const auto& dim : inputShape_)
    {
        std::cout << dim << std::endl;
    }

    std::cout << "Output shape: " << std::endl;

    for (const auto& dim : outputShape_)
    {
        std::cout << dim << std::endl;
    }
    std::cout << "=====================================================================" << std::endl;
}