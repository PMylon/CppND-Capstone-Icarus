#ifndef RUNTIME_H_
#define RUNTIME_H_

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstddef>

class Runtime
{
    public:
    Runtime(const Runtime& other) = delete;
    Runtime& operator=(const Runtime& other) = delete;	
    static Runtime& Instance();
    void Prepare(const char* const modelPath, std::vector<float>& inputData, std::vector<float>& outputData, int64_t batchSize);
    void Execute();
    const std::vector<int64_t>& getInputShape() const noexcept {return inputShape_;}
    const std::vector<int64_t>& getOutputShape() const noexcept {return outputShape_;}
    void PrintModelInfo();

    private:
    Runtime() : session_{nullptr}, memoryInfo_{nullptr}, inputTensor_{nullptr}, outputTensor_{nullptr} {};
    Ort::Session session_;
    Ort::MemoryInfo memoryInfo_;
    Ort::Value inputTensor_;
    Ort::Value outputTensor_;
    std::vector<int64_t> inputShape_;
    std::vector<int64_t> outputShape_;
};

#endif // #ifndef RUNTIME_H_