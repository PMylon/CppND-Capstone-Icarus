#include "runtime.h"
#include "model_handler.h"
#include "image_provider.h"
#include "image_preprocessor.h"
#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <iterator>
#include <utility>
#include <future>

std::condition_variable cvInputAvailable;
std::mutex mtxInput;
std::queue<Image> inputImageQueue;

std::condition_variable cvClassifierResultReady;
std::mutex mtxClassifierResult;
using LabelledImage = std::pair<Image, std::string>;

struct ClassifierResult
{
    LabelledImage labelledImage_;
    std::chrono::milliseconds inferenceTime_;

    ClassifierResult() = default;
    ClassifierResult(LabelledImage labelledImage, std::chrono::milliseconds inferenceTime) : labelledImage_{labelledImage}, inferenceTime_{inferenceTime} {}
};

std::queue<ClassifierResult> classifierResultQueue;

void ImageCaptureThread(std::shared_future<void> futTerminate)
{
    using namespace std::chrono_literals;

    ImageProvider imgProvider;

    while (futTerminate.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
    {
        std::this_thread::sleep_for(500ms);

        auto img = imgProvider.GetImage();

        {
            std::lock_guard<std::mutex> lgInput{mtxInput};
            inputImageQueue.push(img);
            std::cout << "Captured image: " << img.path << std::endl;
        }

        cvInputAvailable.notify_one();
    }
}

void InferenceThread(std::shared_future<void> futTerminate)
{
    MobileNetV2ModelHandler modelHandler;

    std::vector<float>& inputValues = modelHandler.getInputBuffer();
    std::vector<float>& outputValues = modelHandler.getOutputBuffer();

    modelHandler.BuildPreprocessPipeline();

    auto& runtime = Runtime::Instance();

    runtime.Prepare(modelHandler.getModelPath(), inputValues, outputValues, modelHandler.getInputBatches());
    runtime.PrintModelInfo();

    while (futTerminate.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
    {
        Image img;

        {
            std::unique_lock<std::mutex> ulInput{mtxInput};
            cvInputAvailable.wait(ulInput, [&]()
            {
                return !inputImageQueue.empty();
            });

            img = std::move(inputImageQueue.front());
            inputImageQueue.pop();
        }

        LabelledImage labelledImg{img, "Unknown"};

        modelHandler.Preprocess(img);

        inputValues.assign(std::make_move_iterator(img.matrix.begin<float>()), std::make_move_iterator(img.matrix.end<float>()));

        std::chrono::steady_clock::time_point inferenceStartTime = std::chrono::steady_clock::now();
        runtime.Execute();
        std::chrono::steady_clock::time_point inferenceEndTime = std::chrono::steady_clock::now();

        auto inferenceTime = std::chrono::duration_cast<std::chrono::milliseconds>(inferenceEndTime - inferenceStartTime);

        auto predictedClass = modelHandler.Postprocess();

        std::cout << "Predicted image: " << predictedClass << " Inference Time: " << inferenceTime.count() << "ms\n";

        std::get<std::string>(labelledImg) = predictedClass;

        {
            std::lock_guard<std::mutex> lgClassifierResult{mtxClassifierResult};
            classifierResultQueue.emplace(labelledImg, inferenceTime);
        }

        cvClassifierResultReady.notify_one();
    }
}

void ImageDisplayThread(std::promise<void>&& prmsTerminate)
{
    const std::string& displayWindowName = "Classifier Result Window";

    cv::namedWindow(displayWindowName, cv::WINDOW_NORMAL);

    while (true)
    {
        ClassifierResult result;

        {
            std::unique_lock<std::mutex> ulClassifierResult{mtxClassifierResult};
            cvClassifierResultReady.wait(ulClassifierResult, [&]()
            {
                return !classifierResultQueue.empty();
            });

            result = std::move(classifierResultQueue.front());
            classifierResultQueue.pop();
        }

        auto img = std::get<Image>(result.labelledImage_);
        auto predictedClass = std::get<std::string>(result.labelledImage_);
        auto time = result.inferenceTime_;

        std::ostringstream inferenceTimeOss;

        inferenceTimeOss << "Inference Time: " << time.count() << "ms";

        cv::Size titleSize = cv::getTextSize(predictedClass, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, nullptr);
        int displayWindowHeight = img.height + titleSize.height + 20;
        int displayWindowWidth = std::max(img.width, titleSize.width + 20);

        cv::resizeWindow(displayWindowName, displayWindowWidth, displayWindowHeight);
        cv::imshow(displayWindowName, img.matrix);

        cv::setWindowTitle(displayWindowName, predictedClass + "----" + inferenceTimeOss.str());

        cv::waitKey(1500); 

        // Check window's property in order to determine if window was closed
        if (cv::getWindowProperty(displayWindowName, cv::WND_PROP_AUTOSIZE) < 0)
        {
            prmsTerminate.set_value();
            break;
        }
    }
}

int main() {
    std::cout << "Image Classification" << "\n";

    std::promise<void> prmsTerminate;

    std::shared_future<void> futTerminate{prmsTerminate.get_future()};

    std::thread imageCaptureThread{ImageCaptureThread, futTerminate};
    std::thread inferenceThread{InferenceThread, futTerminate};
    std::thread imageDisplayThread{ImageDisplayThread, std::move(prmsTerminate)};

    imageCaptureThread.join();
    inferenceThread.join();
    imageDisplayThread.join();

    return EXIT_SUCCESS;
}
