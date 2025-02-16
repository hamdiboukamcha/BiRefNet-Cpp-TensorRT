#include "birefnet.h"
#include <iostream>
#define isFP16 true

using namespace nvinfer1;

BiRefNet::BiRefNet(std::string model_path, nvinfer1::ILogger& logger)
{
    // Open the engine file in binary mode
    std::ifstream engineStream(model_path, std::ios::binary);
    if (!engineStream.is_open())
    {
        throw std::runtime_error("Failed to open engine file: " + model_path);
    }

    // Determine the size of the engine file
    engineStream.seekg(0, std::ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, std::ios::beg);

    if (modelSize == 0)
    {
        throw std::runtime_error("Engine file is empty or unreadable: " + model_path);
    }

    // Read the serialized engine data into memory
    std::unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Create the TensorRT runtime
    runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime)
    {
        throw std::runtime_error("Failed to create TensorRT runtime.");
    }

    // Deserialize the engine from memory
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    if (!engine)
    {
        throw std::runtime_error("Failed to deserialize CUDA engine.");
    }

    // Create the execution context
    context = engine->createExecutionContext();
    if (!context)
    {
        throw std::runtime_error("Failed to create TensorRT execution context.");
    }

#if NV_TENSORRT_MAJOR < 10
    // For TensorRT versions < 10, getBindingDimensions() is used.
    auto input_dims = engine->getBindingDimensions(0);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#else
    // For TensorRT versions >= 10, getTensorShape() is used.
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#endif

    // Create a CUDA stream for inference tasks
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        throw std::runtime_error("Failed to create CUDA stream.");
    }

    // Allocate device memory for input and output buffers
    if (cudaMalloc(&buffer[0], 3 * input_h * input_w * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&buffer[1], input_h * input_w * sizeof(float)) != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate CUDA buffers.");
    }

    // Allocate host memory for depth data (or other output purposes)
    output_Data = new float[input_h * input_w];
    if (!output_Data)
    {
        throw std::bad_alloc();
    }
}

BiRefNet::~BiRefNet()
{
    cudaFree(stream);
    cudaFree(buffer[0]);
    cudaFree(buffer[1]);

    delete[] output_Data;
}

std::tuple<cv::Mat, int, int> BiRefNet::resize_depth(cv::Mat& img, int w, int h)
{
    cv::Mat result;
    int nw, nh;
    int ih = img.rows;
    int iw = img.cols;
    float aspectRatio = (float)img.cols / (float)img.rows;

    if (aspectRatio >= 1)
    {
        nw = w;
        nh = int(h / aspectRatio);
    }
    else
    {
        nw = int(w * aspectRatio);
        nh = h;
    }
    cv::resize(img, img, cv::Size(nw, nh));
    result = cv::Mat::ones(cv::Size(w, h), CV_8UC1) * 128;
    cv::cvtColor(result, result, cv::COLOR_GRAY2RGB);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(h, w, CV_8UC3, 0.0);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

    std::tuple<cv::Mat, int, int> res_tuple = std::make_tuple(out, (w - nw) / 2, (h - nh) / 2);

    return res_tuple;
}

std::vector<float> BiRefNet::preprocess(cv::Mat& image)
{
    std::tuple<cv::Mat, int, int> resized = resize_depth(image, input_w, input_h);
    cv::Mat resized_image = std::get<0>(resized);
    std::vector<float> input_tensor;
    for (int k = 0; k < 3; k++)
    {
        for (int i = 0; i < resized_image.rows; i++)
        {
            for (int j = 0; j < resized_image.cols; j++)
            {
                input_tensor.emplace_back(((float)resized_image.at<cv::Vec3b>(i, j)[k] - mean[k]) / std[k]);
            }
        }
    }
    return input_tensor;
}

cv::Mat BiRefNet::predict(cv::Mat& image)
{
    // 1. Basic validation: Ensure the input image is not empty
    if (image.empty())
    {
        std::cerr << "[BiRefNet::predict] Error: Input image is empty." << std::endl;
        return cv::Mat();  // Return an empty Mat to indicate failure
    }

    // 2. Retrieve input dimensions from the OpenCV Mat
    const int img_w = image.cols;
    const int img_h = image.rows;

    // 3. Preprocess the input image: 
    //    Convert to float, normalize, and/or reshape as required by the model.
    std::vector<float> inputData = preprocess(image);

    // 4. Copy the preprocessed data from host (CPU) to device (GPU).
    //    buffer[0] is assumed to be the GPU memory designated for input.
    const size_t inputSize = 3 * input_h * input_w * sizeof(float);
    cudaError_t status = cudaMemcpyAsync(
        buffer[0],
        inputData.data(),
        inputSize,
        cudaMemcpyHostToDevice,
        stream
    );

    if (status != cudaSuccess)
    {
        std::cerr << "[BiRefNet::predict] Error: cudaMemcpyAsync (HostToDevice) failed: "
            << cudaGetErrorString(status) << std::endl;
        return cv::Mat();
    }

    // 5. Execute inference using the TensorRT context. 
    //    This step runs the forward pass of the model on the GPU.
#if NV_TENSORRT_MAJOR < 10
    // For older TensorRT versions, enqueueV2 requires an extra argument for the stream.
    if (!context->enqueueV2(buffer, stream, nullptr))
    {
        std::cerr << "[BiRefNet::predict] Error: context->enqueueV2 failed." << std::endl;
        return cv::Mat();
    }
#else
    // For TensorRT version >= 10, executeV2 is used.
    if (!context->executeV2(buffer))
    {
        std::cerr << "[BiRefNet::predict] Error: context->executeV2 failed." << std::endl;
        return cv::Mat();
    }
#endif

    // 6. Synchronize the stream to ensure all GPU operations (inference) are done 
    //    before proceeding to copy output back to the host.
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess)
    {
        std::cerr << "[BiRefNet::predict] Error: cudaStreamSynchronize after inference failed: "
            << cudaGetErrorString(status) << std::endl;
        return cv::Mat();
    }

    // 7. Copy inference results from device (GPU) to host (CPU).
    //    buffer[1] is assumed to be the GPU memory designated for output.
    const size_t outputSize = input_h * input_w * sizeof(float);
    status = cudaMemcpyAsync(
        output_Data,
        buffer[1],
        outputSize,
        cudaMemcpyDeviceToHost,
        stream
    );

    if (status != cudaSuccess)
    {
        std::cerr << "[BiRefNet::predict] Error: cudaMemcpyAsync (DeviceToHost) failed: "
            << cudaGetErrorString(status) << std::endl;
        return cv::Mat();
    }

    // 8. Synchronize again to ensure the output data has been successfully transferred.
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess)
    {
        std::cerr << "[BiRefNet::predict] Error: cudaStreamSynchronize after output copy failed: "
            << cudaGetErrorString(status) << std::endl;
        return cv::Mat();
    }

    // 9. Postprocess the output data: 
    //    Convert the raw float buffer to a meaningful output Mat (e.g., a depth map).
    cv::Mat output = postprocess(output_Data, img_w, img_h);

    // 10. Return the final output
    return output;
}

cv::Mat BiRefNet::postprocess(float* output_Data, int img_w, int img_h)
{
    // 1. Validate the pointer to avoid segmentation faults
    if (!output_Data)
    {
        throw std::invalid_argument("[postprocess] Output data pointer is null.");
    }

    // 2. Convert the raw float array into a cv::Mat of size (input_h x input_w).
    //    CV_32FC1 indicates a single-channel 32-bit floating-point matrix.
    //    Typically, this represents a depth or probability map from the model.
    cv::Mat floatMat(input_h, input_w, CV_32FC1, output_Data);

    // 3. Resize the matrix to match the original image dimensions (img_w x img_h).
    //    If you want to maintain exact aspect ratio differently, adjust accordingly.
    cv::resize(floatMat, floatMat, cv::Size(img_w, img_h));

    // ------------------------------------------------------------------------
    // OPTION A: Convert the single-channel float image to an 8-bit grayscale
    // ------------------------------------------------------------------------

    // 4a. Convert [0,1] float range to [0,255] 8-bit range. 
    //     - If your float values are in some other range (e.g., 0 to 10), adjust the scaling factor accordingly.
    //     - If floatMat has values in [0,1], multiply by 255.
    cv::Mat gray8U;
    floatMat.convertTo(gray8U, CV_8UC1, 255.0);

    // 5a. Convert the single-channel grayscale to a 3-channel BGR for display or saving as a “color” image.
    cv::Mat colorBGR;
    cv::cvtColor(gray8U, colorBGR, cv::COLOR_GRAY2BGR);

 
    return colorBGR;
}

