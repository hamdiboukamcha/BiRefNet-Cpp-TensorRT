#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

/// \class BiRefNet
/// \brief A class Bilateral Reference for High-Resolution Dichotomous Image Segmentation.
///
/// This class is responsible for initializing a TensorRT engine from a given model path,
/// preparing input data, running inference on input images, and returning the predicted output.
/// It manages CUDA resources (e.g., the TensorRT runtime, engine, context) and provides
/// convenient methods to preprocess input images and postprocess the inference results.
class BiRefNet
{
public:

	/// \brief Constructor for the BiRefNet class.
	///
	/// \param model_path Path to the serialized TensorRT engine file (e.g., .trt).
	/// \param logger A reference to a TensorRT ILogger implementation for logging.
	BiRefNet(std::string model_path, nvinfer1::ILogger& logger);

	/// \brief Runs inference on the given input image.
	///
	/// This method preprocesses the input image, runs inference on the loaded
	/// TensorRT engine, and returns a pair of cv::Mat containing the results (e.g., depth map, segmentation mask, etc.).
	///
	/// \param image The input cv::Mat image on which inference is to be performed.
	/// \return A  cv::Mat objects representing the inference outputs.
	cv::Mat predict(cv::Mat& image);

	/// \brief Destructor for the BiRefNet class.
	///
	/// Cleans up all allocated resources, including GPU buffers and CUDA streams.
	~BiRefNet();
	
private:
	int input_w = 1024;  ///< The input width for the model.
	int input_h = 1024;  ///< The input height for the model.

	float mean[3] = { 123.675, 116.28, 103.53 };  ///< Mean values for preprocessing.
	float std[3] = { 58.395, 57.12, 57.375 };     ///< Standard deviation values for preprocessing.

	std::vector<int> offset;                      ///< Offset values for internal calculations.

	nvinfer1::IRuntime* runtime = nullptr;        ///< Pointer to the TensorRT Runtime.
	nvinfer1::ICudaEngine* engine = nullptr;      ///< Pointer to the TensorRT Engine.
	nvinfer1::IExecutionContext* context = nullptr;///< Pointer to the TensorRT Execution Context.
	nvinfer1::INetworkDefinition* network = nullptr; ///< (Optional) Pointer to the Network definition if needed.

	void* buffer[2] = { nullptr, nullptr };       ///< I/O buffer pointers on the GPU.
	float* output_Data = nullptr;                  ///< Host pointer for depth output (example usage).
	cudaStream_t stream;                          ///< CUDA stream for asynchronous operations.

	/// \brief Resizes the given depth map image to the specified dimensions and returns extra info.
	///
	/// \param img The input depth map image (e.g., CV_32FC1 or CV_8UC1).
	/// \param w   The target width for the resized image.
	/// \param h   The target height for the resized image.
	/// \return A std::tuple containing:
	///         - A cv::Mat with the resized depth map.
	///         - The resized width (int).
	///         - The resized height (int).
	std::tuple<cv::Mat, int, int> resize_depth(cv::Mat& img, int w, int h);


	/// \brief Preprocessing function to convert the input image into a suitable tensor format.
	///
	/// \param image The input cv::Mat image.
	/// \return A vector of floats representing the preprocessed image data.
	std::vector<float> preprocess(cv::Mat& image);

	/// \brief Postprocesses the raw model output into a segmentation map.
	/// \param output_Data Pointer to the float array containing the model's output.
	/// \param img_w The width of the output segmentation map.
	/// \param img_h The height of the output segmentation map.
	/// \return A cv::Mat representing the postprocessed segmentation map.
	cv::Mat BiRefNet::postprocess(float* output_Data, int img_w, int img_h);

};
