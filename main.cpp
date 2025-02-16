#include <iostream>
#include <string>
#include <tuple>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "birefnet.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace std;

// Helper function to replace all occurrences of a character in a string
void replaceChar(std::string& str, char find, char replace) {
    size_t pos = 0;
    while ((pos = str.find(find, pos)) != std::string::npos) {
        str[pos] = replace;
        pos++;
    }
}

bool IsPathExist(const std::string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}

bool IsFile(const std::string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

bool createFolder(const std::string& folderPath) {
#ifdef _WIN32
    if (!CreateDirectory(folderPath.c_str(), NULL)) {
        DWORD error = GetLastError();
        if (error == ERROR_ALREADY_EXISTS) {
            std::cout << "Folder already exists!" << std::endl;
            return true; // Folder already exists
        }
        else {
            std::cerr << "Failed to create folder! Error code: " << error << std::endl;
            return false; // Failed to create folder
        }
    }
#else
    if (mkdir(folderPath.c_str(), 0777) != 0) {
        if (errno == EEXIST) {
            std::cout << "Folder already exists!" << std::endl;
            return true; // Folder already exists
        }
        else {
            std::cerr << "Failed to create folder! Error code: " << errno << std::endl;
            return false; // Failed to create folder
        }
    }
#endif
    std::cout << "Folder created successfully!" << std::endl;
    return true; // Folder created successfully
}

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

int main(int argc, char** argv)
{
    // -------------------------------------------------------------------
    // 1. Define input engine file and input path (image or video).
    //    In production, these would typically come from command-line args.
    // -------------------------------------------------------------------
    const std::string engine_file_path = "BiRefNet-tiny.engine";
    const std::string path = "1693479941020.jpg";  // Replace with your input path
    std::vector<std::string> imagePathList;

    // Flag to indicate if we're processing a video
    bool isVideo = false;

    // -------------------------------------------------------------------
    // 2. Determine if the input path is a single file or a folder.
    //    If it's an image file (jpg/jpeg/png), we push it into the list.
    //    If it's a folder, we glob for all .jpg images.
    //    In production, you might add logic for other file types as well.
    // -------------------------------------------------------------------
    if (IsFile(path))
    {
        // Extract file extension
        std::string suffix = path.substr(path.find_last_of('.') + 1);

        // Check if it's one of the supported image formats
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png")
        {
            imagePathList.push_back(path);
        }
        else
        {
            std::cerr << "[Error] Unsupported file extension: " << suffix << std::endl;
            std::abort();
        }
    }
    else if (IsPathExist(path))
    {
        // If it's a folder or a valid path, glob for all .jpg images.
        // You can expand this to handle more extensions.
        cv::glob(path + "/*.jpg", imagePathList);
    }
    else
    {
        std::cerr << "[Error] Specified path does not exist: " << path << std::endl;
        return -1;
    }

    // -------------------------------------------------------------------
    // 3. Initialize the BiRefNet model with the specified engine file.
    // -------------------------------------------------------------------
    std::cout << "Loading model from " << engine_file_path << "..." << std::endl;
    BiRefNet birefnet_model(engine_file_path, logger);
    std::cout << "The model has been successfully loaded!" << std::endl;

    // -------------------------------------------------------------------
    // 4. If the input is a video, process it frame-by-frame.
    //    Otherwise, assume we are dealing with images.
    // -------------------------------------------------------------------
    if (isVideo)
    {
        // Open the video capture using the path (e.g., "video.mp4")
        cv::VideoCapture cap(path);
        if (!cap.isOpened())
        {
            std::cerr << "[Error] Unable to open video: " << path << std::endl;
            return -1;
        }

        // Retrieve video width and height
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        // Create a VideoWriter for saving the processed video
        // Adjust codec, fps, and output size as needed
        cv::VideoWriter output_video(
            "output_video.avi",
            cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
            30,
            cv::Size(width * 2, height) // Example: side-by-side output
        );

        // Read frames in a loop
        while (true)
        {
            cv::Mat frame;
            cap >> frame; // Read the next frame

            if (frame.empty())
            {
                // End of video or read error
                break;
            }

            // Optionally create a copy for display or other processing
            cv::Mat show_frame;
            frame.copyTo(show_frame);

            // Start timing
            auto start = std::chrono::system_clock::now();

            // Run inference using the BiRefNet model
            cv::Mat result = birefnet_model.predict(frame);

            // End timing
            auto end = std::chrono::system_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "[Info] Processing time per frame: " << elapsed_ms << " ms" << std::endl;

            // Write the processed frame to the output video
            output_video.write(result);
        }

        // Clean up resources
        cv::destroyAllWindows();
        cap.release();
        output_video.release();
    }
    else
    {
        // -------------------------------------------------------------------
        // Process a list of images (either a single file or multiple in a folder).
        // -------------------------------------------------------------------
        const std::string imageFolderPath_out = "results/";

        // Create a folder to store results (assuming createFolder is defined)
        createFolder(imageFolderPath_out);

        // Iterate over all images in imagePathList
        for (const auto& imagePath : imagePathList)
        {
            // Read the image
            cv::Mat frame = cv::imread(imagePath);
            if (frame.empty())
            {
                std::cerr << "[Warning] Failed to read image: " << imagePath << std::endl;
                continue;
            }

            // Optionally make a copy if needed for display
            cv::Mat show_frame;
            frame.copyTo(show_frame);

            // Start timing
            auto start = std::chrono::system_clock::now();

            // Predict using BiRefNet
            cv::Mat result = birefnet_model.predict(frame);

            // End timing
            auto end = std::chrono::system_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "[Info] Processing time: " << elapsed_ms << " ms for " << imagePath << std::endl;

            // Extract the filename from the path to construct an output path
            // A simple way is to split on '/' (or '\\' for Windows) 
            // then take the last token.
            std::istringstream iss(imagePath);
            std::string token;
            while (std::getline(iss, token, '/')) { /* no-op */ }
            // For Windows paths, you may need additional handling
            token = token.substr(token.find_last_of("/\\") + 1);

            // Print the path to confirm
            std::string output_path = imageFolderPath_out + token;
            std::cout << "[Info] Saving result to: " << output_path << std::endl;

            // Save the result image to disk
            if (!cv::imwrite(output_path, result))
            {
                std::cerr << "[Warning] Failed to write image: " << output_path << std::endl;
            }
        }
    }

    // -------------------------------------------------------------------
    // 5. Inform the user the process is complete and exit.
    // -------------------------------------------------------------------
    std::cout << "[Info] Processing finished." << std::endl;
    return 0;
}