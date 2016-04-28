///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////




/**************************************************************************************************
 ** This sample demonstrates how to record a SVO file with the ZED SDK function                   **
 ** optionally, images are displayed using OpenCV                                                 **
 ***************************************************************************************************/

/* this sample use BOOST for program options arguments */


//standard includes
#include <cstdio>
#include <cstring>
#include <signal.h>
#include <cstdlib>
#include <chrono>
#include <thread>


//Opencv Include (for display
#include <opencv2/opencv.hpp>

//ZED Include
#include <zed/Mat.hpp>
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>


//boost for program options
#include <boost/program_options.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace sl::zed;
using namespace std;
namespace po = boost::program_options;

Camera* zed_ptr;

#ifdef _WIN32

BOOL CtrlHandler(DWORD fdwCtrlType) {
    switch (fdwCtrlType) {
            // Handle the CTRL-C signal.
        case CTRL_C_EVENT:
            printf("\nSaving file...\n");
            zed_ptr->stopRecording();
            delete zed_ptr;
            exit(0);
        default:
            return FALSE;
    }
}
#else

void nix_exit_handler(int s) {
    printf("\nSaving file...\n");
    zed_ptr->stopRecording();
    delete zed_ptr;
    exit(1);
}
#endif

int main(int argc, char **argv) {
    std::string filename = "zed_record.svo";

    bool display = 0;
    int resolution = 2; //Default resolution is set to HD720

    // Option handler (boost program_options)
    int opt;
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("display,d", "toggle image display (might slow down the recording)")
            ("resolution,r", po::value<int>(&opt)->default_value(2), "ZED Camera resolution \narg: 0: HD2K   1: HD1080   2: HD720   3: VGA")
            ("filename,f", po::value< std::string >(), "Record filename")
            ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }
    if (vm.count("display")) {
        cout << "Display on" << endl;
        display = 1;
    }
    if (vm.count("filename")) {
        filename = vm["filename"].as<std::string>();
        cout << "Filename was set to " << filename << endl;
    }
    if (vm.count("resolution")) {
        resolution = vm["resolution"].as<int>();
        switch (resolution) {
            case 0: cout << "Resolution set to HD2K" << endl;
                break;
            case 1: cout << "Resolution set to HD1080" << endl;
                break;
            case 2: cout << "Resolution set to HD720" << endl;
                break;
            case 3: cout << "Resolution set to VGA" << endl;
                break;
        }
    }
    // Camera init
    ERRCODE err;
    // Create the camera at HD 720p resolution
    // The realtime recording will depend on the write speed of your disk.
    Camera* zed = new Camera(static_cast<ZEDResolution_mode> (resolution));
    // ! not the same Init function - a more lighter one , specific for recording !//
    err = zed->initRecording(filename);

    zed_ptr = zed; // To call Camera::stop_recording() from the exit handler function

    std::cout << "ERR code : " << errcode2str(err) << std::endl;

    // Quit if an error occurred
    if (err != SUCCESS) {
        delete zed;
        return 1;
    }

    // CTRL-C (= kill signal) handler
#ifdef _WIN32
    SetConsoleCtrlHandler((PHANDLER_ROUTINE) CtrlHandler, TRUE);
#else // unix
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = nix_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
#endif

    // Wait for the auto exposure and white balance
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Recording loop
    cout << "Recording..." << endl;
    cout << "Press 'Ctrl+C' to stop and exit " << endl;
    while (1) {
        //simple recording function
        zed->record(); // record the current frame with the predefined size

        if (display) {
            sl::zed::Mat imgSbS_YUV_cpu = zed->getCurrentRawRecordedFrame();
            cv::Mat displayImg(imgSbS_YUV_cpu.height, imgSbS_YUV_cpu.width * 2, CV_8UC2, imgSbS_YUV_cpu.data);
            cv::cvtColor(displayImg, displayImg, CV_YUV2BGRA_YUYV);
            cv::namedWindow("ZED display", 0);
            cv::imshow("ZED display", displayImg);
            cv::waitKey(25);
        }
        //if (display) zed->displayRecorded(); // convert the image to RGB and display it
    }

    return 0;
}
