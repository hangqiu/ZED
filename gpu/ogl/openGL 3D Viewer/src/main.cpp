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




/***************************************************************************************************
 ** This sample demonstrates how to grab images and depth map with the ZED SDK                    **
 ** and apply the result in a 3D view "point cloud style" with OpenGL /freeGLUT                   **
 ** Some of the functions of the ZED SDK are linked with a key press event		          **
 ***************************************************************************************************/


//standard Include
#include <stdio.h>
#include <string.h>
#include <chrono>

//ZED Include
#include <zed/Mat.hpp>
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

//our point cloud generator and viewer.
#include "PointCloud.hpp"
#include "CloudViewer.hpp"

using namespace sl::zed;
using namespace std;

//main Loop

int main(int argc, char **argv) {

    if (argc > 2) {
        std::cout << "Only the path of a SVO can be passed in arg" << std::endl;
        return -1;
    }

    Camera* zed;

    if (argc == 1) // Use in Live Mode
        zed = new Camera(HD720);
    else // Use in SVO playback mode
        zed = new Camera(argv[1]);

    ERRCODE err = zed->init(MODE::PERFORMANCE, -1, true); //need quite a powerful graphic card in QUALITY

    // ERRCODE display
    cout << errcode2str(err) << endl;

    // Quit if an error occurred
    if (err != SUCCESS) {
        delete zed;
        return 1;
    }

    int width = zed->getImageSize().width;
    int height = zed->getImageSize().height;

    // remove the not to be trusted data
    zed->setConfidenceThreshold(90);

    Mat BufferXYZRGBA;

    PointCloud *cloud = new PointCloud(width, height);
    CloudViewer *viewer = new CloudViewer();

    std::mutex ptr_points_locked;

    viewer->setUp(zed->getParameters()->LeftCam, sl::zed::resolution(width, height));
    viewer->addData(cloud, &ptr_points_locked);

    unsigned char key = ' ';

    while ((key != 'q') && (key != 27)) {
        // Get frames and launch the computation
        if (!zed->grab(SENSING_MODE::RAW)) {
            BufferXYZRGBA = zed->retrieveMeasure(MEASURE::XYZRGBA);
            ptr_points_locked.try_lock();
            cloud->fill((float*) BufferXYZRGBA.data);
            ptr_points_locked.unlock();
        }
        key = viewer->getKey();
    }

    delete zed;
    delete cloud;
    delete viewer;
    return 0;
}
