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




/****************************************************************************************************
 ** This sample demonstrates how to play an SVO file using the ZED SDK. You can easily pause,      **
 ** fast-forward or rewind the SVO file.                                                           **
 **                                                                                                **
 ** A SVO file must be specified in cmd arguments //See Recorder sample or ZEDExplorer to save SVO **
 ****************************************************************************************************/


//standard includes
#include <ctime>
#include <chrono>
#include <random>

//opencv includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//ZED Includes
#include <zed/Camera.hpp>

using namespace sl::zed;
using namespace std;

//Main function

int main(int argc, char **argv) {
    ERRCODE err;
    sl::zed::Camera *zed;

    //Open the SVO file specified in the cmd line
    if (argc == 2) {
        zed = new sl::zed::Camera(argv[1]);
        cout << "Playing SVO file : " << argv[1] << endl;
    } else {
        cout << " !!!! You must specify the SVO filename !!!!\n" << endl;
    }

    err = zed->init(sl::zed::MODE::NONE, 0, true);

    //ERRCODE display
    cout << errcode2str(err) << endl;

    //Quit if an error occurred
    if (err != SUCCESS) {
        delete zed;
        return 1;
    }

    cout << "\nCOMMANDS:\n\nSPACEBAR : PLAY/PAUSE\nf : FAST-FORWARD\nr : REWIND\nq : QUIT\n" << endl;

    //Get the width and height of the recorded image
    int width = zed->getImageSize().width;
    int height = zed->getImageSize().height;
    cv::Size size(width, height);

    //Set the display size
    cv::Size displaySize(720, 404);

    //Mats to store view and resized view
    cv::Mat view, viewDisplay;

    //Create an Opencv display window
    cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);

    //Initialize variables
    char key = ' '; // key pressed
    int ViewID = 2; // view type
    int svoPosition = 0; // SVO frame index
    bool paused = false; // play/pause toggle
    int ff = 1; // fast-forward speed
    int r = 0; // rewind speed

    cout<<" Press 'q' to exit"<<endl;
    cout << endl;
    cout << " > PLAY" << flush;

    zed->setSVOPosition(svoPosition);
    //loop until 'q' is pressed
    while (key != 'q') {
        if (!paused) {

            //Fast-forward
            if (ff != 1 && r == 0) {
                svoPosition = zed->getSVOPosition();
                svoPosition += ff;
                zed->setSVOPosition(svoPosition);
            }//Rewind
            else if (r != 0 && ff == 1) {
                svoPosition = zed->getSVOPosition();
                svoPosition = (svoPosition - r >= 0) ? svoPosition - r : 0;
                zed->setSVOPosition(svoPosition);
                if (svoPosition == 0) {
                    paused = true;
                    cout << "\r" << flush;
                    cout << "|| PAUSE    " << flush;
                }
            }

            //Get frames
            zed->grab(sl::zed::SENSING_MODE::RAW, false, false);

            //Even if Left and Right images are still available through getView() function, it's better since v0.8.1 to use retrieveImage for cpu readback because GPU->CPU is done async during depth estimation.
            // Therefore :
            // -- if disparity estimation is enabled in grab function, retrieveImage will take no time because GPU->CPU copy has already been done during disp estimation
            // -- if disparity estimation is not enabled, GPU->CPU copy is done in retrieveImage fct, and this function will take the time of copy.
            if (ViewID == sl::zed::STEREO_LEFT || ViewID == sl::zed::STEREO_RIGHT) {
                slMat2cvMat(zed->retrieveImage(static_cast<SIDE> (ViewID))).copyTo(view);
                cv::resize(view, viewDisplay, displaySize);
            } else {
                slMat2cvMat(zed->getView(static_cast<VIEW_MODE> (ViewID))).copyTo(view);
                cv::resize(view, viewDisplay, displaySize);
            }

            cv::imshow("Display", viewDisplay);
        }

        key = cv::waitKey(5);

        switch (key) {

                // ______________  VIEW __________________
            case '0': // left
                ViewID = 0;
                break;
            case '1': // right
                ViewID = 1;
                break;
            case '2': // anaglyph
                ViewID = 2;
                break;
            case '3': // gray scale diff
                ViewID = 3;
                break;
            case '4': // Side by side
                ViewID = 4;
                break;
            case '5': // overlay
                ViewID = 5;
                break;
                // ___________ PLAY / PAUSE ______________
            case ' ':
            {
                paused = !paused;
                if (r != 0) {
                    r = 0;
                    if (paused) {
                        cout << "\r" << flush;
                        cout << "|| PAUSE    " << flush;
                    } else {
                        cout << "\r" << flush;
                        cout << " > PLAY   1X" << flush;
                    }
                } else {
                    if (paused) {
                        cout << "\r" << flush;
                        cout << "|| PAUSE " << flush;
                    } else {
                        cout << "\r" << flush;
                        cout << " > PLAY  " << flush;
                    }
                }


                break;
            }
                // ___________ FAST-FORWARD ______________
            case 'f':
            {
                r = 0;
                if (paused) {
                    paused = false;
                    ff = 0;
                }

                ff = (ff > 0 && ff <= 4) ? ff * 2 : 1;

                if (ff != 1) {
                    cout << "\r" << flush;
                    cout << ">> PLAY   " << ff << "X    " << flush;
                } else {
                    cout << "\r" << flush;
                    cout << " > PLAY   " << ff << "X    " << flush;
                }

                break;
            }
                // ______________ REWIND _________________
            case 'r':
            {
                ff = 1;
                if (paused) {
                    paused = false;
                    r = 0;
                }
                r = (r > 0 && r <= 4) ? r * 2 : 1;
                if (r != 1) {
                    cout << "\r" << flush;
                    cout << "<< REWIND " << r << "X    " << flush;
                } else {
                    cout << "\r" << flush;
                    cout << " < REWIND " << r << "X    " << flush;
                }

                break;
            }
        }

    }

    delete zed;

    cout << "\nQuitting ..." << endl;

    return 0;
}
