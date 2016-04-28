//CloudViewer.hpp

#ifndef __CLOUD_VIEWER_INCLUDE__
#define __CLOUD_VIEWER_INCLUDE__

#include "PointCloud.hpp"
#include "utils.hpp"
#include <math.h>
#include <thread>         // std::thread
#include <mutex>          // std::mutex
#include "GL/glut.h"    /* OpenGL Utility Toolkit header */

class TrackBallCamera {
public:

    TrackBallCamera() {};
    TrackBallCamera(vect3 p, vect3 la);
    void applyTransformations();
    void show();
    void rotation(float angle, vect3 v);
    void rotate(float speed, vect3 v);
    void translate(vect3 v);
    void translateLookAt(vect3 v);
    void translateAll(vect3 v);
    void zoom(float z);

    vect3 getPosition();
    vect3 getPositionFromLookAt();
    vect3 getLookAt();
    vect3 getForward();
    vect3 getUp();
    vect3 getLeft();

    void setPosition(vect3 p);
    void setLookAt(vect3 p);

private:
    vect3 position;
    vect3 lookAt;
    vect3 forward;
    vect3 up;
    vect3 left;
    float angleX;

    void setAngleX();
};

class CloudViewer {
public:
    CloudViewer();
    ~CloudViewer();
    void setUp(sl::zed::CamParameters& param, sl::zed::resolution res);
	void addData(PointCloud *cloud, std::mutex * ptr);
	unsigned char getKey();

	void init();
private:
	//! OpenGL Functions CALLBACKs
    static void redrawCallback();
    static void mouseCallback(int button, int state, int x, int y);
    static void motionCallback(int x, int y);
	static void reshapeCallback(int width, int height);
	static void keyboardCallback(unsigned char c, int x, int y);

	//! object
	TrackBallCamera camera;
	std::thread* toRun;
	std::mutex *ptr_points_locked;
    PointCloud *cloud;

	//! functions
	void visualize();
    void redraw();
    void mouse(int button, int state, int x, int y);
    void motion(int x, int y);
	void reshape(int width, int height);
	void keyboard(unsigned char c, int x, int y);


	//! member
    bool Rotate;
    bool Translate;
    bool Zoom;
    int startx;
	int starty;
	unsigned char keyPressed;
};
#endif	/* __CLOUD_VIEWER_INCLUDE__ */
