//CloudViewer.cpp

#include "CloudViewer.hpp"

using namespace sl::zed;

CloudViewer* ptr;

void idle(void) {
    glutPostRedisplay();
}

TrackBallCamera::TrackBallCamera(vect3 p, vect3 la) {
    position.x = p.x;
    position.y = p.y;
    position.z = p.z;

    lookAt.x = la.x;
    lookAt.y = la.y;
    lookAt.z = la.z;
    angleX = 0.0f;
    applyTransformations();
}

void TrackBallCamera::applyTransformations() {
    forward = vect3(lookAt.x - position.x,
            lookAt.y - position.y,
            lookAt.z - position.z);
    left = vect3(forward.z, 0, -forward.x);
    up = vect3(left.y * forward.z - left.z * forward.y,
            left.z * forward.x - left.x * forward.z,
            left.x * forward.y - left.y * forward.x);
    forward.normalise();
    left.normalise();
    up.normalise();
}

void TrackBallCamera::show() {
    gluLookAt(position.x, position.y, position.z,
            lookAt.x, lookAt.y, lookAt.z,
            0.0, 1.0, 0.0);
}

void TrackBallCamera::rotation(float angle, vect3 v) {
    translate(vect3(-lookAt.x, -lookAt.y, -lookAt.z));
    position.rotate(angle, v);
    translate(vect3(lookAt.x, lookAt.y, lookAt.z));
    setAngleX();
}

void TrackBallCamera::rotate(float speed, vect3 v) {
    float tmpA;
    float angle = speed / 360.0f;
    (v.x != 0.0f) ? tmpA = angleX - 90.0f + angle : tmpA = angleX - 90.0f;
    if (tmpA < 89.5f && tmpA > -89.5f) {
        translate(vect3(-lookAt.x, -lookAt.y, -lookAt.z));
        position.rotate(angle, v);
        translate(vect3(lookAt.x, lookAt.y, lookAt.z));
    }
    setAngleX();
}

void TrackBallCamera::translate(vect3 v) {
    position.x += v.x;
    position.y += v.y;
    position.z += v.z;
}

void TrackBallCamera::translateLookAt(vect3 v) {
    lookAt.x += v.x;
    lookAt.y += v.y;
    lookAt.z += v.z;
}

void TrackBallCamera::translateAll(vect3 v) {
    translate(v);
    translateLookAt(v);
}

void TrackBallCamera::zoom(float z) {
    float dist = vect3::length(vect3(position.x - lookAt.x, position.y - lookAt.y, position.z - lookAt.z));
    if (dist - z > z) {
        translate(vect3(forward.x * z, forward.y * z, forward.z * z));
    }
}

vect3 TrackBallCamera::getPosition() {
    return vect3(position.x, position.y, position.z);
}

vect3 TrackBallCamera::getPositionFromLookAt() {
    return vect3(position.x - lookAt.x, position.y - lookAt.y, position.z - lookAt.z);
}

vect3 TrackBallCamera::getLookAt() {
    return vect3(lookAt.x, lookAt.y, lookAt.z);
}

vect3 TrackBallCamera::getForward() {
    return vect3(forward.x, forward.y, forward.z);
}

vect3 TrackBallCamera::getUp() {
    return vect3(up.x, up.y, up.z);
}

vect3 TrackBallCamera::getLeft() {
    return vect3(left.x, left.y, left.z);
}

void TrackBallCamera::setPosition(vect3 p) {
    position.x = p.x;
    position.y = p.y;
    position.z = p.z;
    setAngleX();
}

void TrackBallCamera::setLookAt(vect3 p) {
    lookAt.x = p.x;
    lookAt.y = p.y;
    lookAt.z = p.z;
    setAngleX();
}

void TrackBallCamera::setAngleX() {
    angleX = vect3::getAngle(vect3(position.x, position.y + 1, position.z),
            vect3(position.x, position.y, position.z),
            vect3(lookAt.x, lookAt.y, lookAt.z));
}

void glutThreadFunc() {
	ptr->init();
	glutMainLoop();
}

CloudViewer::CloudViewer() {
    camera = TrackBallCamera(vect3(0.0f, .1f, .5f), vect3(0.0f, 0.0f, 0.0f));

    Translate = false;
    Rotate = false;
    Zoom = false;

	cloud = NULL;

    ptr = this;
    toRun = new std::thread(glutThreadFunc);
}

CloudViewer::~CloudViewer() {
	glutDestroyWindow(1);
    toRun->join();
}

void CloudViewer::redrawCallback() {
    ptr->redraw();
}

void CloudViewer::mouseCallback(int button, int state, int x, int y) {
    ptr->mouse(button, state, x, y);
}

void CloudViewer::motionCallback(int x, int y) {
    ptr->motion(x, y);
}

void CloudViewer::reshapeCallback(int width, int height) {
    ptr->reshape(width, height);
}

void CloudViewer::keyboardCallback(unsigned char c, int x, int y) {
	ptr->keyboard(c, x, y);
}

void CloudViewer::init() {
	char *argv[1];
	int argc = 1;
#if _WIN32
	argv[0] = _strdup("ZED View");
#else
	argv[0] = strdup("ZED View");
#endif
	glutInit(&argc, argv);
	
    glutInitWindowSize(1000, 1000);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("ZED 3D Viewer");
	
    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    gluPerspective(75.0, 1.0, .20, 25000.0);
    glMatrixMode(GL_MODELVIEW);
    gluLookAt(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, .10, 0.0);

    glShadeModel(GL_SMOOTH);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    glutDisplayFunc(redrawCallback);
    glutMouseFunc(mouseCallback);
    glutMotionFunc(motionCallback);
	glutReshapeFunc(reshapeCallback);
	glutKeyboardFunc(keyboardCallback);
	glutIdleFunc(idle);

    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
}

void CloudViewer::setUp(sl::zed::CamParameters& param, sl::zed::resolution res) {
    float FoV = (2. * atan(param.cx / param.fx)) / 3.141592654 * 180.;
    float Aspect = res.width / res.height;

    gluPerspective(FoV, Aspect, .20, 25000.0);
}

void CloudViewer::visualize() {
    glPointSize(2);
    ptr_points_locked->try_lock();
    glBegin(GL_POINTS);
    for (auto it = cloud->cbegin(); it != cloud->cend(); ++it) {
        if (it->z > 0) {
            glColor3f(it->r, it->g, it->b);
            glVertex3f(it->x, -it->y, -it->z);
        }
    }
    glEnd();
    ptr_points_locked->unlock();

}

void CloudViewer::addData(PointCloud *cloud, std::mutex * ptr) {
    this->cloud = cloud;
	this->ptr_points_locked = ptr;
}

void CloudViewer::redraw() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    //glPushMatrix();
    camera.applyTransformations();
    camera.show();
    glEnable(GL_BLEND);
    glDisable(GL_LIGHTING);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glClearColor(0.12f, 0.12f, 0.12f, 1.0f);

	if (cloud)
		visualize();

    //glPopMatrix();
    glDisable(GL_BLEND);
    glutSwapBuffers();
}

void CloudViewer::mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            Rotate = true;
            startx = x;
            starty = y;
        }
        if (state == GLUT_UP) {
            Rotate = false;
        }
    }
    if (button == GLUT_RIGHT_BUTTON) {
        if (state == GLUT_DOWN) {
            Translate = true;
            startx = x;
            starty = y;
        }
        if (state == GLUT_UP) {
            Translate = false;
        }
    }

    if (button == GLUT_MIDDLE_BUTTON) {
        if (state == GLUT_DOWN) {
            Zoom = true;
            startx = x;
            starty = y;
        }
        if (state == GLUT_UP) {
            Zoom = false;
        }
    }

    if ((button == 3) || (button == 4)) {
        if (state == GLUT_UP) return;
        if (button == 3) {
            camera.zoom(0.2);
        } else
            camera.zoom(-0.2);
    }
}

void CloudViewer::motion(int x, int y) {
    if (Translate) {
        float Trans_x = (x - startx) / 60.0f;
        float Trans_y = (y - starty) / 60.0f;

        vect3 left = camera.getLeft();
        vect3 up = camera.getUp();

        camera.translateAll(vect3(left.x * Trans_x, left.y * Trans_x, left.z * Trans_x));
        camera.translateAll(vect3(up.x * -Trans_y, up.y * -Trans_y, up.z * -Trans_y));

        startx = x;
        starty = y;
    }

    if (Zoom) {
        camera.zoom((float) (y - starty) / 5.0f);
        starty = y;
    }

    if (Rotate) {
        float sensitivity = 100.0f;
        float Rot = y - starty;
        vect3 tmp = camera.getPositionFromLookAt();
        tmp.y = tmp.x;
        tmp.x = -tmp.z;
        tmp.z = tmp.y;
        tmp.y = 0.0f;
        tmp.normalise();
        camera.rotate(Rot * sensitivity, tmp);

        Rot = x - startx;
        camera.rotate(-Rot * sensitivity, vect3(0.0f, 1.0f, 0.0f));

        startx = x;
        starty = y;
    }

    glutPostRedisplay();
}

void CloudViewer::reshape(int width, int height) {
	glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	gluPerspective(75.0, width / height, .20, 25000.0);
    glMatrixMode(GL_MODELVIEW);
}

void CloudViewer::keyboard(unsigned char c, int x, int y) {
	keyPressed = c;
	glutPostRedisplay();
}

unsigned char CloudViewer::getKey() {
	unsigned char key_swap = keyPressed;
	keyPressed = ' ';
	return key_swap;
}