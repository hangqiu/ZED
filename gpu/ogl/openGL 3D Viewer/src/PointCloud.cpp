#include "PointCloud.hpp"

PointCloud::PointCloud(size_t width, size_t height) {
    width = width;
    height = height;
    data.resize(width * height);
}

void PointCloud::fill(const float* bufferXYZRGBA) {
    int index4 = 0;

    const float fact = 0.001f;

    for (auto it = begin(); it != end(); it++) {
        it->x = bufferXYZRGBA[index4++] * fact;
        it->y = bufferXYZRGBA[index4++] * fact;
        it->z = bufferXYZRGBA[index4++] * fact;
        it->setColor(bufferXYZRGBA[index4++]);
    }
}

POINT3D PointCloud::getPoint(size_t i, size_t j) {
    return data[i + width * j];
}

size_t PointCloud::getNbPoints() {
    return data.size();
}

size_t PointCloud::getWidth() {
    return width;
}

size_t PointCloud::getHeight() {
    return height;
}