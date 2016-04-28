// PointCloud.hpp
#ifndef __SLPOINTCLOUD_H__
#define __SLPOINTCLOUD_H__

#include <zed/utils/GlobalDefine.hpp>
#include "utils.hpp"

class PointCloud {
public:
    PointCloud(size_t width, size_t height);

    void fill(const float* BufferXYZRGBA);

    POINT3D getPoint(size_t i, size_t j);

    size_t getNbPoints();
    size_t getWidth();
    size_t getHeight();

    // Iterator definition
    typedef std::vector<POINT3D>::iterator iterator;
    typedef std::vector<POINT3D>::const_iterator const_iterator;

    iterator begin() {
        return data.begin();
    }

    iterator end() {
        return data.end();
    }

    const_iterator cbegin() {
        return data.cbegin();
    }

    const_iterator cend() {
        return data.cend();
    }

private:
    std::vector<POINT3D> data;

    int width;
    int height;
};
#endif /* __SLPOINTCLOUD_H__ */
