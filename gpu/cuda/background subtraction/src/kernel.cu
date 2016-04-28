#include "kernel.cuh"

/*________________________________________________________* 
*														  *
*   		CUDA KERNELS AND ASSOCIATED FUNCTIONS		  *
*														  *
*_________________________________________________________*/

// device Kernel (can only be call by a kernel) :: define the modulo operation
inline __device__ int modulo(int val, int c){
	return (val & (c - 1));
}

// Kernel :: fill an image with a chekcerboard pattern
__global__ void _checkerboard(uchar4 *image, unsigned int width, unsigned int height, unsigned int imStep)
{
	// get the position of the current pixel
	int x_local = blockIdx.x * blockDim.x + threadIdx.x;
	int y_local = blockIdx.y * blockDim.y + threadIdx.y;

	// exit if the pixel is out of the size of the image
	if (x_local >= width || y_local >= height) return;
	
	// define the two colors of the checkerboard
	uchar4 color1 = make_uchar4(250, 250, 250, 255);
	uchar4 color2 = make_uchar4(236, 185, 25, 255);

	// define the size of the square
	int step = 14;
	int step_2 = step / 2;

	// fill the image, alternate the colors
	if (modulo(x_local, step) < step_2){
		if (modulo(y_local, step) < step_2)
			image[y_local * imStep + x_local] = color1;
		else
			image[y_local * imStep + x_local] = color2;
	}else{
		if (modulo(y_local, step) < step_2)
			image[y_local * imStep + x_local] = color2;
		else
			image[y_local * imStep + x_local] = color1;
	}
}

// Function :: fill an image with a chekcerboard pattern
void cuCreateCheckerboard(sl::zed::Mat &image)
{
	// get the image size
	unsigned int width = image.width;
	unsigned int height = image.height;

	// define the block dimension for the parallele computation
	dim3 dimGrid, dimBlock;
	dimBlock.x = 32;
	dimBlock.y = 8;

	dimGrid.x = ceill(width / (float)dimBlock.x);
	dimGrid.y = ceill(height / (float)dimBlock.y);

	// call the kernel
	_checkerboard << <dimGrid, dimBlock >> >((uchar4 *)image.data, width, height, image.step / sizeof(uchar4));
}

// Kernel :: replace the current image by an other if the depth if above the threshold
__global__ void _croppImage(float* depth, uchar4 * imageIn, uchar4 * imageOut, uchar4 * mask, float threshold,
	unsigned int width, unsigned int height, unsigned int depthStep, unsigned int imInStep, unsigned int imOutStep, unsigned int maskStep)
{
	// get the position of the current pixel
	int x_local = blockIdx.x * blockDim.x + threadIdx.x;
	int y_local = blockIdx.y * blockDim.y + threadIdx.y;

	// exit if the pixel is out of the size of the image
	if (x_local >= width || y_local >= height) return;

	// get the depth of the current pixel
	float D = depth[y_local * depthStep + x_local];

	// the depth is strickly positive, if not it means that the depth can not be computed
	// the depth should be below the threshold	
	if ((D > 0) && (D < threshold))// keep the current image if true
		imageOut[y_local * imOutStep + x_local] = imageIn[y_local * imInStep + x_local];
	else // if false : replace current pixel by the pixel of the mask
		imageOut[y_local * imOutStep + x_local] = mask[y_local * maskStep + x_local];
}

// Function :: replace the current image by an other if the depth if above the threshold
void cuCroppImageByDepth(sl::zed::Mat &depth, sl::zed::Mat &imageLeft, sl::zed::Mat &imageCut, sl::zed::Mat &mask, float threshold)
{
	// get the image size
	unsigned int width = depth.width;
	unsigned int height = depth.height;

	// define the block dimension for the parallele computation
	dim3 dimGrid, dimBlock;
	dimBlock.x = 32;
	dimBlock.y = 8;

	dimGrid.x = ceill(width / (float)dimBlock.x);
	dimGrid.y = ceill(height / (float)dimBlock.y);

	// convert the threshold in mm
	threshold *= 1000.;

	// call the kernel
	_croppImage << <dimGrid, dimBlock >> >((float *)depth.data, (uchar4 *)imageLeft.data, (uchar4 *)imageCut.data, (uchar4 *)mask.data, threshold, width, height,
		depth.step / sizeof(float), imageLeft.step / sizeof(uchar4), imageCut.step / sizeof(uchar4), mask.step / sizeof(uchar4));
}