
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <stdio.h>
#include <iostream>

#define ThreadsPerBlock 256

void negatywStart(uchar4* input, uchar4* output, int height, int width);

__global__ void NegativeKernel(uchar4* Cuda_in, uchar4*  Cuda_out, int height, int width) {
	const long index = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ uchar4 local_array[ThreadsPerBlock];

	uchar4 const v = Cuda_in[index];
	local_array[threadIdx.x].x = 255 - v.x;
	local_array[threadIdx.x].y = 255 - v.y;
	local_array[threadIdx.x].z = 255 - v.z;
	__syncthreads();
	uchar4 pixel = local_array[(threadIdx.x)];

	//write back the value to global memory
	Cuda_out[index] = pixel;
}

int main()
{
	std::string filePath;
	std::cout << "Œcie¿ka do zdjêcia: ";
	std::cin >> filePath;
	cv::Mat obrazek = cv::imread(filePath, CV_LOAD_IMAGE_UNCHANGED);
	if (obrazek.empty())
	{
		std::cout << "Image Not Found" << std::endl;
		return;
	}
	cv::Mat doEdycji;
	cv::cvtColor(obrazek, doEdycji, CV_BGR2BGRA);

	const int wysokosc = obrazek.rows;
	const int szerokosc = obrazek.cols;
	uchar4 *input = (uchar4*)(doEdycji.data);
	uchar4 *output = new uchar4[wysokosc*szerokosc];
	negatywStart(input, output, wysokosc, szerokosc);
	cv::Mat negatyw = cv::Mat(wysokosc, szerokosc, CV_8UC4, output);

	cv::imshow("Negatyyyw :)", negatyw);
	cv::waitKey(30);
	system("pause");
	return 0;
}

void negatywStart(uchar4* input, uchar4* output, int height, int width) {
	uchar4* Cuda_in;
	uchar4* Cuda_out;
	cudaMalloc((void**)&Cuda_in, sizeof(uchar4) * height*width);
	cudaMalloc((void**)&Cuda_out, sizeof(uchar4) * height*width);
	cudaMemset(Cuda_out, 0, height*width * sizeof(uchar4));
	cudaMemcpy(Cuda_in, input, height*width * sizeof(uchar4), cudaMemcpyHostToDevice);

	int numberOfBlocks = ((height*width) / ThreadsPerBlock);
	dim3 blockSize(ThreadsPerBlock);
	dim3 gridSize(numberOfBlocks);
	NegativeKernel << <gridSize, blockSize >> > (Cuda_in, Cuda_out, height, width);
	cudaDeviceSynchronize();

	cudaMemcpy(output, Cuda_out, height*width * sizeof(uchar4), cudaMemcpyDeviceToHost);
	cudaFree(Cuda_in);
	cudaFree(Cuda_out);

}