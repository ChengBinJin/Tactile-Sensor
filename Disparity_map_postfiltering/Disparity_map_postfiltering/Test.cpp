#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("../../img/cones_imL.png");

	namedWindow("Test Opencv", WINDOW_NORMAL);
	imshow("Test Opencv", img);
	waitKey(0);

	return 0;

}