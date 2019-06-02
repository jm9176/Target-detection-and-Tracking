//Template matching cameraFeed

#include <sstream>
#include <string>
#include <Windows.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/opencv_modules.hpp>
#include <stdio.h>
#include<opencv2/video/tracking.hpp>
#include <ctype.h>

using namespace std;
using namespace cv;

/// Global Variables
Mat img; Mat templ; Mat result;
char* image_window = "Source Image";
char* result_window = "Result window";

int match_method;
int max_Trackbar = 5;

/// Function Headers
void MatchingMethod(int, void*);
Mat cameraFeed;
Mat img_display, gray1, gray2;
/** @function main */
int main(int argc, char** argv)
{
	VideoCapture cap;

	cap.open(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	templ = imread("temp.jpg");
	
	while (1){
		/// Load image and template
		cap.read(cameraFeed);
		cameraFeed.copyTo(img_display);
		
		//Converting the colored images to grayscale
		cvtColor(templ, gray1, COLOR_BGR2GRAY);
		cvtColor(cameraFeed, gray2, COLOR_BGR2GRAY);
		
		/// Create Trackbar
		char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
		createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod);

		MatchingMethod(0, 0);

		waitKey(30);
	}
	return 0;
}

/**
* @function MatchingMethod
* @brief Trackbar callback
*/
void MatchingMethod(int, void*)
{
	/// Create the result matrix
	int result_cols = gray2.cols - gray1.cols + 1;
	int result_rows = gray2.rows - gray1.rows + 1;

	result.create(result_rows, result_cols, CV_32FC1);

	/// Do the Matching and Normalize
	matchTemplate(gray2, gray1, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	/// Show me what you got
	rectangle(img_display, matchLoc, Point(matchLoc.x + gray1.cols, matchLoc.y + gray1.rows), Scalar(0,0,255), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + gray1.cols, matchLoc.y + gray1.rows), Scalar(0,0,255), 2, 8, 0);

	imshow(image_window, img_display);
	imshow(result_window, result);
	return;
}
