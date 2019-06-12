// Color-based Target tracking
//initialising the header files
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/highgui.h>


using namespace cv;
using namespace std;


// Function to apply the morphological operations
void morph_ft(Mat &thresh, int num_iter){
	
	/*
	// Creating a self-defined shape for the structuring element
	Mat erosion_st_elem = (Mat_<uchar>(4,4) << 0, 1, 1, 0,
						   1, 1, 1, 1,
						   1, 1, 1, 1,
						   0, 1, 1, 0);
	*/
	
	// Creating the structuring elements for the erosion and dilation
	// process of the morphological operations. 
	Mat erosion_st_elem = getStructuringElement(MORPH_RECT, Size(13,13));
	Mat dilate_st_elem = getStructuringElement(MORPH_RECT, Size(17,17));

	for (int i = 0; i < num_iter; i++){
		erode(thresh, thresh, erosion_st_elem);
	}

	for (int i = 0; i < num_iter; i++){
		dilate(thresh, thresh, dilate_st_elem);
	}
	
}


// Function to track the thresholded targets in the image plane
void track_target(float &x, float &y, Mat &thresh){

	// Defining the min area value for the tracking.
	// If yes, then target is found, else, it is considered
	// as noise.
	int min_objArea = 400;

	// Calculating Image moments to get the centroid
	// and the area of the white pixelated region in
	// the threshold image
	Moments moment1 = moments(thresh);
	double area1 = moment1.m00;

	// Notations for the moments
	// m00 = area
	// m10/area = center_x
	// m01/area = center_y

	if (area1>min_objArea){
		x = (moment1.m10 / area1);
		y = (moment1.m01 / area1);
	}

	else {
		x = 0;
		y = 0;
	}

}


// Main function
int main(int argc, char** argv[]){

	// Defining the frame width and the height of the 
	// output window
	int frameHeight = 480;
	int frameWidth = 640;

	// Initializing variables for the target tracking
	float x, y = 0;

	// Creating the Image objects
	Mat cameraFeed, HSV_img, thresh_img;

	// Defining the number of iterations for the
	// morphological operations
	int num_iter = 2;

	// Initializing the capturing object to 
	// be used to read the camera frames
	VideoCapture cap;
	cap.open(0);

	// Defining the image output window size
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, frameHeight);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, frameWidth);

	while (1){
	
		// Updating the frames 
		cap.read(cameraFeed);

		// Converting the BGR image to HSV
		cvtColor(cameraFeed, HSV_img, COLOR_BGR2HSV);

		// Generating a threshold image for a given color range
		inRange(HSV_img, Scalar(0, 120, 100), Scalar(10, 255, 255), thresh_img);

		// Applying the morphological filters
		morph_ft(thresh_img, num_iter);

		// Tracking the white pixelated area
		track_target(x, y, thresh_img);
		circle(cameraFeed, Point(x, y), 3, Scalar(0, 0, 255), 3, 8, 0);

		// Generating the image outputs
		imshow("BGR Image", cameraFeed);
		imshow("HSV Image", HSV_img);
		imshow("Threshold Image", thresh_img);
	
		waitKey(1);
	
	}
	
	return 0;
}
