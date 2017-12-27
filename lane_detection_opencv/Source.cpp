#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/videoio/videoio.hpp"
#include <iostream>

//#define DEBUG

using namespace cv;
using namespace std;

void laneDetection(Mat & inputImg);

template <class T>
string to_string(T value)
{
   ostringstream os;
   os << value;
   return os.str();
}

template <class T>
void median(vector<T> &vec, T &medVal)
{
	sort(vec.begin(), vec.end());
	medVal = vec[(vec.size() / 2 )];
}

int main()
{
	VideoCapture vid("solidWhiteRight.mp4");
	if (!vid.isOpened())
		{	cout << " !! Check Video Input" << endl; return -1;	 }

	
	Mat frame, grayFrame, binary, edge;
	int nFrames = vid.get(CV_CAP_PROP_FRAME_COUNT);
	int i = 0;
	
	while (i < nFrames)
	{
		i++;
		vid >> frame;
		if (frame.data == NULL) return 0;

		int tic1 = getTickCount();		  
		laneDetection(frame);		
		int tic2 = getTickCount();
		int time = (int)1000.0 * ((1.0*(tic2 - tic1)) / getTickFrequency());

		// display
		putText(frame, "Frame # " + to_string(i), Point(frame.cols*0.8, 20), 
	 				CV_FONT_HERSHEY_COMPLEX_SMALL,	0.9, Scalar(80, 60, 55), 1, 8);

		putText(frame, "Time : " + to_string(time) + " ms" , Point(frame.cols*0.05, 20),
					CV_FONT_HERSHEY_COMPLEX_SMALL, 0.9, Scalar(80, 60, 55), 1, 8);

		imshow("Lane Detection", frame);
		if (waitKey(0) == 27)
			break;
	}
}

void laneDetection(Mat & inputImg)
{
	// variables
	Mat gray = Mat::zeros(inputImg.rows, inputImg.cols, CV_8UC1);
	Rect roi = Rect(0, 0, inputImg.cols, inputImg.rows *0.6);
	Mat binaryImg, edgeImg;
	float slope = 0.0;

	/************ Step 1 : preprocessing ******/
	cvtColor(inputImg, gray, CV_BGR2GRAY);
	//GaussianBlur(gray, gray, Size(15, 15), 0, 0);
	gray(roi).setTo(0);
	threshold(gray, binaryImg, 150, 255, CV_THRESH_BINARY);
	Canny(binaryImg, edgeImg, 30, 100, 3, true);

	/************ Step 2 : line detection and candidate selection ******/
	vector<Vec4i> lines;
	vector<float> slope_left, slope_right;
	vector<float> bias_left,  bias_right;

	HoughLinesP(edgeImg, lines, 1, CV_PI / 180, 1, 15, 5);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];		

		slope = 1.0*(l[1] - l[3]) / (l[0] - l[2] );
		float bias = l[1] - slope * l[0];
		
		if (slope > 0.5 && slope < 0.7)
		{
			slope_right.push_back(slope);
			bias_right.push_back(bias);

#ifdef DEBUG
			line(inputImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
#endif
		}

		if (slope > -0.9 && slope < -0.6)
		{
			slope_left.push_back(slope);	
			bias_left.push_back(bias);

#ifdef DEBUG
			line(inputImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
#endif
		}
	}  

	/*************** Step 3 : Find lanes from candidates ***********/
	float left_slope_med, left_bias_med, right_slope_med, right_bias_med;
	left_slope_med = left_bias_med = right_slope_med = right_bias_med = 0.0;


	// right lane	
	median(slope_right, right_slope_med);
	median(bias_right,  right_bias_med);

	int x1_right = (inputImg.rows - right_bias_med) / right_slope_med;
	int x2_right = (inputImg.rows*0.6 - right_bias_med) / right_slope_med;
	line(inputImg, Point(x2_right,inputImg.rows*0.6), Point(x1_right, inputImg.rows), Scalar(255, 0, 0), 5, 8);
	
	// left lane
	median(slope_left,  left_slope_med);
	median(bias_left,   left_bias_med);

	int x1_left = (inputImg.rows - left_bias_med) / left_slope_med;
	int x2_left = (inputImg.rows*0.6 - left_bias_med) / left_slope_med;
	line(inputImg, Point(x1_left, inputImg.rows), Point(x2_left, inputImg.rows*0.6), Scalar(255, 0, 0), 5, 8);

	 
	/*************** Step 4 : Temporal Smoothing ***********/
	// TODO 	
	





	/*************** Step 5 : Overlay ***********/
	vector<Point> tmp;
	tmp.push_back(Point(x2_right, inputImg.rows*0.6));
	tmp.push_back(Point(x1_right, inputImg.rows));
	tmp.push_back(Point(x1_left, inputImg.rows));
	tmp.push_back(Point(x2_left, inputImg.rows*0.6));

	Mat mask = Mat::zeros(inputImg.rows, inputImg.cols, CV_8UC3);
	
	const Point* elementPoints[1] = { &tmp[0] };
	int numberOfPoints = (int)tmp.size();

	fillPoly(mask, elementPoints, &numberOfPoints, 1, Scalar(0, 255	, 0), 8);
	addWeighted(inputImg, 0.7, mask, 0.3, 0.0, inputImg, -1);

	/*********** clear vectors *************/
	lines.clear();
	slope_left.clear();
	slope_right.clear();
	bias_left.clear();
	bias_right.clear();
	
}
