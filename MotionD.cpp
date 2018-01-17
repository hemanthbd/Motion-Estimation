#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

char Cameraframe(Mat,int);

vector<Point2f> points[2];
std::vector<KeyPoint> kp;
std::vector<KeyPoint> kp1;
std::vector<Point2f>point1; 


Mat gray, gray1,prevGray, image, frame, mask,dst;
char c;

int main()
{ 
int count=0;

namedWindow("HEYO", CV_WINDOW_AUTOSIZE);

cv::String path( "/home/owner/Desktop/Opencv_DIP/CamSeq01/*.png"); 
vector<cv::String> fn;
vector<cv::Mat> data;
cv::glob(path,fn,true); // recurse
for (size_t k=0; k<fn.size(); k=k+2)
{
 cv::Mat im = cv::imread(fn[k]);

 Cameraframe(im,count);

if( c == 27 )
   break;

count++;
}
   return 0;
}


char Cameraframe(Mat im,int count)
{ 
TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
Size winSize(20,20);
std::vector<KeyPoint> keypoints;
std::vector<KeyPoint> keypoints1;    
double alpha = 0.7, beta; 

int nfeatures=500,nlevels=8,edgeThreshold=10,firstLevel=0,WTA_K=3,scoreType=ORB::HARRIS_SCORE;
int patchSize=15,fastThreshold=10;
float scaleFactor=2.0f;
    
im.copyTo(frame);
frame.copyTo(image);
cvtColor(image, gray, COLOR_BGR2GRAY);

int rows= gray.rows;
int cols= gray.cols;

mask = Mat::ones(rows, cols, CV_8UC3);

Ptr<ORB> detector = ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold );

if(count==0)
{
detector->detect(gray, kp);

KeyPoint::convert(kp, points[1]);

/*for(int j=0;j<kp.size();j++)
{ cout<<points[0][j]<<"\n";
}
*/
imshow("HEYO", image);
c = (char)waitKey(50);
}

else if( !points[0].empty() )
{
  if (count%3==0)
{ 
detector->detect(prevGray, kp1);
KeyPoint::convert(kp1, point1);

for( size_t i1 = 0; i1 < points[0].size(); i1++ ) {
  keypoints.push_back(KeyPoint(points[0][i1], 1.f));
}

for( size_t i2 = 0; i2 < point1.size(); i2++ ) {
  keypoints.push_back(KeyPoint(point1[i2], 1.f));
}

KeyPoint::convert(keypoints, points[0]);
}

cout<<"SIZE"<<points[0].size()<<"\n";

if(points[0].size()>500)
{ 
for(int r=0;r<500;r++)
keypoints1.push_back(KeyPoint(points[0][r], 1.f));

KeyPoint::convert(keypoints1, points[0]);
}

vector<uchar> status;
vector<float> err;

calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,3,termcrit, 0, 0.001);

size_t i, k,m;
for( i = k = m= 0; i < points[1].size(); i++ )
{
if( !status[i] )
  continue;
{points[1][k++] = points[1][i];
points[0][m++]= points[0][i];}
               
circle(image, points[1][i], 2, Scalar(0,255,0), -1, 8);
}
points[1].resize(k);
points[0].resize(m);

for(int j=0; j<points[0].size(); j++)
{ 
line(image, points[0][j], points[1][j], Scalar(0,255,0), 2, 4, 0);
}


imshow("HEYO", image);
c = (char)waitKey(50);
}

std::swap(points[1], points[0]);
cv::swap(gray, prevGray);

count++;

return c;

}




