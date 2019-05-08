/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libelas.
Authors: Andreas Geiger

libelas is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

libelas is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libelas; if not, write to the Free Software Foundation, Inc., 51 Franklin
//Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

// Demo program showing how libelas can be used, try "./elas -h" for help

#include <iostream>
#include <opencv2/opencv.hpp>
#include "elas.h"
#include "image.h"

using namespace std;
using namespace cv;

// compute disparities of pgm image input pair file_1, file_2
void process(Mat l, Mat r, Mat &disl, Mat &disr)
{

  // get image width and height
  int32_t width = l.size().width;
  int32_t height = l.size().height;

  // allocate memory for disparity images
  const int32_t dims[3] = {width, height, width}; // bytes per line = width

  // process
  Elas::parameters param;
  param.postprocess_only_left = false;
  Elas elas(param);
  elas.process(l.data, r.data, (float *)disl.data, (float *)disr.data, dims);

  // find maximum disparity for scaling output disparity images to [0..255]
  float displ_max = 0, dispr_max=0;
  for (int32_t i=0; i<width*height; i++) {
    if (((float*)disl.data)[i]>displ_max) displ_max = ((float*)disl.data)[i];
    if (((float*)disr.data)[i]>dispr_max) dispr_max = ((float*)disr.data)[i];
  }
 // cout << displ_max<<"\n";
  // copy float to uchar
  for (int32_t i = 0; i < width * height; i++)
  {
    ((float *)disl.data)[i] = ((float *)disl.data)[i] / displ_max;
    ((float *)disr.data)[i] = ((float *)disr.data)[i] / dispr_max;
  }

  // save disparity images
}

int main(int argc, char **argv)
{

  // run demo
  cout << "Read video files\n";
  VideoCapture cap("Left_video.avi", 0);
  VideoCapture cap2("Right_video.avi", 0);

  // Check if camera opened successfully
  if (!cap.isOpened() || !cap2.isOpened())
  {
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  float camera_mat_left[3][3] = {462.901520, 0.0, 407.490466, 0.0, 461.090515, 268.767129, 0.0, 0.0, 1.0};
  float camera_mat_right[3][3] = {470.984593, 0.0, 361.474843, 0.0, 472.153861, 249.673963, 0.0, 0.0, 1.0};

  float dist_left[1][5] = {-0.399212, 0.142994, -0.001086, -0.000506, 0.0};
  float dist_right[1][5] = {-0.361678, 0.104504, 0.000347, 0.007105, 0.0};

  float projection_right[] = {433.425654, 0.000000, 474.889399, -42.895992, 0.000000, 433.425654, 256.096243, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000};
  float projection_left[] = {433.425654, 0.000000, 474.889399, 0.000000, 0.000000, 433.425654, 256.096243, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000};

  float rectification_right[] = {0.989720, -0.002812, -0.142992, 0.001587, 0.999961, -0.008679, 0.143011, 0.008363, 0.989686};
  float rectification_left[] = {0.993583, -0.000705, -0.113105, 0.001672, 0.999963, 0.008462, 0.113095, -0.008597, 0.993547};

  Mat intrinsic_left = Mat(3, 3, CV_32FC1, camera_mat_left);
  Mat intrinsic_right = Mat(3, 3, CV_32FC1, camera_mat_right);
  Mat disCoeffs_left = Mat(1, 5, CV_32FC1, dist_left);
  Mat disCoeffs_right = Mat(1, 5, CV_32FC1, dist_right);
  Mat P_left = Mat(3, 4, CV_32FC1, projection_left);
  Mat P_right = Mat(3, 4, CV_32FC1, projection_right);
  Mat R_left = Mat(3, 3, CV_32FC1, rectification_left);
  Mat R_right = Mat(3, 3, CV_32FC1, rectification_right);

  Mat Lmap1, Lmap2, Rmap1, Rmap2;

  Size imgsize = cvSize(752, 480);

  initUndistortRectifyMap(intrinsic_left, disCoeffs_left, R_left, P_left, imgsize, CV_32F, Lmap1, Lmap2);
  initUndistortRectifyMap(intrinsic_right, disCoeffs_right, R_right, P_right, imgsize, CV_32F, Rmap1, Rmap2);

  Mat DispL, DispR;
  Mat lg, rg;
  bool p = false;
  int frame_num=100;
  int tempframe=0;
  while (1)
  { 
    tempframe++;

    Mat frameL;
    Mat frameR;
    // Capture frame-by-frame
    cap >> frameL;
    cap2 >> frameR;
    cvtColor(frameL, lg, COLOR_RGB2GRAY);
    cvtColor(frameR, rg, COLOR_RGB2GRAY);
    equalizeHist( frameL, frameL );
    equalizeHist( frameR, frameR );
    if (!p)
    {
      DispL = Mat(frameL.size().height, frameL.size().width, CV_32FC1);
      DispR = Mat(frameL.size().height, frameL.size().width, CV_32FC1);
      p=1;
    }
    // If the frame is empty, break immediately
    if (frameL.empty() || frameR.empty())
      break;

    remap(lg, lg, Lmap1, Lmap2, INTER_LINEAR);
    remap(rg, rg, Rmap1, Rmap2, INTER_LINEAR);

    process(lg, rg, DispL, DispR);
    // DisplaLy the resulting frame
    imshow("Leftv", lg);
    imshow("Rightv", rg);
    imshow("Leftd", DispL);
    imshow("Rightd", DispR);
    if(tempframe==frame_num)
    {
      imwrite("left.png",frameL);
      imwrite("right.png",frameR);
    }
    // Press  ESC on keyboard to exit
    char c = (char)waitKey(25);
    if (c == 27)
      break;
  }

  // When everything done, release the video capture object
  cap.release();

  // Closes all the frames
  destroyAllWindows();

  return 0;
}
