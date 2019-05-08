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
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

// Demo program showing how libelas can be used, try "./elas -h" for help

#include <iostream>
#include <opencv2/opencv.hpp>
#include "elas.h"
#include "image.h"

using namespace std;
using namespace cv;

// compute disparities of pgm image input pair file_1, file_2
void process (Mat l, Mat r, Mat& disl, Mat& disr) {

  // get image width and height
  int32_t width  = l.size().width;
  int32_t height = l.size().height;

  // allocate memory for disparity images
  const int32_t dims[3] = {width,height,width}; // bytes per line = width

  // process
  Elas::parameters param;
  param.postprocess_only_left = false;
  Elas elas(param);
  elas.process(l.data, r.data, (float*)disl.data, (float*)disr.data, dims);
  

  // find maximum disparity for scaling output disparity images to [0..255]
  /*float displ_max = 0, dispr_max=0;
  for (int32_t i=0; i<width*height; i++) {
    if (((float*)disl.data)[i]>displ_max) displ_max = ((float*)disl.data)[i];
    if (((float*)disr.data)[i]>dispr_max) dispr_max = ((float*)disr.data)[i];
  }
  cout << displ_max<<"\n";*/
  // copy float to uchar
  for (int32_t i=0; i<width*height; i++) {
    ((float*)disl.data)[i] = ((float*)disl.data)[i]/255.0;
    ((float*)disr.data)[i] = ((float*)disr.data)[i]/255.0;
  }

  // save disparity images
}

int main (int argc, char** argv) {

  // run demo
  cout << "Read video files\n";
  VideoCapture cap("Left_video.avi"); 
  VideoCapture cap2("Right_video.avi"); 
    
  // Check if camera opened successfully
  if(!cap.isOpened() || !cap2.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  Mat DispL, DispR;
  bool p = false;
  while(1){
 
    Mat frameL;
    Mat frameR;
    // Capture frame-by-frame
    cap >> frameL;
    cap2 >> frameR;
    if(!p)
    {
      DispL = Mat(frameL.size().height, frameL.size().width, CV_32FC1);
      DispR = Mat(frameL.size().height, frameL.size().width, CV_32FC1);
    }
    // If the frame is empty, break immediately
    if (frameL.empty() || frameR.empty())
      break;
    process(frameL, frameR, DispL, DispR);
    // DisplaLy the resulting frame
    imshow( "Leftv", frameL );
    imshow( "Rightv", frameR );
    imshow( "Leftd", DispL );
    imshow( "Rightd", DispR );
 
    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;
  }
  
  // When everything done, release the video capture object
  cap.release();
 
  // Closes all the frames
  destroyAllWindows();


  return 0;
}


