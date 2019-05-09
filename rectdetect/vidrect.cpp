// Copyright Naoki Shibata 2018. Distributed under the MIT License.

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <time.h>
#include <string>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace std;
using namespace cv;

#include "vec234.h"

#include "helper.h"
#include "oclhelper.h"

#include "oclimgutil.h"
#include "oclpolyline.h"
#include "oclrect.h"

#include "../src/elas.h"
Mat Lmap1, Lmap2, Rmap1, Rmap2;
Mat DispL, DispR;
float displ_max = 0, dispr_max = 0;
int min_mean = 0, min_variance = 0;

// compute disparities of pgm image input pair file_1, file_2
void process(Mat l, Mat r)
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
  elas.process(l.data, r.data, (float *)DispL.data, (float *)DispR.data, dims);

  // find maximum disparity for scaling output disparity images to [0..255]
  for (int32_t i = 0; i < width * height; i++)
  {
    if (((float *)DispL.data)[i] > displ_max)
      displ_max = ((float *)DispL.data)[i];
    if (((float *)DispR.data)[i] > dispr_max)
      dispr_max = ((float *)DispR.data)[i];
  }
  // cout << displ_max<<"\n";
  // copy float to uchar
  for (int32_t i = 0; i < width * height; i++)
  {
    ((float *)DispL.data)[i] = ((float *)DispL.data)[i] / displ_max;
    ((float *)DispR.data)[i] = ((float *)DispR.data)[i] / dispr_max;
  }

  // save disparity images
}
bool check_angle(Point a, Point b, Point c)
{
  float ag = abs(atan2(abs(b.y - a.y), abs(b.x - a.x)) - atan2(abs(c.y - a.y) , abs(c.x - a.x)));
  return ag > 3.14 / 4 && ag < 3.14 / 2;
}
bool checkRect(Rect a, rect_t c)
{
  if (a.height * a.width < 1600)
    return 0;
  for (int i = 1; i <= 4; i++)
  {
    bool ag = check_angle(Point(c.c2[i % 4].a[0], c.c2[i % 4].a[1]), Point(c.c2[(i + 1) % 4].a[0], c.c2[(i + 1) % 4].a[1]), Point(c.c2[(i - 1) % 4].a[0], c.c2[(i - 1) % 4].a[1]));
    if (!ag)
      return false;
  }
  return true;
}

static void showRect(rect_t rect, int r, int g, int b, int thickness, Mat &img)
{
  int font = FONT_HERSHEY_SCRIPT_SIMPLEX;
  std::ostringstream ss;
  float d = DispL.at<float>(cvPoint(rect.c2[0].a[0], rect.c2[0].a[1]));
  if (d < 0)
    return;
  d *= displ_max;

  Rect rec(cvPoint(rect.c2[1].a[0], rect.c2[1].a[1]), cvPoint(min(752, (int)rect.c2[3].a[0]), min(480, (int)rect.c2[3].a[1])));
  if (rec.height == 0 || rec.width == 0 || rec.x < 0 || rec.y < 0)
    return;
  if (rec.height * rec.width < 1600)
    return;

    if(!checkRect(rec, rect)) return;
  Mat ROI = DispL(rec);

  Scalar mean, dev;
  // ss << "h";
  meanStdDev(ROI, mean, dev);
  if ((int)(mean[0] * displ_max) < min_mean || mean[0] < 0 || dev[0] * displ_max < 3)
    return;
  //cout << min_mean - mean[0]*displ_max << "\n";
  ss << dev[0] * displ_max << " h " << mean[0] * displ_max;
  std::string s(ss.str());
  for (int i = 0; i < 4; i++)
  {
    line(img, cvPoint(rect.c2[i].a[0], rect.c2[i].a[1]), cvPoint(rect.c2[(i + 1) % 4].a[0], rect.c2[(i + 1) % 4].a[1]), Scalar(r, g, b), thickness, 8, 0);
  }

  line(img,
       cvPoint(rect.c2[0].a[0], rect.c2[0].a[1]),
       cvPoint(rect.c2[2].a[0], rect.c2[2].a[1]), Scalar(r, g, b), 1, 8, 0);

  line(img,
       cvPoint(rect.c2[1].a[0], rect.c2[1].a[1]),
       cvPoint(rect.c2[3].a[0], rect.c2[3].a[1]), Scalar(r, g, b), 1, 8, 0);

  putText(img, s.c_str(), cvPoint(rect.c2[3].a[0], rect.c2[3].a[1]), font, 1, (255, 255, 255), 2);
}

static int fourcc(const char *s)
{
  return (((uint32_t)s[0]) << 0) | (((uint32_t)s[1]) << 8) | (((uint32_t)s[2]) << 16) | (((uint32_t)s[3]) << 24);
}

void recitfy()
{

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

  Size imgsize = cvSize(752, 480);

  initUndistortRectifyMap(intrinsic_left, disCoeffs_left, R_left, P_left, imgsize, CV_32F, Lmap1, Lmap2);
  initUndistortRectifyMap(intrinsic_right, disCoeffs_right, R_right, P_right, imgsize, CV_32F, Rmap1, Rmap2);
  DispL = Mat(imgsize.height, imgsize.width, CV_32FC1);
  DispR = Mat(imgsize.height, imgsize.width, CV_32FC1);
}

int main(int argc, char **argv)
{
  recitfy();
  //
  VideoCapture *cap = new VideoCapture("Left_video.avi");
  VideoCapture *capR = new VideoCapture("Right_video.avi");

  int iw = cap->get(CV_CAP_PROP_FRAME_WIDTH);
  int ih = cap->get(CV_CAP_PROP_FRAME_HEIGHT);

  printf("Resolution : %d x %d\n", iw, ih);

  VideoWriter *writer = NULL;
  const char *winname = "Rectangle Detection Demo";

  namedWindow(winname, WINDOW_NORMAL);
  writer = new VideoWriter("out_left.avi", fourcc("PIM1"), 30, cvSize(iw, ih), true);
  VideoWriter *writerR = new VideoWriter("out_right.avi", fourcc("PIM1"), 30, cvSize(iw, ih), true);
  if (!writer->isOpened() || !writerR->isOpened())
  {
    fprintf(stderr, "Cannot open %s\n", argv[3]);
    exit(-1);
  }

  //

  double aov = 90;

  int did = 0;

  printf("Horizontal angle of view : %g degrees\n", aov);

  cl_device_id device = simpleGetDevice(did);
  printf("%s\n", getDeviceName(device));
  cl_context context = simpleCreateContext(device);

  cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

  if (writer == NULL)
    printf("\n>>>>> Press ENTER on the window to exit <<<<<\n");

  oclimgutil_t *oclimgutil = init_oclimgutil(device, context);
  oclpolyline_t *oclpolyline = init_oclpolyline(device, context);
  oclrect_t *oclrect = init_oclrect(oclimgutil, oclpolyline, device, context, queue, iw, ih);

  const double tanAOV = tan(aov / 2 / 180.0 * M_PI);
  Mat vimg, rimg, img[2];

  int nFrame = 0, lastNFrame = 0;
  uint64_t tm = currentTimeMillis();

  cap->grab();
  {
    cap->retrieve(vimg, 0);
    assert(vimg.channels() == 3);
    img[0] = vimg.clone();
    img[1] = vimg.clone();

    uint8_t *data = (uint8_t *)img[nFrame & 1].data;
    int ws = img[nFrame & 1].step;
    oclrect_enqueueTask(oclrect, data, ws);
    cap->grab();

    nFrame++;
  }

  capR->grab();
  capR->grab();
  Size imgsize = cvSize(752, 480);

  Mat l2img(imgsize, CV_8UC1);
  Mat r2img(imgsize, CV_8UC1);
  createTrackbar("mean", winname, &min_mean, 100);

  for (;;)
  {
    if (!cap->retrieve(vimg, 0) || !capR->retrieve(rimg, 0))
      break;
    remap(vimg, vimg, Lmap1, Lmap2, INTER_LINEAR);
    //
    remap(rimg, rimg, Rmap1, Rmap2, INTER_LINEAR);
    //equalizeHist(rimg, rimg);

    cvtColor(vimg, l2img, COLOR_RGB2GRAY);
    equalizeHist(l2img, l2img);

    //Canny( vimg, vimg, 25, 150, 3 );

    cvtColor(rimg, r2img, COLOR_RGB2GRAY);
    equalizeHist(r2img, r2img);
    //Canny( rimg, rimg, 25, 150, 3 );

    process(l2img, r2img);

    vimg.copyTo(img[nFrame & 1]);

    uint8_t *data = (uint8_t *)img[nFrame & 1].data;
    int ws = img[nFrame & 1].step;

    oclrect_enqueueTask(oclrect, data, ws);

    cap->grab();
    capR->grab();
    nFrame++;
    rect_t *ret = oclrect_pollTask(oclrect, tanAOV);

    for (int i = 1; i < ret->nItems; i++)
    { // >>>> This starts from 1 <<<<
      switch (ret[i].status)
      {
      case 0:
        showRect(ret[i], 0, 255, 0, 1, img[nFrame & 1]);
        break;
      case 2:
        showRect(ret[i], 255, 0, 0, 1, img[nFrame & 1]);
        break;
      case 1:
        showRect(ret[i], 0, 200, 255, 2, img[nFrame & 1]);
        break;
      case 3:
        showRect(ret[i], 0, 0, 255, 2, img[nFrame & 1]);
        break;
      }
    }
    writer->write(img[nFrame & 1]);

    uint64_t t = currentTimeMillis();
    if (t - tm > 1000)
    {
      printf("%.3g fps\n", 1000.0 * (nFrame - lastNFrame) / ((double)(t - tm)));
      tm = t;
      lastNFrame = nFrame;
    }

    imshow(winname, img[nFrame & 1]);
    imshow("dis", DispL);
    int key = waitKey(1) & 0xff;
    if (key == 27 || key == 13)
      break;
  }

  //

  dispose_oclrect(oclrect);
  dispose_oclpolyline(oclpolyline);
  dispose_oclimgutil(oclimgutil);

  ce(clReleaseCommandQueue(queue));
  ce(clReleaseContext(context));

  if (writer != NULL)
    delete writer;
  delete cap;
  delete capR;
  destroyAllWindows();

  //

  exit(0);
}
