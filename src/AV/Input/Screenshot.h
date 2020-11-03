#include <opencv2/stitching.hpp>
#include <X11/Xlib.h>
#include <iostream>
#include "X11/keysym.h"

class Screenshot
{
  public:
    static void do_screenshot(unsigned int h, unsigned int w, uint8_t *image_data, int64_t timestamp);
  private:
    static int64_t timestamp_last;
    static int64_t timestamp_last_pressed;
    //constexpr static int64_t timestamp_gap_pressed = 50000;
    constexpr static int64_t timestamp_gap_pressed = 1;
    constexpr static int64_t timestamp_gap = 1000000;
    static bool key_is_pressed(KeySym ks);
    static int to_python(unsigned int h, unsigned int w, cv::Mat& img, int64_t timestamp);
};
