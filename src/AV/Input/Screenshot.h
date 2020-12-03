#pragma once
#include "Global.h"
//#include <opencv2/stitching.hpp>
//#include <X11/Xlib.h>
//#include <iostream>
//#include "X11/keysym.h"
// prevent compilation error https://github.com/NixOS/nixpkgs/issues/42811
#ifdef slots
#undef slots
#endif
#include <Python.h>

class Screenshot
{
  public:
    static void check_screenshot(unsigned int h, unsigned int w, uint8_t *image_data, int64_t timestamp);
  private:
    constexpr static int64_t timestamp_gap_pressed = 100;
    //constexpr static int64_t timestamp_gap = 10000000;
    constexpr static int64_t timestamp_gap = 1000000;
    constexpr static int verbosity = 1;

    static int64_t timestamp_last;
    static int64_t timestamp_last_pressed;
    static PyObject *pName;
    static PyObject *pModule;
    static PyObject *pDict;
    static PyObject *pFunc;
    static bool init;

    static int do_screenshot(unsigned int h, unsigned int w, uint8_t *image_data, int64_t timestamp);
    static bool key_is_pressed(KeySym ks);
    static int to_python(unsigned int h, unsigned int w, uint8_t* image_data, int64_t timestamp);
};
