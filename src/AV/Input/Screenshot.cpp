#include <opencv2/opencv.hpp>
#include "Screenshot.h"
#include "Logger.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <ctime>

int64_t Screenshot::timestamp_last = 0;
int64_t Screenshot::timestamp_last_pressed = 0;
bool Screenshot::init = 0;
PyObject *Screenshot::pName = nullptr;
PyObject *Screenshot::pModule = nullptr;
PyObject *Screenshot::pDict = nullptr;
PyObject *Screenshot::pFunc = nullptr;
    
bool Screenshot::key_is_pressed(KeySym ks) {
    Display *dpy = XOpenDisplay(":0");
    char keys_return[32];
    XQueryKeymap(dpy, keys_return);
    KeyCode kc2 = XKeysymToKeycode(dpy, ks);
    bool isPressed = !!(keys_return[kc2 >> 3] & (1 << (kc2 & 7)));
    XCloseDisplay(dpy);
    return isPressed;
}

void Screenshot::check_screenshot(unsigned int h, unsigned int w, uint8_t *image_data, int64_t timestamp)
{
  const clock_t begin_time = clock();
  int ret = do_screenshot(h, w, image_data, timestamp);

  char msg[256];
  if (verbosity > 1)
    Logger::LogInfo(std::string("SZ took " + std::to_string(1000 * float( clock () - begin_time ) /  CLOCKS_PER_SEC) + std::string(" ms")).c_str());
  else if (verbosity == 1 && ret == 0) // screenshot done
    Logger::LogInfo(std::string("SZ took " + std::to_string(1000 * float( clock () - begin_time ) /  CLOCKS_PER_SEC) + std::string(" ms")).c_str());
  //if(ret == 0)
  //  std::cout << "SZ took " << 1000 * float( clock () - begin_time ) /  CLOCKS_PER_SEC << " ms" << std::endl;
}

int Screenshot::do_screenshot(unsigned int h, unsigned int w, uint8_t *image_data, int64_t timestamp)
{
  if(!key_is_pressed(XK_Alt_L))
    return 1;
  if((timestamp - timestamp_last) < timestamp_gap)
    return 1;
  if(timestamp_last_pressed == 0)
  {
    timestamp_last_pressed = timestamp;
    return 1;
  }
  if((timestamp - timestamp_last_pressed) < timestamp_gap_pressed)
    return 1;
  printf("SZ timestamp: %ld gap: %ld pressed: %ld\n", timestamp, timestamp - timestamp_last, timestamp - timestamp_last_pressed);
  timestamp_last_pressed = 0;
  timestamp_last = timestamp;
  int ret = to_python(h, w, image_data, timestamp);
  return ret;
  //printf("SZ to_python ret = %d\n", ret);
}

int Screenshot::to_python(unsigned int h, unsigned int w, uint8_t* image_data, int64_t timestamp)
{
  //return 0;
  if(init == 0)
  {
    printf("dupa00\n");
    init = 1;
    setenv("PYTHONPATH",".",1);
    //setenv("PYTHON3PATH",".",1);
    Py_Initialize ();
    pName = PyUnicode_FromString ("pycode");
    //pName = PyUnicode_FromString ("pycode_tmp");
    pModule = PyImport_Import(pName);
    pDict = PyModule_GetDict(pModule);
    pFunc = PyDict_GetItemString (pDict, (char*)"py_droprec"); 
    // Required for the C-API : http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
    import_array();
  }

  printf("dupa0\n");
  const char* path = "/home/zenaiev/games/Diablo2/502/screens/";
  //int ret = this->save_frame_as_jpeg(GetCodecContext(), packet->GetPacket(), nframe, path);
  cv::Mat img = cv::Mat(h, w, CV_8UC4, image_data);
  //std::cout << img.at<uchar>(0, 0) << std::endl;
  //std::cout << img << std::endl;
  /*char fname[256];
  sprintf(fname, "%s/%ld.png", path, timestamp);
  cv::imwrite(fname, img);
  printf("SZ pushed image = %s\n", fname);*/

  int row = 0;
  //float *p = img.ptr<float>(row);
  uchar *p = img.ptr<uchar>(row);
  //npy_intp dims[2] = { h, w };
  npy_intp dims[3] = { h, w, 4 };
  //PyObject *py_array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, p);
  PyObject *py_array = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, p);
  PyObject *pArgs = PyTuple_New (3);
  PyTuple_SetItem (pArgs, 0, PyUnicode_FromString ("newrun"));
  PyTuple_SetItem (pArgs, 1, py_array);
  PyTuple_SetItem (pArgs, 2, PyLong_FromLong(timestamp));

  printf("dupa\n");
  if (PyCallable_Check (pFunc))
  {
    PyObject *py_ret = PyObject_CallObject(pFunc, pArgs);
    //std::cout << PyObject_Repr(py_ret) << std::endl;
    //PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    PyObject* py_str = PyUnicode_AsEncodedString(py_ret, "utf-8", "~E~");
    const char *s = PyBytes_AS_STRING(py_str);
    printf("s = %s\n", s);
    Logger::LogDrop(s);
    //Logger::LogError(s);
    Py_DECREF (py_ret);
    Py_DECREF (py_str);
  }
  else
    std::cout << "Function is not callable !" << std::endl;

  //Py_DECREF(pName);
  Py_DECREF (py_array);                             
  //Py_DECREF (pModule);
  //Py_DECREF (pDict);
  //Py_DECREF (pFunc);

  //Py_Finalize ();                                  

  return 0;
}
