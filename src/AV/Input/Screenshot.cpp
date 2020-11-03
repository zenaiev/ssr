#include <Screenshot.h>
#include <opencv2/opencv.hpp>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

int64_t Screenshot::timestamp_last = 0;
int64_t Screenshot::timestamp_last_pressed = 0;

bool Screenshot::key_is_pressed(KeySym ks) {
    Display *dpy = XOpenDisplay(":0");
    char keys_return[32];
    XQueryKeymap(dpy, keys_return);
    KeyCode kc2 = XKeysymToKeycode(dpy, ks);
    bool isPressed = !!(keys_return[kc2 >> 3] & (1 << (kc2 & 7)));
    XCloseDisplay(dpy);
    return isPressed;
}

void Screenshot::do_screenshot(unsigned int h, unsigned int w, uint8_t *image_data, int64_t timestamp)
{
  if(!key_is_pressed(XK_Alt_L))
    return;
  if((timestamp - timestamp_last) < timestamp_gap)
    return;
  if(timestamp_last_pressed == 0)
  {
    timestamp_last_pressed = timestamp;
    return;
  }
  if((timestamp - timestamp_last_pressed) < timestamp_gap_pressed)
    return;
  printf("SZ timestamp: %ld gap: %ld pressed: %ld\n", timestamp, timestamp - timestamp_last, timestamp - timestamp_last_pressed);
  timestamp_last_pressed = 0;
  timestamp_last = timestamp;
  //static int nframe = 0;
  //nframe++;
  const char* path = "/home/zenaiev/games/Diablo2/502/screens/";
  //int ret = this->save_frame_as_jpeg(GetCodecContext(), packet->GetPacket(), nframe, path);
  cv::Mat img = cv::Mat(h, w, CV_8UC4, image_data);
  char fname[256];
  sprintf(fname, "%s/%ld.png", path, timestamp);
  cv::imwrite(fname, img);
  printf("SZ pushed image = %s\n", fname);
  int ret = to_python(h, w, img, timestamp);
  printf("SZ to_python ret = %d\n", ret);
}

int Screenshot::to_python(unsigned int h, unsigned int w, cv::Mat& mat1, int64_t timestamp)
{
  int row = 0;
  float *p = mat1.ptr<float>(row);

  printf("SZ to_python()\n");
  //std::cout << "Mat" << mat1 <<std::endl;

  PyObject *pName, *pModule, *pDict, *pFunc, *pArgs;
  npy_intp dims[2] = { h, w };
  PyObject *py_array;

  setenv("PYTHONPATH",".",1);
  Py_Initialize ();
  //pName = PyUnicode_FromString ("/home/zenaiev/games/Diablo2/ssr/pycode.py");
  pName = PyUnicode_FromString ("pycode");
  
  pModule = PyImport_Import(pName);

  pDict = PyModule_GetDict(pModule);
  //printf("dupa\n");

  // Required for the C-API : http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
  static int ncall = 0;
  ncall++;
  if(ncall == 1)
    import_array ();

  py_array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, p);

  pArgs = PyTuple_New (2);
  PyTuple_SetItem (pArgs, 0, py_array);
  PyTuple_SetItem (pArgs, 1, PyLong_FromLong(timestamp));

  pFunc = PyDict_GetItemString (pDict, (char*)"pyArray"); 

  if (PyCallable_Check (pFunc))
  {
      PyObject_CallObject(pFunc, pArgs);
  } else
  {
      std::cout << "Function is not callable !" << std::endl;
  }

  Py_DECREF(pName);
  Py_DECREF (py_array);                             
  Py_DECREF (pModule);
  Py_DECREF (pDict);
  Py_DECREF (pFunc);

  Py_Finalize ();                                  

  return 0;
}
