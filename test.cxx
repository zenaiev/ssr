// g++ test.cxx -o test `python3-config --cflags --ldflags` -no-pi
// ./test

#include <Python.h>
#include <numpy/arrayobject.h>

int main()
{
  PyObject *pName = nullptr;
  PyObject *pModule = nullptr;
  PyObject *pDict = nullptr;
  PyObject *pFunc = nullptr;
  printf("dupa00\n");
  //setenv("PYTHONPATH","/home/zenaiev/soft/root-6.22.02-bin/lib:/home/zenaiev/soft/deps/lhapdf/lib/python2.7/site-packages:/home/zenaiev/soft/deps/lhapdf/lib/python2.7/site-packages",1);
  setenv("PYTHONPATH",".",1);
  Py_Initialize ();
  pName = PyUnicode_FromString ("pycode_tmp");
  //pName = PyUnicode_FromString ("pycode_tmp");
  pModule = PyImport_Import(pName);
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString (pDict, (char*)"py_droprec"); 
  // Required for the C-API : http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
  import_array();
  return 0;
}
