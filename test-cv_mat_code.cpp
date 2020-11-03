// g++ -g -fPIC test-cv_mat_code.cpp -o ./test-cv_mat_code -lpython3.6m  -I/usr/include/python3.6m/  -I/usr/include/ -lopencv_core -lopencv_imgproc -lopencv_highgui

#include <iostream>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main (int argc, char *argv[])
{
    float data[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    Mat mat1 (cv::Size (5, 2), CV_32F, data, Mat::AUTO_STEP);
    int row = 0;
    float *p = mat1.ptr<float>(row);

    cout << "Mat" << mat1 <<endl;

    PyObject *pName, *pModule, *pDict, *pFunc, *pArgs;
    npy_intp dims[2] = { 2, 5 };
    PyObject *py_array;

    setenv("PYTHONPATH",".",1);
    Py_Initialize ();
    pName = PyUnicode_FromString ("test-pyCode");
    
    pModule = PyImport_Import(pName);

    pDict = PyModule_GetDict(pModule);
    printf("dupa\n");

    // Required for the C-API : http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
    import_array ();

    py_array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, p);

    pArgs = PyTuple_New (1);
    PyTuple_SetItem (pArgs, 0, py_array);

    pFunc = PyDict_GetItemString (pDict, (char*)"pyArray"); 

    if (PyCallable_Check (pFunc))
    {
        PyObject_CallObject(pFunc, pArgs);
    } else
    {
        cout << "Function is not callable !" << endl;
    }

    Py_DECREF(pName);
    Py_DECREF (py_array);                             
    Py_DECREF (pModule);
    Py_DECREF (pDict);
    Py_DECREF (pFunc);

    Py_Finalize ();                                  

    return 0;
}
