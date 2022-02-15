#define PY_SSIZE_T_CLEAN
#include <Python.h>
// #include <wiringPi.h>

#include "definitions.h"

///////////////////////////////
// ADC part
static PyObject *adc_open_wrapper(PyObject *self, PyObject *args)
{
    int spifd = terminal_tedium_adc_open();
    return Py_BuildValue("i", spifd);
}


static PyObject *adc_close_wrapper(PyObject *self, PyObject *args)
{
    int spifd = -1;
    if (!PyArg_ParseTuple(args, "i", &spifd))
        return NULL;
    int statusVal = terminal_tedium_adc_close(spifd);
    return Py_BuildValue("i", statusVal);
}

static PyObject *adc_bang_wrapper(PyObject *self, PyObject *args)
{
    int a2d[8];

    int spifd = -1;
    if (!PyArg_ParseTuple(args, "i", &spifd))
        return NULL;

    terminal_tedium_adc_bang(a2d, spifd);

    // Yeah this is ugly but I really don't want to create a custom PyObject
    // Just for 8 int values :)
    // (6 CVs tho, I don't know what the last 2 values are)
    return Py_BuildValue("iiiiii", a2d[0], a2d[1], a2d[2], a2d[3], a2d[4], a2d[5]);
}

///////////////////////////////
// Switches part
// static PyObject *switches_open_wrapper(PyObject *self, PyObject *args)
// {
//     wiringPiSetupGpio()
//     return NULL;
// }
//
// static PyObject *switches_open_wrapper(PyObject *self, PyObject *args)
// {
//     wiringPiSetupGpio()
//     return NULL;
// }

///////////////////////////////
// Wrapper functions

// Function(s) Table
static PyMethodDef tediumControlFunctionsTable[] = {
    {
        "adc_bang",      // name exposed to Python
        adc_bang_wrapper, // C wrapper function
        METH_VARARGS,          // received variable args (but really just 1)
        "" // documentation
    }, {
        "adc_open",      // name exposed to Python
        adc_open_wrapper, // C wrapper function
        METH_VARARGS,          // received variable args (but really just 1)
        "" // documentation
    }, {
        "adc_close",      // name exposed to Python
        adc_close_wrapper, // C wrapper function
        METH_VARARGS,          // received variable args (but really just 1)
        "" // documentation
    }, {
        NULL, NULL, 0, NULL
    }
};

// Module definition
static struct PyModuleDef tediumControl = {
    PyModuleDef_HEAD_INIT,
    "tediumControl",     // name of module exposed to Python
    "", // module documentation
    -1,
    tediumControlFunctionsTable
};

// Initialization Function
PyMODINIT_FUNC PyInit_tediumControl(void) {
    return PyModule_Create(&tediumControl);
}
