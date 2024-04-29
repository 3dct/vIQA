/*
 * Authors
 * #######
 * Author: Lukas Behammer
 * Research Center Wels
 * University of Applied Sciences Upper Austria, 2023
 * CT Research Group
 *
 * Modifications
 * #############
 * Original code, 2024, Lukas Behammer
 */

#define PY_SSIZE_T_CLEAN
// #include <Python.h>
#include "C:\Users\p42938\.conda\envs\IQA_Library\include\Python.h"
#include <numpy/arrayobject.h>
#include "qmeasurecalcmodule.h"
#include "iAQMeasureCalculation.h"

// Method Table
static PyMethodDef qmeasureMethods[] = {
        {"qmeasure", qmeasure, METH_VARARGS,
                "qmeasure(image: np.ndarray, data_range: int, hist_bins: int, num_peaks: int, analyze_peak: bool) -> float\n\n"
                "Calculate the Q Measure."},
        {NULL, NULL, 0, NULL}
};

// Module Definition
static struct PyModuleDef qmeasureModule = {
        PyModuleDef_HEAD_INIT,
        "qmeasurecalc",
        "Calculate the Q Measure.",
        -1,
        qmeasureMethods
};

// Module Initialization
PyMODINIT_FUNC PyInit_qmeasure(void) {
    _import_array();
    return PyModule_Create(&qmeasureModule);
}

static PyObject
*qmeasure(PyObject *self, PyObject *args)
{
    // Declare variables
    PyArrayObject *matin;
    float *data;
    double qvalue;
    int idatarange, ihistbins, inumpeaks;
    bool banalyzepeak;
    npy_intp dims[3];

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "O!iiib", &PyArray_Type, &matin, &idatarange, &ihistbins, &inumpeaks, &banalyzepeak)) {
        return NULL;
    }
    if (NULL == matin) return NULL;
    // Get dimensions of input array
    dims[0] = matin->dimensions[0];
    dims[1] = matin->dimensions[1];
    dims[2] = matin->dimensions[2];

    double data_range[2];
    data_range[0] = 0;
    data_range[1] = idatarange;

    data = (float *) matin->data;

    // Call the C++ function
    iAQMeasureCalculation iAQMeasureCalculation;
    auto result = iAQMeasureCalculation.computeOrigQ(data, reinterpret_cast<const int *>(dims), data_range, ihistbins, inumpeaks, banalyzepeak);

    qvalue = result["Q (orig, equ 1)"];
    // Return output array
    return PyFloat_FromDouble(qvalue);
}
