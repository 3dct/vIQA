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
 *
 * License
 * #######
 * BSD-3-Clause License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "qmeasurecalcmodule.h"
#include "QMeasureCalculation.h"

// Method Table
static PyMethodDef qmeasureMethods[] = {
        {"qmeasure", qmeasure, METH_VARARGS,
                "qmeasure(image: np.ndarray, min: int, max: int, hist_bins: int, num_peaks: int) -> float\n\n"
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
PyMODINIT_FUNC PyInit_qmeasurecalc(void) {
    _import_array();
    return PyModule_Create(&qmeasureModule);
}

static PyObject
*qmeasure(PyObject *self, PyObject *args)
{
    // Declare variables
    PyArrayObject *matin;
    double qvalue;
    int imin, imax, ihistbins, inumpeaks;
    bool banalyzepeak;
    int dims[3];

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "O!iiii", &PyArray_Type, &matin, &imin, &imax, &ihistbins, &inumpeaks)) {
        return NULL;
    }
    if (NULL == matin) return NULL;

    // Get dimensions of input array and cast to int
    dims[0] = static_cast<int>(matin->dimensions[0]);
    dims[1] = static_cast<int>(matin->dimensions[1]);
    dims[2] = static_cast<int>(matin->dimensions[2]);

    // Get data range
    double data_range[2];
    data_range[0] = imin;
    data_range[1] = imax;

    // Get data from input array
    auto data = (float *) PyArray_DATA(matin);

    // Call the C++ function
    QMeasureCalculation QMeasureCalculation;
    auto result = QMeasureCalculation.computeOrigQ(data, dims, data_range, ihistbins, inumpeaks, false);

    // Get Q value from map
    qvalue = result["Q (orig, equ 1)"];

    // Return output array
    return PyFloat_FromDouble(qvalue);
}
