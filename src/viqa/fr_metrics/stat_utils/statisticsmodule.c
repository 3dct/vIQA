/*
 * Authors
 * #######
 * Author: Eric Larson
 * Department of Electrical and Computer Engineering
 * Oklahoma State University, 2008
 * University Of Washington Seattle, 2009
 * Image Coding and Analysis Lab
 *
 * Adaption: Lukas Behammer
 * Research Center Wels
 * University of Applied Sciences Upper Austria, 2023
 * CT Research Group
 *
 * Modifications
 * #############
 * Original code, 2008, Eric Larson
 * Adaption for Python, 2024, Lukas Behammer
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "statisticsmodule.h"

// Method Table
static PyMethodDef statisticsMethods[] = {
    {"minstd", minstd, METH_VARARGS,
         "minstd(image: np.ndarray, blocksize: int, stride: int) -> np.ndarray\n\n"
         "Calculate the minimum standard deviation of blocks of a given image."},
    {"getstatistics", getstatistics, METH_VARARGS,
         "getstatistics(image: np.ndarray, blocksize: int, stride: int) -> (np.ndarray, np.ndarray, np.ndarray)\n\n"
         "Calculate the statistics of blocks of a given image."},
    {NULL, NULL, 0, NULL}
};

// Module Definition
static struct PyModuleDef statisticsModule = {
    PyModuleDef_HEAD_INIT,
    "statisticscalc",
    "Calculate image statistics.",
    -1,
    statisticsMethods
};

// Module Initialization
PyMODINIT_FUNC PyInit_statisticscalc(void) {
    _import_array();
    return PyModule_Create(&statisticsModule);
}

static PyObject
*minstd(PyObject *self, PyObject *args)
{
    // Declare variables
    PyArrayObject *matin, *matout, *tmp;
    double **cin, **cout, **tmp_data, mean, stdev, val;
    int iblocksize, istride, i, j, u, v, n, m;
    npy_intp dims[2];

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &matin, &iblocksize, &istride)) {
        return NULL;
    }
    if (NULL == matin) return NULL;
    // Get dimensions of input array
    n = dims[0] = matin->dimensions[0];
    m = dims[1] = matin->dimensions[1];

    // Create output arrays
    matout=(PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    tmp=(PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    // Convert input and output arrays to C pointers
    tmp_data=pymatrix_to_Carrayptrs(tmp);
    cin=pymatrix_to_Carrayptrs(matin);
    cout=pymatrix_to_Carrayptrs(matout);

    // For each area of size istride x istride, calculate the standard deviation
    for (i = 0; i < m-(iblocksize-1); i += istride) {
        for (j = 0; j < n-(iblocksize-1); j += istride) {
            // Calculate the mean for each block
            mean = 0;
            for (u = i; u < i+iblocksize; u++) {
                for (v = j; v < j+iblocksize; v++) {
                    mean += cin[u][v];
                }
            }
            mean /= pow(iblocksize, 2);

            // Calculate the standard deviation for each block
            stdev = 0;
            for (u = i; u < i+iblocksize; u++) {
                for (v = j; v < j+iblocksize; v++) {
                    stdev += pow((cin[u][v]-mean), 2);
                }
            }
            stdev = sqrt(stdev/((iblocksize*iblocksize)-1));

            // Assign calculated values to temp and output arrays
            for (u = i; u < i+istride; u++) {
                for (v = j; v < j+istride; v++) {
                    tmp_data[u][v] = stdev;
                    cout[u][v] = stdev;
                }
            }
        }
    }

    // Calculate minimum standard deviation for each area
    for (i = 0; i < m-(iblocksize-1); i += istride) {
        for (j = 0; j < n-(iblocksize-1); j += istride) {
            // Look for minimum standard deviation in blocks of size istride x istride
            val = tmp_data[i][j];
            for (u = i; u < (iblocksize/2); u += (istride+1)) {
                for (v = j; v < (iblocksize/2); v += (istride+1)) {
                    if (tmp_data[u][v] < val)
                    {
                        val = tmp_data[u][v];
                    }
                }
            }

            // Assign minimum standard deviation to output array
            for (u = i; u < (i+istride); u++) {
                for (v = j; v < (j + istride); v++) {
                    cout[u][v] = val;
                }
            }
        }
    }

    // Free memory
    free_Carrayptrs(cin);
    free_Carrayptrs(cout);
    free_Carrayptrs(tmp_data);

    // Return output array
    return PyArray_Return(matout);
}

static PyObject
*getstatistics(PyObject *self, PyObject *args)
{
    // Declare variables
    PyArrayObject *matin, *stdevout, *skwout, *krtout;
    double **cin, **stdev_data, **skw_data, **krt_data, tmp, stmp, mean, stdev, krt, skw;
    int iblocksize, istride, i, j, u, v, n, m;
    npy_intp dims[2];

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &matin, &iblocksize, &istride)) {
        return NULL;
    }
    if (NULL == matin) return NULL;

    // Get dimensions of input array
    n = dims[0] = matin->dimensions[0];
    m = dims[1] = matin->dimensions[1];

    // Create output arrays
    stdevout=(PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    skwout=(PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    krtout=(PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    // Convert input and output arrays to C pointers
    cin=pymatrix_to_Carrayptrs(matin);
    stdev_data=pymatrix_to_Carrayptrs(stdevout);
    skw_data=pymatrix_to_Carrayptrs(skwout);
    krt_data=pymatrix_to_Carrayptrs(krtout);

    // For each area of size istride x istride, calculate the standard deviation, skewness and kurtosis
    for (i = 0; i < m-(iblocksize-1); i += istride) {
        for (j = 0; j < n-(iblocksize-1); j += istride) {
            // Calculate the mean for each block
            mean = 0;
            for (u=i; u<i+iblocksize; u++) {
                for (v=j; v<j+iblocksize; v++) {
                    mean += cin[u][v];
                }
            }
            mean /= pow((iblocksize), 2);

            // Calculate the standard deviation, skewness and kurtosis for each block
            stdev = 0;
            skw = 0;
            krt = 0;
            for (u = i; u < i+iblocksize; u++) {
                for (v = j; v < j+iblocksize; v++) {
                    // Calculate numerators
                    tmp = cin[u][v]-mean;
                    stdev += pow(tmp, 2);
                    skw += pow(tmp, 3);
                    krt += pow(tmp, 4);
                }
            }
            stmp = sqrt(stdev/((iblocksize*iblocksize))); // Temporary variable for denominator calculation
            stdev = sqrt(stdev/((iblocksize*iblocksize)-1)); // No denominator needed for standard deviation

            // Avoid division by zero
            if (stmp != 0) { // If denominator is not zero{
                skw = skw/((iblocksize*iblocksize)*pow(stmp, 3));
                krt = krt/((iblocksize*iblocksize)*pow(stmp, 4));
                // krt -= 3 // krt is defined differently than original code
            }
            else {
                skw = 0;
                krt = 0;
            }

            // Assign calculated values to output arrays
            for (u = i; u < i+istride; u++) {
                for (v = j; v < j+istride; v++) {
                    stdev_data[u][v] = stdev;
                    skw_data[u][v] = skw;
                    krt_data[u][v] = krt;
                }
            }
        }
    }

    // Free memory
    free_Carrayptrs(cin);
    free_Carrayptrs(stdev_data);
    free_Carrayptrs(skw_data);
    free_Carrayptrs(krt_data);

    // Return output arrays as Python tuple
    return PyTuple_Pack(3, stdevout, skwout, krtout);
}

/*
 * C matrix utility functions
 *
 * Copyright (c) 2006 Lou Pecora
 * Available under https://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html
 */
double
**pymatrix_to_Carrayptrs(PyArrayObject *arrayin)
{
    double **c, *a;
    int i,n,m;

    n = arrayin->dimensions[0];
    m = arrayin->dimensions[1];
    c = ptrvector(n);
    a = (double *) arrayin->data; /* pointer to arrayin data as double */
    for (i = 0; i < n; i++) {
        c[i] = a + i*m;
    }
    return c;
}

double
**ptrvector(long n)
{
    double **v;
    v = (double **)malloc((size_t) (n*sizeof(double *)));
    if (!v) {
        printf("In **ptrvector. Allocation of memory for double array failed.");
        exit(0);
    }
    return v;
}

void
free_Carrayptrs(double **v)
{
    free((char*) v);
}
