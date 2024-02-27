//#include "C:\Users\p42938\.conda\envs\IQA_Library\include\Python.h"
#include "Python.h"
//#include "C:\Users\p42938\.conda\envs\IQA_Library\Lib\site-packages\numpy\core\include\numpy\arrayobject.h"
#include "statisticsmodule.h"

static PyMethodDef statisticsMethods[] = {
        {"minstd", minstd, METH_VARARGS, "Calculate the minimum standard deviation of blocks of a given image."},
        // {"getstatistics", getstatistics, METH_VARARGS, "Calculate the statistics of blocks of a given image."},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef statisticsModule = {
        PyModuleDef_HEAD_INIT,
        "_statistics",
        "Calculate image statistics.",
        -1,
        statisticsMethods
};

PyMODINIT_FUNC PyInit_statistics(void) {
    import_array();
    return PyModule_Create(&statisticsModule);
}

static PyObject *minstd(PyObject *self, PyObject *args) {
    PyArrayObject *matin, *matout, *tmp;
    double **cin, **cout, **tmp_data, mean, stdev, val;
    int iblocksize, istride, i, j, u, v, n, m, dims[2];

    if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &matin, &iblocksize, &istride)) {
        return NULL;
    }
    if (NULL == matin) return NULL;

    n = dims[0] = matin->dimensions[0];
    m = dims[1] = matin->dimensions[1];

    matout=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
    tmp=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);

    tmp_data=pymatrix_to_Carrayptrs(tmp);
    cin=pymatrix_to_Carrayptrs(matin);
    cout=pymatrix_to_Carrayptrs(matout);

    for ( i=0; i<m-(iblocksize-1); i+=istride )
    {
        for ( j=0; j<n-(iblocksize-1); j+=istride )
        {
            mean = 0;
            for ( u=i; u<i+(iblocksize/2); u++ )
            {
                for ( v=j; v<j+(iblocksize/2); v++ )
                {
                    mean += cin[u][v];
                }
            }
            mean /= pow((iblocksize/2), 2);

            stdev = 0;
            for ( u=i; u<i+(iblocksize/2); u++ )
            {
                for ( v=j; v<j+(iblocksize/2); v++ )
                {
                    stdev += pow((cin[u][v]-mean), 2);
                }
            }
            stdev = sqrt(stdev/((iblocksize*iblocksize)-1));

            for ( u=i; u<i+istride; u++ )
            {
                for ( v=j; v<j+istride; v++ )
                {
                    tmp_data[u][v] = stdev;
                    cout[u][v] = stdev;
                }
            }
        }
    }

    for ( i=0; i<m-(iblocksize-1); i+=istride)
    {
        for ( j=0; j<n-(iblocksize-1); j+=istride)
        {
            val = tmp_data[i][j];
            for ( u=i; i<(iblocksize/2); u+=(istride+1))
            {
                for ( v=j; j<(iblocksize/2); v+=(istride+1))
                {
                    if (tmp_data[u][v] < val)
                    {
                        val = tmp_data[u][v];
                    }
                }
            }

            for ( u=i; i<(i+istride); u++)
            {
                for (v = j; j < (j + istride); v++)
                {
                    cout[u][v] = val;
                }
            }
        }
    }

    free_Carrayptrs(cin);
    free_Carrayptrs(cout);

    return PyArray_Return(matout);
}

/* static PyObject *getstatistics(PyObject *self, PyObject *args) {

} */

double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin) {
    double **c, *a;
    int i,n,m;

    n=arrayin->dimensions[0];
    m=arrayin->dimensions[1];
    c=ptrvector(n);
    a=(double *) arrayin->data; /* pointer to arrayin data as double */
    for ( i=0; i<n; i++) {
        c[i]=a+i*m;
    }
    return c;
}

double **ptrvector(long n) {
    double **v;
    v=(double **)malloc((size_t) (n*sizeof(double *)));
    if (!v)   {
        printf("In **ptrvector. Allocation of memory for double array failed.");
        exit(0);
    }
    return v;
}

void free_Carrayptrs(double **v) {
    free((char*) v);
}