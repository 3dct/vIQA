/* .... Python callable C functions .... */
static PyObject *minstd(PyObject *self, PyObject *args);
static PyObject *getstatistics(PyObject *self, PyObject *args);

/* .... C matrix utility functions ....*/
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
double **ptrvector(long n);
void free_Carrayptrs(double **v);
