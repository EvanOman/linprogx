#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

static double as_double(PyObject *row, Py_ssize_t index) {
    PyObject *item = PySequence_Fast_GET_ITEM(row, index);
    return PyFloat_AsDouble(item);
}

static int set_double(PyObject *row, Py_ssize_t index, double value) {
    PyObject *boxed = PyFloat_FromDouble(value);
    if (boxed == NULL) {
        return -1;
    }
    int result = PyList_SetItem(row, index, boxed);
    if (result != 0) {
        Py_DECREF(boxed);
    }
    return result;
}

static PyObject *pivot(PyObject *self, PyObject *args) {
    PyObject *tableau;
    Py_ssize_t pivot_row;
    Py_ssize_t pivot_col;
    double eps;

    if (!PyArg_ParseTuple(args, "Onnd", &tableau, &pivot_row, &pivot_col, &eps)) {
        return NULL;
    }
    if (!PyList_Check(tableau)) {
        PyErr_SetString(PyExc_TypeError, "tableau must be a list of lists");
        return NULL;
    }

    Py_ssize_t height = PyList_GET_SIZE(tableau);
    if (pivot_row < 0 || pivot_row >= height) {
        PyErr_SetString(PyExc_IndexError, "pivot row out of range");
        return NULL;
    }
    PyObject *prow = PyList_GET_ITEM(tableau, pivot_row);
    if (!PyList_Check(prow)) {
        PyErr_SetString(PyExc_TypeError, "tableau rows must be lists");
        return NULL;
    }
    Py_ssize_t width = PyList_GET_SIZE(prow);
    if (pivot_col < 0 || pivot_col >= width) {
        PyErr_SetString(PyExc_IndexError, "pivot column out of range");
        return NULL;
    }

    double pivot_value = PyFloat_AsDouble(PyList_GET_ITEM(prow, pivot_col));
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (fabs(pivot_value) <= eps) {
        PyErr_SetString(PyExc_ZeroDivisionError, "pivot value is too close to zero");
        return NULL;
    }

    for (Py_ssize_t col = 0; col < width; col++) {
        double value = PyFloat_AsDouble(PyList_GET_ITEM(prow, col));
        if (PyErr_Occurred()) {
            return NULL;
        }
        if (set_double(prow, col, value / pivot_value) != 0) {
            return NULL;
        }
    }

    for (Py_ssize_t row = 0; row < height; row++) {
        if (row == pivot_row) {
            continue;
        }
        PyObject *target = PyList_GET_ITEM(tableau, row);
        if (!PyList_Check(target) || PyList_GET_SIZE(target) != width) {
            PyErr_SetString(PyExc_ValueError, "tableau rows must have the same width");
            return NULL;
        }
        double factor = PyFloat_AsDouble(PyList_GET_ITEM(target, pivot_col));
        if (PyErr_Occurred()) {
            return NULL;
        }
        if (fabs(factor) <= eps) {
            if (set_double(target, pivot_col, 0.0) != 0) {
                return NULL;
            }
            continue;
        }
        for (Py_ssize_t col = 0; col < width; col++) {
            double value = PyFloat_AsDouble(PyList_GET_ITEM(target, col));
            double pvalue = PyFloat_AsDouble(PyList_GET_ITEM(prow, col));
            if (PyErr_Occurred()) {
                return NULL;
            }
            double updated = value - factor * pvalue;
            if (fabs(updated) <= eps) {
                updated = 0.0;
            }
            if (set_double(target, col, updated) != 0) {
                return NULL;
            }
        }
    }

    Py_RETURN_NONE;
}

static PyObject *dot(PyObject *self, PyObject *args) {
    PyObject *left_obj;
    PyObject *right_obj;

    if (!PyArg_ParseTuple(args, "OO", &left_obj, &right_obj)) {
        return NULL;
    }

    PyObject *left = PySequence_Fast(left_obj, "left operand must be a sequence");
    if (left == NULL) {
        return NULL;
    }
    PyObject *right = PySequence_Fast(right_obj, "right operand must be a sequence");
    if (right == NULL) {
        Py_DECREF(left);
        return NULL;
    }

    Py_ssize_t n = PySequence_Fast_GET_SIZE(left);
    if (PySequence_Fast_GET_SIZE(right) != n) {
        Py_DECREF(left);
        Py_DECREF(right);
        PyErr_SetString(PyExc_ValueError, "vectors must have the same length");
        return NULL;
    }

    double total = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        double a = as_double(left, i);
        double b = as_double(right, i);
        if (PyErr_Occurred()) {
            Py_DECREF(left);
            Py_DECREF(right);
            return NULL;
        }
        total += a * b;
    }

    Py_DECREF(left);
    Py_DECREF(right);
    return PyFloat_FromDouble(total);
}

static PyMethodDef methods[] = {
    {"pivot", pivot, METH_VARARGS, "Perform a Gauss-Jordan tableau pivot in place."},
    {"dot", dot, METH_VARARGS, "Compute a dot product for numeric sequences."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_cfast",
    "C helpers for linprogx.",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit__cfast(void) {
    return PyModule_Create(&module);
}
