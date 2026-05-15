#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

typedef struct {
    PyObject_HEAD
    Py_ssize_t rows;
    Py_ssize_t cols;
    Py_ssize_t nnz;
    Py_ssize_t *indptr;
    Py_ssize_t *indices;
    double *data;
} CSRMatrixObject;

static void CSRMatrix_dealloc(CSRMatrixObject *self) {
    free(self->indptr);
    free(self->indices);
    free(self->data);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int fill_index_array(PyObject *source, Py_ssize_t expected, Py_ssize_t *target, const char *name) {
    PyObject *seq = PySequence_Fast(source, name);
    if (seq == NULL) {
        return -1;
    }
    if (PySequence_Fast_GET_SIZE(seq) != expected) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "%s must contain %zd entries", name, expected);
        return -1;
    }
    for (Py_ssize_t i = 0; i < expected; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        Py_ssize_t value = PyLong_AsSsize_t(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return -1;
        }
        target[i] = value;
    }
    Py_DECREF(seq);
    return 0;
}

static int fill_double_array(PyObject *source, Py_ssize_t expected, double *target, const char *name) {
    PyObject *seq = PySequence_Fast(source, name);
    if (seq == NULL) {
        return -1;
    }
    if (PySequence_Fast_GET_SIZE(seq) != expected) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "%s must contain %zd entries", name, expected);
        return -1;
    }
    for (Py_ssize_t i = 0; i < expected; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        double value = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return -1;
        }
        target[i] = value;
    }
    Py_DECREF(seq);
    return 0;
}

static double max_abs(double a, double b) {
    return fabs(a) > fabs(b) ? fabs(a) : fabs(b);
}

static void csr_scaled_matvec(CSRMatrixObject *self, const double *x, const double *row_scale, double *out) {
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        double total = 0.0;
        for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
            total += self->data[offset] * x[self->indices[offset]];
        }
        out[row] = total * row_scale[row];
    }
}

static void csr_scaled_transpose_matvec(CSRMatrixObject *self, const double *y, const double *row_scale, double *out) {
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        out[col] = 0.0;
    }
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        double scaled_y = y[row] * row_scale[row];
        for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
            out[self->indices[offset]] += self->data[offset] * scaled_y;
        }
    }
}

static double l2_norm(const double *values, Py_ssize_t count) {
    double total = 0.0;
    for (Py_ssize_t i = 0; i < count; i++) {
        total += values[i] * values[i];
    }
    return sqrt(total);
}

static double estimate_scaled_operator_norm(CSRMatrixObject *self, const double *row_scale) {
    double *x = calloc((size_t)self->cols, sizeof(double));
    double *y = calloc((size_t)self->rows, sizeof(double));
    double *z = calloc((size_t)self->cols, sizeof(double));
    if (x == NULL || y == NULL || z == NULL) {
        free(x);
        free(y);
        free(z);
        return -1.0;
    }
    double initial = self->cols > 0 ? 1.0 / sqrt((double)self->cols) : 0.0;
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        x[col] = initial;
    }
    double norm = 1.0;
    for (int iter = 0; iter < 30; iter++) {
        csr_scaled_matvec(self, x, row_scale, y);
        double ynorm = l2_norm(y, self->rows);
        if (ynorm <= 0.0) {
            break;
        }
        for (Py_ssize_t row = 0; row < self->rows; row++) {
            y[row] /= ynorm;
        }
        csr_scaled_transpose_matvec(self, y, row_scale, z);
        double znorm = l2_norm(z, self->cols);
        if (znorm <= 0.0) {
            break;
        }
        for (Py_ssize_t col = 0; col < self->cols; col++) {
            x[col] = z[col] / znorm;
        }
        norm = ynorm;
    }
    csr_scaled_matvec(self, x, row_scale, y);
    norm = l2_norm(y, self->rows);
    free(x);
    free(y);
    free(z);
    return norm > 0.0 ? norm : 1.0;
}

static double projected_value(double value, double lower, double upper) {
    if (isfinite(lower) && value < lower) {
        return lower;
    }
    if (isfinite(upper) && value > upper) {
        return upper;
    }
    return value;
}

static PyObject *CSRMatrix_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    (void)kwds;
    Py_ssize_t rows;
    Py_ssize_t cols;
    PyObject *indptr_obj;
    PyObject *indices_obj;
    PyObject *data_obj;

    if (!PyArg_ParseTuple(args, "nnOOO", &rows, &cols, &indptr_obj, &indices_obj, &data_obj)) {
        return NULL;
    }
    if (rows < 0 || cols < 0) {
        PyErr_SetString(PyExc_ValueError, "matrix dimensions must be nonnegative");
        return NULL;
    }

    PyObject *data_seq = PySequence_Fast(data_obj, "data must be a sequence");
    if (data_seq == NULL) {
        return NULL;
    }
    Py_ssize_t nnz = PySequence_Fast_GET_SIZE(data_seq);
    Py_DECREF(data_seq);

    CSRMatrixObject *self = (CSRMatrixObject *)type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    self->rows = rows;
    self->cols = cols;
    self->nnz = nnz;
    self->indptr = calloc((size_t)rows + 1, sizeof(Py_ssize_t));
    self->indices = calloc((size_t)nnz, sizeof(Py_ssize_t));
    self->data = calloc((size_t)nnz, sizeof(double));
    if (self->indptr == NULL || self->indices == NULL || self->data == NULL) {
        Py_DECREF(self);
        PyErr_NoMemory();
        return NULL;
    }
    if (fill_index_array(indptr_obj, rows + 1, self->indptr, "indptr") != 0 ||
        fill_index_array(indices_obj, nnz, self->indices, "indices") != 0 ||
        fill_double_array(data_obj, nnz, self->data, "data") != 0) {
        Py_DECREF(self);
        return NULL;
    }
    if (rows > 0 && self->indptr[0] != 0) {
        Py_DECREF(self);
        PyErr_SetString(PyExc_ValueError, "indptr must start with 0");
        return NULL;
    }
    if (self->indptr[rows] != nnz) {
        Py_DECREF(self);
        PyErr_SetString(PyExc_ValueError, "indptr[-1] must equal nnz");
        return NULL;
    }
    for (Py_ssize_t row = 0; row < rows; row++) {
        if (self->indptr[row] > self->indptr[row + 1]) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_ValueError, "indptr must be nondecreasing");
            return NULL;
        }
    }
    for (Py_ssize_t i = 0; i < nnz; i++) {
        if (self->indices[i] < 0 || self->indices[i] >= cols) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_ValueError, "column index out of range");
            return NULL;
        }
    }
    return (PyObject *)self;
}

static PyObject *CSRMatrix_shape(CSRMatrixObject *self, void *closure) {
    (void)closure;
    return Py_BuildValue("(nn)", self->rows, self->cols);
}

static PyObject *CSRMatrix_nnz(CSRMatrixObject *self, void *closure) {
    (void)closure;
    return PyLong_FromSsize_t(self->nnz);
}

static PyObject *CSRMatrix_density(CSRMatrixObject *self, PyObject *Py_UNUSED(ignored)) {
    if (self->rows == 0 || self->cols == 0) {
        return PyFloat_FromDouble(0.0);
    }
    return PyFloat_FromDouble((double)self->nnz / ((double)self->rows * (double)self->cols));
}

static PyObject *CSRMatrix_matvec(CSRMatrixObject *self, PyObject *args) {
    PyObject *vector_obj;
    if (!PyArg_ParseTuple(args, "O", &vector_obj)) {
        return NULL;
    }
    PyObject *vector = PySequence_Fast(vector_obj, "vector must be a sequence");
    if (vector == NULL) {
        return NULL;
    }
    if (PySequence_Fast_GET_SIZE(vector) != self->cols) {
        Py_DECREF(vector);
        PyErr_SetString(PyExc_ValueError, "vector length must match matrix column count");
        return NULL;
    }

    PyObject *result = PyList_New(self->rows);
    if (result == NULL) {
        Py_DECREF(vector);
        return NULL;
    }
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        double total = 0.0;
        for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
            PyObject *item = PySequence_Fast_GET_ITEM(vector, self->indices[offset]);
            double value = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                Py_DECREF(vector);
                Py_DECREF(result);
                return NULL;
            }
            total += self->data[offset] * value;
        }
        PyObject *boxed = PyFloat_FromDouble(total);
        if (boxed == NULL) {
            Py_DECREF(vector);
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, row, boxed);
    }
    Py_DECREF(vector);
    return result;
}

static PyObject *CSRMatrix_transpose_matvec(CSRMatrixObject *self, PyObject *args) {
    PyObject *vector_obj;
    if (!PyArg_ParseTuple(args, "O", &vector_obj)) {
        return NULL;
    }
    PyObject *vector = PySequence_Fast(vector_obj, "vector must be a sequence");
    if (vector == NULL) {
        return NULL;
    }
    if (PySequence_Fast_GET_SIZE(vector) != self->rows) {
        Py_DECREF(vector);
        PyErr_SetString(PyExc_ValueError, "vector length must match matrix row count");
        return NULL;
    }
    double *totals = calloc((size_t)self->cols, sizeof(double));
    if (totals == NULL) {
        Py_DECREF(vector);
        PyErr_NoMemory();
        return NULL;
    }
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        PyObject *item = PySequence_Fast_GET_ITEM(vector, row);
        double value = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            free(totals);
            Py_DECREF(vector);
            return NULL;
        }
        for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
            totals[self->indices[offset]] += self->data[offset] * value;
        }
    }
    PyObject *result = PyList_New(self->cols);
    if (result == NULL) {
        free(totals);
        Py_DECREF(vector);
        return NULL;
    }
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        PyObject *boxed = PyFloat_FromDouble(totals[col]);
        if (boxed == NULL) {
            free(totals);
            Py_DECREF(vector);
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, col, boxed);
    }
    free(totals);
    Py_DECREF(vector);
    return result;
}

static PyObject *CSRMatrix_to_components(CSRMatrixObject *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *indptr = PyList_New(self->rows + 1);
    PyObject *indices = PyList_New(self->nnz);
    PyObject *data = PyList_New(self->nnz);
    if (indptr == NULL || indices == NULL || data == NULL) {
        Py_XDECREF(indptr);
        Py_XDECREF(indices);
        Py_XDECREF(data);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self->rows + 1; i++) {
        PyList_SET_ITEM(indptr, i, PyLong_FromSsize_t(self->indptr[i]));
    }
    for (Py_ssize_t i = 0; i < self->nnz; i++) {
        PyList_SET_ITEM(indices, i, PyLong_FromSsize_t(self->indices[i]));
        PyList_SET_ITEM(data, i, PyFloat_FromDouble(self->data[i]));
    }
    return Py_BuildValue("(NNN)", indptr, indices, data);
}

static PyObject *CSRMatrix_to_dense(CSRMatrixObject *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *rows = PyList_New(self->rows);
    if (rows == NULL) {
        return NULL;
    }
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        PyObject *dense_row = PyList_New(self->cols);
        if (dense_row == NULL) {
            Py_DECREF(rows);
            return NULL;
        }
        for (Py_ssize_t col = 0; col < self->cols; col++) {
            PyList_SET_ITEM(dense_row, col, PyFloat_FromDouble(0.0));
        }
        for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
            PyObject *boxed = PyFloat_FromDouble(self->data[offset]);
            if (boxed == NULL) {
                Py_DECREF(dense_row);
                Py_DECREF(rows);
                return NULL;
            }
            if (PyList_SetItem(dense_row, self->indices[offset], boxed) != 0) {
                Py_DECREF(boxed);
                Py_DECREF(dense_row);
                Py_DECREF(rows);
                return NULL;
            }
        }
        PyList_SET_ITEM(rows, row, dense_row);
    }
    return rows;
}

static PyObject *CSRMatrix_solve_eq_box_pdhg(CSRMatrixObject *self, PyObject *args, PyObject *kwds) {
    PyObject *c_obj;
    PyObject *b_obj;
    PyObject *lo_obj;
    PyObject *hi_obj;
    Py_ssize_t max_iter = 20000;
    Py_ssize_t check_interval = 500;
    double tol = 1e-6;
    double objective_scale = 0.0;
    static char *kwlist[] = {
        "c", "b", "lo", "hi", "max_iter", "tol", "check_interval", "objective_scale", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "OOOO|ndnd",
            kwlist,
            &c_obj,
            &b_obj,
            &lo_obj,
            &hi_obj,
            &max_iter,
            &tol,
            &check_interval,
            &objective_scale)) {
        return NULL;
    }
    if (max_iter < 0 || check_interval <= 0) {
        PyErr_SetString(PyExc_ValueError, "max_iter must be nonnegative and check_interval positive");
        return NULL;
    }

    double *c = calloc((size_t)self->cols, sizeof(double));
    double *b = calloc((size_t)self->rows, sizeof(double));
    double *lo = calloc((size_t)self->cols, sizeof(double));
    double *hi = calloc((size_t)self->cols, sizeof(double));
    double *row_scale = calloc((size_t)self->rows, sizeof(double));
    double *scaled_b = calloc((size_t)self->rows, sizeof(double));
    double *x = calloc((size_t)self->cols, sizeof(double));
    double *x_next = calloc((size_t)self->cols, sizeof(double));
    double *xbar = calloc((size_t)self->cols, sizeof(double));
    double *y = calloc((size_t)self->rows, sizeof(double));
    double *ax = calloc((size_t)self->rows, sizeof(double));
    double *aty = calloc((size_t)self->cols, sizeof(double));
    if (c == NULL || b == NULL || lo == NULL || hi == NULL || row_scale == NULL || scaled_b == NULL ||
        x == NULL || x_next == NULL || xbar == NULL || y == NULL || ax == NULL || aty == NULL) {
        free(c);
        free(b);
        free(lo);
        free(hi);
        free(row_scale);
        free(scaled_b);
        free(x);
        free(x_next);
        free(xbar);
        free(y);
        free(ax);
        free(aty);
        PyErr_NoMemory();
        return NULL;
    }
    if (fill_double_array(c_obj, self->cols, c, "c") != 0 ||
        fill_double_array(b_obj, self->rows, b, "b") != 0 ||
        fill_double_array(lo_obj, self->cols, lo, "lo") != 0 ||
        fill_double_array(hi_obj, self->cols, hi, "hi") != 0) {
        free(c);
        free(b);
        free(lo);
        free(hi);
        free(row_scale);
        free(scaled_b);
        free(x);
        free(x_next);
        free(xbar);
        free(y);
        free(ax);
        free(aty);
        return NULL;
    }

    double c_scale = objective_scale > 0.0 ? objective_scale : 1.0;
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        if (objective_scale <= 0.0) {
            c_scale = max_abs(c_scale, c[col]);
        }
        if (isfinite(lo[col]) && isfinite(hi[col]) && hi[col] < lo[col]) {
            free(c);
            free(b);
            free(lo);
            free(hi);
            free(row_scale);
            free(scaled_b);
            free(x);
            free(x_next);
            free(xbar);
            free(y);
            free(ax);
            free(aty);
            PyErr_SetString(PyExc_ValueError, "upper bound is lower than lower bound");
            return NULL;
        }
    }
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        double row_norm_sq = 0.0;
        for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
            row_norm_sq += self->data[offset] * self->data[offset];
        }
        double row_norm = sqrt(row_norm_sq);
        row_scale[row] = row_norm > 0.0 ? 1.0 / row_norm : 1.0;
        scaled_b[row] = row_scale[row] * b[row];
    }

    double norm = estimate_scaled_operator_norm(self, row_scale);
    if (norm < 0.0) {
        free(c);
        free(b);
        free(lo);
        free(hi);
        free(row_scale);
        free(scaled_b);
        free(x);
        free(x_next);
        free(xbar);
        free(y);
        free(ax);
        free(aty);
        PyErr_NoMemory();
        return NULL;
    }
    double tau = 0.99 / norm;
    double sigma = 0.99 / norm;
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        double start = 0.0;
        if (isfinite(lo[col]) && start < lo[col]) {
            start = lo[col];
        }
        if (isfinite(hi[col]) && start > hi[col]) {
            start = hi[col];
        }
        if (isfinite(lo[col]) && isfinite(hi[col]) && lo[col] <= hi[col]) {
            start = 0.5 * (lo[col] + hi[col]);
        }
        x[col] = start;
        xbar[col] = start;
    }

    double objective = 0.0;
    double max_residual = INFINITY;
    double l2_residual = INFINITY;
    Py_ssize_t iterations = 0;
    const char *status = "iteration_limit";

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t iter = 1; iter <= max_iter; iter++) {
        csr_scaled_matvec(self, xbar, row_scale, ax);
        for (Py_ssize_t row = 0; row < self->rows; row++) {
            y[row] += sigma * (ax[row] - scaled_b[row]);
        }
        csr_scaled_transpose_matvec(self, y, row_scale, aty);
        for (Py_ssize_t col = 0; col < self->cols; col++) {
            double updated = x[col] - tau * (aty[col] + c[col] / c_scale);
            x_next[col] = projected_value(updated, lo[col], hi[col]);
        }
        for (Py_ssize_t col = 0; col < self->cols; col++) {
            xbar[col] = 2.0 * x_next[col] - x[col];
            x[col] = x_next[col];
        }
        iterations = iter;
        if (iter % check_interval == 0 || iter == max_iter) {
            objective = 0.0;
            for (Py_ssize_t col = 0; col < self->cols; col++) {
                objective += c[col] * x[col];
            }
            max_residual = 0.0;
            l2_residual = 0.0;
            for (Py_ssize_t row = 0; row < self->rows; row++) {
                double total = 0.0;
                for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
                    total += self->data[offset] * x[self->indices[offset]];
                }
                double residual = fabs(total - b[row]);
                max_residual = residual > max_residual ? residual : max_residual;
                l2_residual += residual * residual;
            }
            l2_residual = sqrt(l2_residual);
            if (max_residual <= tol) {
                status = "optimal";
                break;
            }
        }
    }
    Py_END_ALLOW_THREADS

    if (max_iter == 0) {
        objective = 0.0;
        for (Py_ssize_t col = 0; col < self->cols; col++) {
            objective += c[col] * x[col];
        }
    }

    PyObject *x_list = PyList_New(self->cols);
    if (x_list == NULL) {
        free(c);
        free(b);
        free(lo);
        free(hi);
        free(row_scale);
        free(scaled_b);
        free(x);
        free(x_next);
        free(xbar);
        free(y);
        free(ax);
        free(aty);
        return NULL;
    }
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        PyObject *boxed = PyFloat_FromDouble(x[col]);
        if (boxed == NULL) {
            Py_DECREF(x_list);
            free(c);
            free(b);
            free(lo);
            free(hi);
            free(row_scale);
            free(scaled_b);
            free(x);
            free(x_next);
            free(xbar);
            free(y);
            free(ax);
            free(aty);
            return NULL;
        }
        PyList_SET_ITEM(x_list, col, boxed);
    }

    PyObject *result = Py_BuildValue(
        "{s:s,s:d,s:d,s:d,s:n,s:d,s:d,s:N}",
        "status",
        status,
        "objective",
        objective,
        "max_primal_residual",
        max_residual,
        "l2_primal_residual",
        l2_residual,
        "iterations",
        iterations,
        "operator_norm",
        norm,
        "step_size",
        tau,
        "x",
        x_list);

    free(c);
    free(b);
    free(lo);
    free(hi);
    free(row_scale);
    free(scaled_b);
    free(x);
    free(x_next);
    free(xbar);
    free(y);
    free(ax);
    free(aty);
    return result;
}

static PyGetSetDef CSRMatrix_getset[] = {
    {"shape", (getter)CSRMatrix_shape, NULL, "matrix shape", NULL},
    {"nnz", (getter)CSRMatrix_nnz, NULL, "number of stored nonzeros", NULL},
    {NULL}
};

static PyMethodDef CSRMatrix_methods[] = {
    {"matvec", (PyCFunction)CSRMatrix_matvec, METH_VARARGS, "Compute A @ x."},
    {"transpose_matvec", (PyCFunction)CSRMatrix_transpose_matvec, METH_VARARGS, "Compute A.T @ x."},
    {"density", (PyCFunction)CSRMatrix_density, METH_NOARGS, "Return nnz / (rows * cols)."},
    {"to_components", (PyCFunction)CSRMatrix_to_components, METH_NOARGS, "Return indptr, indices, data."},
    {"to_dense", (PyCFunction)CSRMatrix_to_dense, METH_NOARGS, "Materialize as nested Python lists."},
    {"solve_eq_box_pdhg", (PyCFunction)CSRMatrix_solve_eq_box_pdhg, METH_VARARGS | METH_KEYWORDS, "Solve min c'x subject to Ax=b and lo<=x<=hi with native PDHG."},
    {NULL}
};

static PyTypeObject CSRMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "linprogx._csparse.CSRMatrix",
    .tp_basicsize = sizeof(CSRMatrixObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)CSRMatrix_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "C-backed compressed sparse row matrix.",
    .tp_methods = CSRMatrix_methods,
    .tp_getset = CSRMatrix_getset,
    .tp_new = CSRMatrix_new,
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_csparse",
    "C sparse matrix types for linprogx.",
    -1,
    NULL,
};

PyMODINIT_FUNC PyInit__csparse(void) {
    if (PyType_Ready(&CSRMatrixType) < 0) {
        return NULL;
    }
    PyObject *module_obj = PyModule_Create(&module);
    if (module_obj == NULL) {
        return NULL;
    }
    Py_INCREF(&CSRMatrixType);
    if (PyModule_AddObject(module_obj, "CSRMatrix", (PyObject *)&CSRMatrixType) < 0) {
        Py_DECREF(&CSRMatrixType);
        Py_DECREF(module_obj);
        return NULL;
    }
    return module_obj;
}
