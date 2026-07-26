/* PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.
 *
 * CPython glue for the DS2 CHUZC component.  Testing-only: it exists so the
 * validation harness can drive ds2_chuzc and the two Harris controls over
 * real pivot rows harvested from the shipped solver, and time them with
 * rdtsc inside one process.  The component itself (_ds2_chuzc.c) has no
 * dependency on this file, on CPython, or on _csparse.c.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <string.h>

#include "_ds2_chuzc.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
static inline uint64_t ds2_tsc(void) {
    _mm_lfence();
    uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}
static const int ds2_have_tsc = 1;
#else
#include <time.h>
static inline uint64_t ds2_tsc(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
static const int ds2_have_tsc = 0;
#endif

/* ---- State object ----------------------------------------------------- */

typedef struct {
    PyObject_HEAD DS2ChuzcState *st;
    Py_buffer no_flip_buf;
    int has_no_flip;
} StateObject;

static void State_dealloc(StateObject *self) {
    if (self->has_no_flip) PyBuffer_Release(&self->no_flip_buf);
    ds2_chuzc_state_free(self->st);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *State_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"n_total", NULL};
    int n_total = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &n_total))
        return NULL;
    if (n_total < 1) {
        PyErr_SetString(PyExc_ValueError, "n_total must be positive");
        return NULL;
    }
    StateObject *self = (StateObject *)type->tp_alloc(type, 0);
    if (self == NULL) return NULL;
    self->st = ds2_chuzc_state_new(n_total);
    self->has_no_flip = 0;
    if (self->st == NULL) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    return (PyObject *)self;
}

static PyObject *State_set_no_flip(StateObject *self, PyObject *arg) {
    if (self->has_no_flip) {
        PyBuffer_Release(&self->no_flip_buf);
        self->has_no_flip = 0;
        self->st->no_flip = NULL;
    }
    if (arg == Py_None) Py_RETURN_NONE;
    if (PyObject_GetBuffer(arg, &self->no_flip_buf, PyBUF_C_CONTIGUOUS) != 0)
        return NULL;
    if (self->no_flip_buf.len < (Py_ssize_t)self->st->n_total) {
        PyBuffer_Release(&self->no_flip_buf);
        PyErr_SetString(PyExc_ValueError, "no_flip too short");
        return NULL;
    }
    self->has_no_flip = 1;
    self->st->no_flip = (const uint8_t *)self->no_flip_buf.buf;
    Py_RETURN_NONE;
}

static PyObject *State_stats(StateObject *self, PyObject *Py_UNUSED(a)) {
    DS2ChuzcState *st = self->st;
    return Py_BuildValue(
        "{s:L,s:L,s:L,s:L,s:L,s:L,s:L}", "n_call", (long long)st->n_call,
        "n_admitted", (long long)st->n_admitted, "n_prefilter",
        (long long)st->n_prefilter, "n_sweep_visits",
        (long long)st->n_sweep_visits, "n_group", (long long)st->n_group,
        "n_flip_total", (long long)st->n_flip_total, "n_no_group",
        (long long)st->n_no_group);
}

static PyObject *State_set_census(StateObject *self, PyObject *arg) {
    self->st->census = PyObject_IsTrue(arg) ? 1 : 0;
    Py_RETURN_NONE;
}

static PyObject *State_build_range(StateObject *self, PyObject *args) {
    PyObject *lo_obj, *hi_obj;
    if (!PyArg_ParseTuple(args, "OO", &lo_obj, &hi_obj)) return NULL;
    Py_buffer lo_b, hi_b;
    if (PyObject_GetBuffer(lo_obj, &lo_b, PyBUF_C_CONTIGUOUS) != 0) return NULL;
    if (PyObject_GetBuffer(hi_obj, &hi_b, PyBUF_C_CONTIGUOUS) != 0) {
        PyBuffer_Release(&lo_b);
        return NULL;
    }
    PyObject *ret = NULL;
    if (lo_b.len < (Py_ssize_t)self->st->n_total * 8 ||
        hi_b.len < (Py_ssize_t)self->st->n_total * 8) {
        PyErr_SetString(PyExc_ValueError, "bounds shorter than n_total");
    } else {
        ds2_chuzc_build_range(self->st, (const double *)lo_b.buf,
                              (const double *)hi_b.buf);
        Py_INCREF(Py_None);
        ret = Py_None;
    }
    PyBuffer_Release(&hi_b);
    PyBuffer_Release(&lo_b);
    return ret;
}

static PyObject *State_invalidate_range(StateObject *self,
                                        PyObject *Py_UNUSED(a)) {
    ds2_chuzc_invalidate_range(self->st);
    Py_RETURN_NONE;
}

static PyObject *State_census(StateObject *self, PyObject *Py_UNUSED(a)) {
    DS2ChuzcState *st = self->st;
    return Py_BuildValue(
        "{s:i,s:i,s:d,s:d,s:i,s:d,s:i,s:i,s:i}", "n_cand",
        (int)st->last_n_cand, "n_flippable", (int)st->last_n_flippable,
        "absorb", st->last_absorb, "delta", st->last_delta, "stage1_take",
        (int)st->last_stage1_take, "total_change", st->last_total_change,
        "exhausted", (int)st->last_exhausted, "degenerate",
        (int)st->last_degenerate, "groups", (int)st->n_group_cur - 1);
}

static PyObject *State_reset_stats(StateObject *self, PyObject *Py_UNUSED(a)) {
    ds2_chuzc_state_reset_stats(self->st);
    Py_RETURN_NONE;
}

static PyMethodDef State_methods[] = {
    {"set_no_flip", (PyCFunction)State_set_no_flip, METH_O,
     "Install (or clear, with None) the never-flip column mask."},
    {"stats", (PyCFunction)State_stats, METH_NOARGS, "Accumulated counters."},
    {"reset_stats", (PyCFunction)State_reset_stats, METH_NOARGS,
     "Zero the counters."},
    {"set_census", (PyCFunction)State_set_census, METH_O,
     "Enable/disable the per-call flippability census (untimed runs only)."},
    {"census", (PyCFunction)State_census, METH_NOARGS,
     "Last call's flippability census."},
    {"build_range", (PyCFunction)State_build_range, METH_VARARGS,
     "Cache u_j - l_j from (lo, hi); call after any bound change."},
    {"invalidate_range", (PyCFunction)State_invalidate_range, METH_NOARGS,
     "Drop the cached range table."},
    {NULL, NULL, 0, NULL}};

static PyTypeObject StateType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "linprogx._ds2_chuzc.State",
    .tp_basicsize = sizeof(StateObject),
    .tp_dealloc = (destructor)State_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Component-A ratio-test state.",
    .tp_methods = State_methods,
    .tp_new = State_new,
};

/* ---- chuzc() ---------------------------------------------------------- */

typedef DS2Entering (*ds2_chuzc_fn)(const double *, const int32_t *, int32_t,
                                    const double *, const int8_t *,
                                    const double *, const double *, int, double,
                                    void *);

static PyObject *mod_chuzc(PyObject *Py_UNUSED(self), PyObject *args,
                           PyObject *kwds) {
    static char *kwlist[] = {"kind",    "state",         "alpha_row",
                             "pattern", "r_ext",         "bound_status",
                             "lo",      "hi",            "sigma",
                             "delta",   "update_count",  "dual_tol",
                             "expand_tau", "harris_delta", "repeat",
                             NULL};
    const char *kind;
    StateObject *state;
    PyObject *alpha_obj, *pat_obj, *r_obj, *bs_obj, *lo_obj, *hi_obj;
    int sigma;
    double delta;
    int update_count = 0;
    double dual_tol = 1e-7;
    double expand_tau = 0.0;
    double harris_delta = 1e-7;
    int repeat = 1;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "sO!OOOOOOid|idddi", kwlist, &kind, &StateType, &state,
            &alpha_obj, &pat_obj, &r_obj, &bs_obj, &lo_obj, &hi_obj, &sigma,
            &delta, &update_count, &dual_tol, &expand_tau, &harris_delta,
            &repeat))
        return NULL;

    ds2_chuzc_fn fn;
    if (strcmp(kind, "ds2") == 0) {
        fn = ds2_chuzc;
    } else if (strcmp(kind, "harris_dense") == 0) {
        fn = ds2_chuzc_harris_dense;
    } else if (strcmp(kind, "harris_pattern") == 0) {
        fn = ds2_chuzc_harris_pattern;
    } else {
        PyErr_SetString(PyExc_ValueError,
                        "kind must be ds2 | harris_dense | harris_pattern");
        return NULL;
    }

    Py_buffer alpha_b, pat_b, r_b, bs_b, lo_b, hi_b;
    memset(&alpha_b, 0, sizeof(alpha_b));
    memset(&pat_b, 0, sizeof(pat_b));
    memset(&r_b, 0, sizeof(r_b));
    memset(&bs_b, 0, sizeof(bs_b));
    memset(&lo_b, 0, sizeof(lo_b));
    memset(&hi_b, 0, sizeof(hi_b));
    int got = 0;
    PyObject *result = NULL;

#define GETBUF(obj, buf)                                                     \
    do {                                                                     \
        if (PyObject_GetBuffer((obj), &(buf), PyBUF_C_CONTIGUOUS) != 0)      \
            goto done;                                                       \
        got++;                                                               \
    } while (0)

    GETBUF(alpha_obj, alpha_b);
    GETBUF(pat_obj, pat_b);
    GETBUF(r_obj, r_b);
    GETBUF(bs_obj, bs_b);
    GETBUF(lo_obj, lo_b);
    GETBUF(hi_obj, hi_b);
#undef GETBUF

    const int32_t n_total = state->st->n_total;
    if (alpha_b.len < (Py_ssize_t)n_total * 8 ||
        r_b.len < (Py_ssize_t)n_total * 8 ||
        lo_b.len < (Py_ssize_t)n_total * 8 ||
        hi_b.len < (Py_ssize_t)n_total * 8 ||
        bs_b.len < (Py_ssize_t)n_total) {
        PyErr_SetString(PyExc_ValueError, "array shorter than n_total");
        goto done;
    }
    const int32_t nnz = (int32_t)(pat_b.len / 4);
    if (nnz > n_total) {
        PyErr_SetString(PyExc_ValueError, "pattern longer than n_total");
        goto done;
    }

    state->st->delta = delta;
    state->st->update_count = update_count;
    state->st->expand_tau = expand_tau;
    state->st->harris_delta = harris_delta;

    if (repeat < 1) repeat = 1;
    DS2Entering out;
    uint64_t best = (uint64_t)-1;
    PyObject *flips = NULL;
    Py_BEGIN_ALLOW_THREADS for (int rep = 0; rep < repeat; rep++) {
        const uint64_t t0 = ds2_tsc();
        out = fn((const double *)alpha_b.buf, (const int32_t *)pat_b.buf, nnz,
                 (const double *)r_b.buf, (const int8_t *)bs_b.buf,
                 (const double *)lo_b.buf, (const double *)hi_b.buf, sigma,
                 dual_tol, state->st);
        const uint64_t dt = ds2_tsc() - t0;
        if (dt < best) best = dt;
    }
    Py_END_ALLOW_THREADS

        flips = PyList_New(out.n_flip);
    if (flips == NULL) goto done;
    for (int32_t i = 0; i < out.n_flip; i++) {
        PyObject *v = PyLong_FromLong((long)out.flip_cols[i]);
        if (v == NULL) {
            Py_DECREF(flips);
            flips = NULL;
            goto done;
        }
        PyList_SET_ITEM(flips, i, v);
    }
    result = Py_BuildValue("{s:i,s:d,s:d,s:N,s:K,s:i}", "entering",
                           (int)out.entering, "theta_dual", out.theta_dual,
                           "alpha_pivot", out.alpha_pivot, "flips", flips,
                           "cycles", (unsigned long long)best, "groups",
                           (int)state->st->n_group_cur);
    flips = NULL;

done:
    if (got >= 6) PyBuffer_Release(&hi_b);
    if (got >= 5) PyBuffer_Release(&lo_b);
    if (got >= 4) PyBuffer_Release(&bs_b);
    if (got >= 3) PyBuffer_Release(&r_b);
    if (got >= 2) PyBuffer_Release(&pat_b);
    if (got >= 1) PyBuffer_Release(&alpha_b);
    return result;
}

static PyObject *mod_have_tsc(PyObject *Py_UNUSED(s),
                              PyObject *Py_UNUSED(a)) {
    return PyBool_FromLong(ds2_have_tsc);
}

static PyMethodDef module_methods[] = {
    {"chuzc", (PyCFunction)mod_chuzc, METH_VARARGS | METH_KEYWORDS,
     "Run one CHUZC variant on one pivot row; returns the decision and the "
     "minimum observed cycle count over `repeat` runs."},
    {"have_tsc", (PyCFunction)mod_have_tsc, METH_NOARGS,
     "True when `cycles` really are rdtsc cycles (else nanoseconds)."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef ds2_module = {
    PyModuleDef_HEAD_INIT, "linprogx._ds2_chuzc",
    "DS2 component A (CHUZC) -- test harness bindings.", -1, module_methods,
};

PyMODINIT_FUNC PyInit__ds2_chuzc(void) {
    if (PyType_Ready(&StateType) < 0) return NULL;
    PyObject *m = PyModule_Create(&ds2_module);
    if (m == NULL) return NULL;
    Py_INCREF(&StateType);
    if (PyModule_AddObject(m, "State", (PyObject *)&StateType) < 0) {
        Py_DECREF(&StateType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
