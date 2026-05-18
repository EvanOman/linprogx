use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PySequence, PyTuple};

#[pyclass(module = "linprogx._rsparse", name = "CSRMatrix", unsendable)]
struct CsrMatrix {
    rows: usize,
    cols: usize,
    nnz: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
    csc_indptr: Vec<usize>,
    csc_rows: Vec<usize>,
    csc_data: Vec<f64>,
}

fn extract_usize_seq(obj: &Bound<'_, PyAny>, expected: usize, name: &str) -> PyResult<Vec<usize>> {
    let seq = obj.downcast::<PySequence>().map_err(|_| {
        PyValueError::new_err(format!("{name} must be a sequence"))
    })?;
    let len = seq.len()? as usize;
    if len != expected {
        return Err(PyValueError::new_err(format!(
            "{name} must contain {expected} entries"
        )));
    }
    let mut out = Vec::with_capacity(expected);
    for i in 0..expected {
        let item = seq.get_item(i)?;
        let value: isize = item.extract()?;
        if value < 0 {
            return Err(PyValueError::new_err(format!("{name} entry must be nonnegative")));
        }
        out.push(value as usize);
    }
    Ok(out)
}

fn extract_f64_seq(obj: &Bound<'_, PyAny>, expected: usize, name: &str) -> PyResult<Vec<f64>> {
    let seq = obj.downcast::<PySequence>().map_err(|_| {
        PyValueError::new_err(format!("{name} must be a sequence"))
    })?;
    let len = seq.len()? as usize;
    if len != expected {
        return Err(PyValueError::new_err(format!(
            "{name} must contain {expected} entries"
        )));
    }
    let mut out = Vec::with_capacity(expected);
    for i in 0..expected {
        let item = seq.get_item(i)?;
        out.push(item.extract::<f64>()?);
    }
    Ok(out)
}

fn median_nonzero_abs(values: &[f64]) -> f64 {
    let mut nonzero: Vec<f64> = values
        .iter()
        .copied()
        .map(f64::abs)
        .filter(|v| *v > 0.0 && v.is_finite())
        .collect();
    if nonzero.is_empty() {
        return 1.0;
    }
    nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let kept = nonzero.len();
    let mid = kept / 2;
    let median = if kept % 2 == 0 {
        0.5 * (nonzero[mid - 1] + nonzero[mid])
    } else {
        nonzero[mid]
    };
    if median <= 0.0 {
        return 1.0;
    }
    let magnitude = 10f64.powf(median.log10().floor());
    let rounded = (median / magnitude).round() * magnitude;
    if rounded > 0.0 {
        rounded
    } else {
        median
    }
}

#[inline]
fn csr_scaled_matvec(
    rows: usize,
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    x: &[f64],
    row_scale: &[f64],
    out: &mut [f64],
) {
    debug_assert!(indptr.len() >= rows + 1);
    debug_assert!(out.len() >= rows);
    debug_assert!(row_scale.len() >= rows);
    for row in 0..rows {
        let start = unsafe { *indptr.get_unchecked(row) };
        let end = unsafe { *indptr.get_unchecked(row + 1) };
        let row_data = unsafe { data.get_unchecked(start..end) };
        let row_indices = unsafe { indices.get_unchecked(start..end) };
        let mut total = 0.0_f64;
        for (d, i) in row_data.iter().zip(row_indices.iter()) {
            unsafe {
                total += *d * *x.get_unchecked(*i);
            }
        }
        unsafe {
            *out.get_unchecked_mut(row) = total * *row_scale.get_unchecked(row);
        }
    }
}

#[inline]
fn csr_scaled_transpose_matvec(
    cols: usize,
    csc_indptr: &[usize],
    csc_rows: &[usize],
    csc_data: &[f64],
    y: &[f64],
    row_scale: &[f64],
    out: &mut [f64],
) {
    debug_assert!(csc_indptr.len() >= cols + 1);
    debug_assert!(out.len() >= cols);
    for col in 0..cols {
        let start = unsafe { *csc_indptr.get_unchecked(col) };
        let end = unsafe { *csc_indptr.get_unchecked(col + 1) };
        let col_data = unsafe { csc_data.get_unchecked(start..end) };
        let col_rows = unsafe { csc_rows.get_unchecked(start..end) };
        let mut total = 0.0_f64;
        for (d, i) in col_data.iter().zip(col_rows.iter()) {
            unsafe {
                let r = *i;
                total += *d * *y.get_unchecked(r) * *row_scale.get_unchecked(r);
            }
        }
        unsafe {
            *out.get_unchecked_mut(col) = total;
        }
    }
}

fn l2_norm(values: &[f64]) -> f64 {
    let mut total = 0.0_f64;
    for v in values {
        total += v * v;
    }
    total.sqrt()
}

fn estimate_scaled_operator_norm(matrix: &CsrMatrix, row_scale: &[f64]) -> f64 {
    let cols = matrix.cols;
    let rows = matrix.rows;
    let mut x = vec![0.0_f64; cols];
    let mut y = vec![0.0_f64; rows];
    let mut z = vec![0.0_f64; cols];
    let initial = if cols > 0 { 1.0 / (cols as f64).sqrt() } else { 0.0 };
    for v in x.iter_mut() {
        *v = initial;
    }
    let mut norm = 1.0_f64;
    for _ in 0..30 {
        csr_scaled_matvec(rows, &matrix.indptr, &matrix.indices, &matrix.data, &x, row_scale, &mut y);
        let ynorm = l2_norm(&y);
        if ynorm <= 0.0 {
            break;
        }
        for v in y.iter_mut() {
            *v /= ynorm;
        }
        csr_scaled_transpose_matvec(
            cols,
            &matrix.csc_indptr,
            &matrix.csc_rows,
            &matrix.csc_data,
            &y,
            row_scale,
            &mut z,
        );
        let znorm = l2_norm(&z);
        if znorm <= 0.0 {
            break;
        }
        for col in 0..cols {
            x[col] = z[col] / znorm;
        }
        norm = ynorm;
    }
    csr_scaled_matvec(rows, &matrix.indptr, &matrix.indices, &matrix.data, &x, row_scale, &mut y);
    let n = l2_norm(&y);
    if n > 0.0 {
        n
    } else if norm > 0.0 {
        norm
    } else {
        1.0
    }
}

#[pymethods]
impl CsrMatrix {
    #[new]
    fn new(
        rows: isize,
        cols: isize,
        indptr: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        if rows < 0 || cols < 0 {
            return Err(PyValueError::new_err("matrix dimensions must be nonnegative"));
        }
        let rows = rows as usize;
        let cols = cols as usize;
        let data_seq = data.downcast::<PySequence>().map_err(|_| {
            PyValueError::new_err("data must be a sequence")
        })?;
        let nnz = data_seq.len()? as usize;

        let indptr_vec = extract_usize_seq(indptr, rows + 1, "indptr")?;
        let indices_vec = extract_usize_seq(indices, nnz, "indices")?;
        let data_vec = extract_f64_seq(data, nnz, "data")?;

        if rows > 0 && indptr_vec[0] != 0 {
            return Err(PyValueError::new_err("indptr must start with 0"));
        }
        if indptr_vec[rows] != nnz {
            return Err(PyValueError::new_err("indptr[-1] must equal nnz"));
        }
        for row in 0..rows {
            if indptr_vec[row] > indptr_vec[row + 1] {
                return Err(PyValueError::new_err("indptr must be nondecreasing"));
            }
        }
        for &col in indices_vec.iter() {
            if col >= cols {
                return Err(PyValueError::new_err("column index out of range"));
            }
        }

        // Build CSC representation
        let mut csc_indptr = vec![0_usize; cols + 1];
        for &col in indices_vec.iter() {
            csc_indptr[col + 1] += 1;
        }
        for col in 0..cols {
            csc_indptr[col + 1] += csc_indptr[col];
        }
        let mut csc_rows = vec![0_usize; nnz];
        let mut csc_data = vec![0.0_f64; nnz];
        if cols > 0 {
            let mut next = csc_indptr[..cols].to_vec();
            for row in 0..rows {
                for offset in indptr_vec[row]..indptr_vec[row + 1] {
                    let col = indices_vec[offset];
                    let dest = next[col];
                    next[col] += 1;
                    csc_rows[dest] = row;
                    csc_data[dest] = data_vec[offset];
                }
            }
        }

        Ok(CsrMatrix {
            rows,
            cols,
            nnz,
            indptr: indptr_vec,
            indices: indices_vec,
            data: data_vec,
            csc_indptr,
            csc_rows,
            csc_data,
        })
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[getter]
    fn nnz(&self) -> usize {
        self.nnz
    }

    fn density(&self) -> f64 {
        if self.rows == 0 || self.cols == 0 {
            0.0
        } else {
            self.nnz as f64 / (self.rows as f64 * self.cols as f64)
        }
    }

    fn matvec<'py>(&self, py: Python<'py>, vector: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyList>> {
        let seq = vector.downcast::<PySequence>().map_err(|_| {
            PyValueError::new_err("vector must be a sequence")
        })?;
        if seq.len()? as usize != self.cols {
            return Err(PyValueError::new_err(
                "vector length must match matrix column count",
            ));
        }
        let mut x = Vec::with_capacity(self.cols);
        for i in 0..self.cols {
            x.push(seq.get_item(i)?.extract::<f64>()?);
        }
        let mut out = vec![0.0_f64; self.rows];
        for row in 0..self.rows {
            let mut total = 0.0;
            for offset in self.indptr[row]..self.indptr[row + 1] {
                total += self.data[offset] * x[self.indices[offset]];
            }
            out[row] = total;
        }
        Ok(PyList::new_bound(py, out))
    }

    fn transpose_matvec<'py>(
        &self,
        py: Python<'py>,
        vector: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let seq = vector.downcast::<PySequence>().map_err(|_| {
            PyValueError::new_err("vector must be a sequence")
        })?;
        if seq.len()? as usize != self.rows {
            return Err(PyValueError::new_err(
                "vector length must match matrix row count",
            ));
        }
        let mut y = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            y.push(seq.get_item(i)?.extract::<f64>()?);
        }
        let mut out = vec![0.0_f64; self.cols];
        for row in 0..self.rows {
            let v = y[row];
            for offset in self.indptr[row]..self.indptr[row + 1] {
                out[self.indices[offset]] += self.data[offset] * v;
            }
        }
        Ok(PyList::new_bound(py, out))
    }

    fn to_components<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let indptr = PyList::new_bound(py, self.indptr.iter().map(|v| *v as isize));
        let indices = PyList::new_bound(py, self.indices.iter().map(|v| *v as isize));
        let data = PyList::new_bound(py, self.data.iter().copied());
        Ok(PyTuple::new_bound(py, &[indptr.into_any(), indices.into_any(), data.into_any()]))
    }

    fn to_dense<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let outer = PyList::empty_bound(py);
        for row in 0..self.rows {
            let mut row_vec = vec![0.0_f64; self.cols];
            for offset in self.indptr[row]..self.indptr[row + 1] {
                row_vec[self.indices[offset]] = self.data[offset];
            }
            let row_list = PyList::new_bound(py, row_vec);
            outer.append(row_list)?;
        }
        Ok(outer)
    }

    #[pyo3(signature = (c, b, lo, hi, max_iter=20_000, tol=1e-6, check_interval=500, objective_scale=0.0))]
    fn solve_eq_box_pdhg<'py>(
        &self,
        py: Python<'py>,
        c: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        lo: &Bound<'_, PyAny>,
        hi: &Bound<'_, PyAny>,
        max_iter: isize,
        tol: f64,
        check_interval: isize,
        objective_scale: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        if max_iter < 0 || check_interval <= 0 {
            return Err(PyValueError::new_err(
                "max_iter must be nonnegative and check_interval positive",
            ));
        }

        let c_vec = extract_f64_seq(c, self.cols, "c")?;
        let b_vec = extract_f64_seq(b, self.rows, "b")?;
        let lo_vec = extract_f64_seq(lo, self.cols, "lo")?;
        let hi_vec = extract_f64_seq(hi, self.cols, "hi")?;

        let c_scale = if objective_scale > 0.0 {
            objective_scale
        } else {
            median_nonzero_abs(&c_vec)
        };

        let mut scaled_c = vec![0.0_f64; self.cols];
        let mut bound_kind = vec![0u8; self.cols];
        for col in 0..self.cols {
            scaled_c[col] = c_vec[col] / c_scale;
            let has_lo = lo_vec[col].is_finite();
            let has_hi = hi_vec[col].is_finite();
            bound_kind[col] = (if has_lo { 1 } else { 0 }) | (if has_hi { 2 } else { 0 });
            if has_lo && has_hi && hi_vec[col] < lo_vec[col] {
                return Err(PyValueError::new_err("upper bound is lower than lower bound"));
            }
        }

        let mut row_scale = vec![0.0_f64; self.rows];
        let mut scaled_b = vec![0.0_f64; self.rows];
        for row in 0..self.rows {
            let mut row_norm_sq = 0.0_f64;
            for offset in self.indptr[row]..self.indptr[row + 1] {
                let v = self.data[offset];
                row_norm_sq += v * v;
            }
            let row_norm = row_norm_sq.sqrt();
            row_scale[row] = if row_norm > 0.0 { 1.0 / row_norm } else { 1.0 };
            scaled_b[row] = row_scale[row] * b_vec[row];
        }

        let norm = estimate_scaled_operator_norm(self, &row_scale);
        let tau = 0.99 / norm;
        let sigma = 0.99 / norm;

        let mut x = vec![0.0_f64; self.cols];
        let mut xbar = vec![0.0_f64; self.cols];
        let mut y = vec![0.0_f64; self.rows];
        let mut ax = vec![0.0_f64; self.rows];
        let mut aty = vec![0.0_f64; self.cols];
        for col in 0..self.cols {
            let mut start = 0.0_f64;
            if lo_vec[col].is_finite() && start < lo_vec[col] {
                start = lo_vec[col];
            }
            if hi_vec[col].is_finite() && start > hi_vec[col] {
                start = hi_vec[col];
            }
            if lo_vec[col].is_finite() && hi_vec[col].is_finite() && lo_vec[col] <= hi_vec[col] {
                start = 0.5 * (lo_vec[col] + hi_vec[col]);
            }
            x[col] = start;
            xbar[col] = start;
        }

        // Borrow heap-owned data as slices for the GIL-released inner loop.
        let indptr: &[usize] = &self.indptr;
        let indices: &[usize] = &self.indices;
        let data: &[f64] = &self.data;
        let csc_indptr: &[usize] = &self.csc_indptr;
        let csc_rows: &[usize] = &self.csc_rows;
        let csc_data: &[f64] = &self.csc_data;
        let rows = self.rows;
        let cols = self.cols;
        let max_iter = max_iter as usize;
        let check_interval = check_interval as usize;

        let (iterations, status, max_residual, l2_residual) = py.allow_threads(|| {
            let mut iterations: usize = 0;
            let mut status = "iteration_limit";
            let mut max_residual = f64::INFINITY;
            let mut l2_residual = f64::INFINITY;
            for iter in 1..=max_iter {
                csr_scaled_matvec(rows, &indptr, &indices, &data, &xbar, &row_scale, &mut ax);
                for row in 0..rows {
                    y[row] += sigma * (ax[row] - scaled_b[row]);
                }
                csr_scaled_transpose_matvec(
                    cols,
                    &csc_indptr,
                    &csc_rows,
                    &csc_data,
                    &y,
                    &row_scale,
                    &mut aty,
                );
                for col in 0..cols {
                    let old = x[col];
                    let mut updated = x[col] - tau * (aty[col] + scaled_c[col]);
                    match bound_kind[col] {
                        1 => {
                            if updated < lo_vec[col] {
                                updated = lo_vec[col];
                            }
                        }
                        2 => {
                            if updated > hi_vec[col] {
                                updated = hi_vec[col];
                            }
                        }
                        3 => {
                            if updated < lo_vec[col] {
                                updated = lo_vec[col];
                            } else if updated > hi_vec[col] {
                                updated = hi_vec[col];
                            }
                        }
                        _ => {}
                    }
                    x[col] = updated;
                    xbar[col] = 2.0 * updated - old;
                }
                iterations = iter;
                if iter % check_interval == 0 || iter == max_iter {
                    let mut max_r = 0.0_f64;
                    let mut l2_r = 0.0_f64;
                    for row in 0..rows {
                        let mut total = 0.0_f64;
                        for offset in indptr[row]..indptr[row + 1] {
                            total += data[offset] * x[indices[offset]];
                        }
                        let residual = (total - b_vec[row]).abs();
                        if residual > max_r {
                            max_r = residual;
                        }
                        l2_r += residual * residual;
                    }
                    max_residual = max_r;
                    l2_residual = l2_r.sqrt();
                    if max_residual <= tol {
                        status = "optimal";
                        break;
                    }
                }
            }
            (iterations, status, max_residual, l2_residual)
        });

        let mut objective = 0.0_f64;
        for col in 0..self.cols {
            objective += c_vec[col] * x[col];
        }

        let result = PyDict::new_bound(py);
        result.set_item("status", status)?;
        result.set_item("objective", objective)?;
        result.set_item("max_primal_residual", max_residual)?;
        result.set_item("l2_primal_residual", l2_residual)?;
        result.set_item("iterations", iterations as isize)?;
        result.set_item("operator_norm", norm)?;
        result.set_item("step_size", tau)?;
        result.set_item("objective_scale", c_scale)?;
        let x_list = PyList::new_bound(py, x);
        result.set_item("x", x_list)?;
        Ok(result)
    }
}

#[pymodule]
fn _rsparse(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CsrMatrix>()?;
    Ok(())
}
