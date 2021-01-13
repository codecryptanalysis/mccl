# Low-level framework design

# Matrix storage & view

Matrices are quite large and copying submatrices and vectors should be avoided if possible.
Therefore we opt to fundamentaly split the matrix API in two:

- in a storage class `matrix_storage` which actually allocates and manages memory. 
-  and a view class `matrix_view` that points to a `matrix_storage` instance and parameters that define which submatrix to use.

All matrix operations should operate on `matrix_view`, where creating new `matrix_storage` as output should be avoided.
Instead destination matrices should also be passed as input.

# Algorithm optimization ideas

The `matrix_view` is in some sense restricting for performance:

- byte alignment is needed for good performance
- there is no overflow space for improvemnts, e.g., to use 256-bit simd over 248 columns

An invasive design principle for matrix algorithms that may help here is as follows:

- add scratch space columns in matrix_storage
- minimal effort reorder columns such that submatrix columns will be followed by scratch space columns (potentially breaks other matrix_view)
- make matrix_view aware of actual columns and available scratch space to the right
