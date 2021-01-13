# Low-level framework design

# Matrix storage & view

Matrices are quite large and copying submatrices and vectors should be avoided if possible.
Therefore we opt to fundamentaly split the matrix API in a storage class `matrix_storage` and a view class `matrix_view`.
Essentially `matrix_storage` actually allocates and manages memory. 
A `matrix_view` only points to a `matrix_storage` instance and parameters that define which submatrix to use.
All matrix operations should operate on `matrix_view`, where creating new `matrix_storage` as output should be avoided.
