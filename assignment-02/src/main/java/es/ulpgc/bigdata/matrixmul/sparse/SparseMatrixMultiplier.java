package es.ulpgc.bigdata.matrixmul.sparse;

/**
 * Strategy for computing {@code c = a * b} on square CSR matrices.
 *
 * <p>Contract: implementations never mutate {@code a} or {@code b}, perform
 * no I/O, and return the product in canonical CSR form. The timed region of
 * the sparse benchmark includes construction of the output structure (its
 * nonzero count is data-dependent and unknown before the computation); the
 * reusable dense accumulator lives outside the timed region.
 */
public interface SparseMatrixMultiplier {

    CsrMatrix multiply(CsrMatrix a, CsrMatrix b);
}
