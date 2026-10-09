package es.ulpgc.bigdata.matrixmul;

/**
 * Strategy for computing {@code out = a * b} on dense square matrices.
 *
 * <p>Contract: implementations overwrite every entry of {@code out}, never
 * mutate {@code a} or {@code b}, and perform no I/O.
 */
public interface MatrixMultiplier {

    void multiplyInto(DenseMatrix a, DenseMatrix b, DenseMatrix out);
}
