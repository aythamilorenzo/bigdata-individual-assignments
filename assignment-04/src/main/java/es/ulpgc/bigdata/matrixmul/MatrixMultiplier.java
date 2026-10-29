package es.ulpgc.bigdata.matrixmul;

/**
 * Strategy for computing {@code out = a * b} on dense square matrices.
 *
 * <p>Contract: implementations overwrite every entry of {@code out}, never
 * mutate {@code a} or {@code b}, and perform no I/O.
 */
public interface MatrixMultiplier {

    void multiplyInto(DenseMatrix a, DenseMatrix b, DenseMatrix out);

    /**
     * Convenience method: allocates output matrix and calls multiplyInto.
     */
    default DenseMatrix multiply(DenseMatrix a, DenseMatrix b) {
        DenseMatrix out = DenseMatrix.zeros(a.size());
        multiplyInto(a, b, out);
        return out;
    }

    /** Shared precondition of every dense kernel: all three matrices square of one size. */
    static void requireCompatibleSizes(DenseMatrix a, DenseMatrix b, DenseMatrix out) {
        if (a.size() != b.size() || a.size() != out.size()) {
            throw new IllegalArgumentException("matrices must share the same size");
        }
    }
}
