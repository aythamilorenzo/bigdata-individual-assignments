package es.ulpgc.bigdata.matrixmul;

/**
 * Straightforward dense multiplication with the i-j-k triple loop.
 *
 * <p>O(n^3) arithmetic work on row-major storage; column access to {@code b}
 * follows the algorithm's natural order and is part of the measured baseline,
 * not an optimisation target of this assignment.
 */
public final class TripleLoopMultiplier implements MatrixMultiplier {

    @Override
    public void multiplyInto(DenseMatrix a, DenseMatrix b, DenseMatrix out) {
        requireCompatibleSizes(a, b, out);
        int n = a.size();
        for (int i = 0; i < n; i++) {
            double[] rowA = a.row(i);
            double[] rowOut = out.row(i);
            for (int j = 0; j < n; j++) {
                double total = 0.0;
                for (int k = 0; k < n; k++) {
                    total += rowA[k] * b.row(k)[j];
                }
                rowOut[j] = total;
            }
        }
    }

    private static void requireCompatibleSizes(DenseMatrix a, DenseMatrix b, DenseMatrix out) {
        if (a.size() != b.size() || a.size() != out.size()) {
            throw new IllegalArgumentException("matrices must share the same size");
        }
    }
}
