package es.ulpgc.bigdata.matrixmul;

import java.util.Arrays;

/**
 * Dense multiplication with the loop order changed to i-k-j.
 *
 * <p>The k loop moves to the middle position so that both accesses to
 * {@code b} ({@code b[k][j]}) and to {@code out} ({@code out[i][j]}) sweep
 * contiguous row-major memory, removing the strided column access of the
 * i-j-k baseline. O(n^3) arithmetic work is unchanged; only the memory
 * access pattern differs.
 *
 * <p>The output rows are zeroed inside the timed region because the
 * accumulation writes into {@code out} directly.
 */
public final class IkjLoopMultiplier implements MatrixMultiplier {

    @Override
    public void multiplyInto(DenseMatrix a, DenseMatrix b, DenseMatrix out) {
        requireCompatibleSizes(a, b, out);
        int n = a.size();
        for (int i = 0; i < n; i++) {
            double[] rowOut = out.row(i);
            Arrays.fill(rowOut, 0.0);
            double[] rowA = a.row(i);
            for (int k = 0; k < n; k++) {
                double factor = rowA[k];
                double[] rowB = b.row(k);
                for (int j = 0; j < n; j++) {
                    rowOut[j] += factor * rowB[j];
                }
            }
        }
    }

    private static void requireCompatibleSizes(DenseMatrix a, DenseMatrix b, DenseMatrix out) {
        if (a.size() != b.size() || a.size() != out.size()) {
            throw new IllegalArgumentException("matrices must share the same size");
        }
    }
}
