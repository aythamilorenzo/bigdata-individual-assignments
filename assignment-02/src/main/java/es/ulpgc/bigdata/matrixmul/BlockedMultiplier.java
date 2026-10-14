package es.ulpgc.bigdata.matrixmul;

import java.util.Arrays;

/**
 * Dense multiplication with i-k-j loops restricted to blockSized tiles.
 *
 * <p>Cache blocking keeps a {@code blockSize x blockSize} tile of {@code b}
 * (and the matching segments of {@code a} and {@code out}) resident in the
 * cache while it is reused, reducing main-memory traffic for large n.
 * Arithmetic work stays O(n^3) with the same summation order as
 * {@link IkjLoopMultiplier}, so results are element-wise identical to the
 * i-j-k baseline.
 *
 * <p>The block size is a declared experimental factor: it is fixed for the
 * whole campaign and reported with the results.
 */
public final class BlockedMultiplier implements MatrixMultiplier {

    private final int blockSize;

    /** Blocked kernel with the given tile side length; {@code blockSize >= 1}. */
    public BlockedMultiplier(int blockSize) {
        if (blockSize < 1) {
            throw new IllegalArgumentException("blockSize must be >= 1, got " + blockSize);
        }
        this.blockSize = blockSize;
    }

    public int blockSize() {
        return blockSize;
    }

    @Override
    public void multiplyInto(DenseMatrix a, DenseMatrix b, DenseMatrix out) {
        requireCompatibleSizes(a, b, out);
        int n = a.size();
        for (int i = 0; i < n; i++) {
            Arrays.fill(out.row(i), 0.0);
        }
        for (int iBlock = 0; iBlock < n; iBlock += blockSize) {
            int iLimit = Math.min(iBlock + blockSize, n);
            for (int kBlock = 0; kBlock < n; kBlock += blockSize) {
                int kLimit = Math.min(kBlock + blockSize, n);
                for (int jBlock = 0; jBlock < n; jBlock += blockSize) {
                    int jLimit = Math.min(jBlock + blockSize, n);
                    multiplyTile(a, b, out, iBlock, iLimit, kBlock, kLimit, jBlock, jLimit);
                }
            }
        }
    }

    private void multiplyTile(DenseMatrix a, DenseMatrix b, DenseMatrix out,
                              int iBlock, int iLimit, int kBlock, int kLimit,
                              int jBlock, int jLimit) {
        for (int i = iBlock; i < iLimit; i++) {
            double[] rowA = a.row(i);
            double[] rowOut = out.row(i);
            for (int k = kBlock; k < kLimit; k++) {
                double factor = rowA[k];
                double[] rowB = b.row(k);
                for (int j = jBlock; j < jLimit; j++) {
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
