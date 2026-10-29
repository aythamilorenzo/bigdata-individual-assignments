package es.ulpgc.bigdata.matrixmul;

import java.util.Arrays;

/**
 * Dense multiplication with i-k-j loops restricted to blockSized tiles.
 *
 * <p>Cache blocking keeps a {@code blockSize x blockSize} tile of {@code b}
 * (and the matching segments of {@code a} and {@code out}) resident in the
 * cache while it is reused, reducing main-memory traffic for large n.
 * Arithmetic work stays O(n^3) and, for every output entry, the sum over k
 * is accumulated in ascending k order, so results are element-wise identical
 * to a plain i-j-k triple loop.
 *
 * <p>This kernel is the sequential baseline of assignment 3 and the exact
 * computation that {@link ParallelBlockedMultiplier} distributes across
 * threads: {@link #multiplyTile} is shared by both so the parallel method
 * cannot drift arithmetically from its sequential counterpart.
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
        MatrixMultiplier.requireCompatibleSizes(a, b, out);
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

    /**
     * Accumulate {@code out[i][j] += a[i][k] * b[k][j]} for one tile, keeping k
     * ascending; shared verbatim by the sequential and the parallel methods.
     */
    static void multiplyTile(DenseMatrix a, DenseMatrix b, DenseMatrix out,
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
}
