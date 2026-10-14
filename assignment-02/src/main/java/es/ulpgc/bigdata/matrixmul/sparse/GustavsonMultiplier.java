package es.ulpgc.bigdata.matrixmul.sparse;

import java.util.Arrays;

/**
 * Sparse-sparse multiplication in CSR format (Gustavson's row-wise algorithm).
 *
 * <p>For every row i of a, the stored entries a[i][k] scatter the matching
 * rows of b into a dense accumulator indexed by column; the work is
 * O(sum over stored a[i][k] of nnz(row k of b)), scaling with the nonzero
 * counts instead of n^3. The accumulator and the growable output buffers are
 * instance fields allocated once, so repeated benchmark calls reuse them;
 * only the per-row compaction writes into the output arrays, which are
 * copied to their exact size at the end (this output construction is inside
 * the timed region by contract).
 *
 * <p>Summation order within a row follows the ascending column order of the
 * accumulator sweep, which differs from dense i-j-k; floating-point results
 * may therefore differ from the dense methods within the declared tolerance.
 */
public final class GustavsonMultiplier implements SparseMatrixMultiplier {

    private final double[] accumulator;
    private int[] outputColumns;
    private double[] outputValues;

    /** Multiplier with an accumulator for square matrices of the given side length. */
    public GustavsonMultiplier(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("size must be >= 1, got " + size);
        }
        this.accumulator = new double[size];
        this.outputColumns = new int[0];
        this.outputValues = new double[0];
    }

    @Override
    public CsrMatrix multiply(CsrMatrix a, CsrMatrix b) {
        requireCompatibleSizes(a, b);
        int n = a.size();
        int[] rowPointers = new int[n + 1];
        int nonzeroCount = 0;

        for (int row = 0; row < n; row++) {
            for (int position = a.rowStart(row); position < a.rowEnd(row); position++) {
                scatterRowOfB(b, a.columnAt(position), a.valueAt(position));
            }
            nonzeroCount = compactAccumulatorInto(row, rowPointers, nonzeroCount);
            Arrays.fill(accumulator, 0.0);
        }
        rowPointers[n] = nonzeroCount;
        return CsrMatrix.adopt(n, rowPointers,
                Arrays.copyOf(outputColumns, nonzeroCount),
                Arrays.copyOf(outputValues, nonzeroCount));
    }

    private void scatterRowOfB(CsrMatrix b, int columnOfA, double factor) {
        for (int position = b.rowStart(columnOfA); position < b.rowEnd(columnOfA); position++) {
            accumulator[b.columnAt(position)] += factor * b.valueAt(position);
        }
    }

    /** Copy the nonzero accumulator entries of the current row into the output buffers. */
    private int compactAccumulatorInto(int row, int[] rowPointers, int nonzeroCount) {
        int rowNonZeros = 0;
        for (int column = 0; column < accumulator.length; column++) {
            double value = accumulator[column];
            if (value != 0.0) {
                ensureCapacity(nonzeroCount + rowNonZeros + 1);
                outputColumns[nonzeroCount + rowNonZeros] = column;
                outputValues[nonzeroCount + rowNonZeros] = value;
                rowNonZeros++;
            }
        }
        rowPointers[row + 1] = nonzeroCount + rowNonZeros;
        return nonzeroCount + rowNonZeros;
    }

    private void ensureCapacity(int capacity) {
        if (capacity > outputColumns.length) {
            int grown = Math.max(capacity, Math.max(outputColumns.length * 2, 16));
            outputColumns = Arrays.copyOf(outputColumns, grown);
            outputValues = Arrays.copyOf(outputValues, grown);
        }
    }

    private static void requireCompatibleSizes(CsrMatrix a, CsrMatrix b) {
        if (a.size() != b.size()) {
            throw new IllegalArgumentException("matrices must share the same size");
        }
    }
}
