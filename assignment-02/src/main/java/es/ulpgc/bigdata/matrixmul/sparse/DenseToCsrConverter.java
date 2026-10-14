package es.ulpgc.bigdata.matrixmul.sparse;

import es.ulpgc.bigdata.matrixmul.DenseMatrix;

/**
 * Converts a dense matrix into canonical CSR, dropping the entries that are
 * exactly zero.
 *
 * <p>The conversion is a measured cost of the sparse pipeline, reported
 * separately from the multiplication kernel: it is never hidden inside the
 * kernel timing, because a user holding dense inputs pays it exactly once
 * before multiplying.
 */
public final class DenseToCsrConverter {

    public CsrMatrix convert(DenseMatrix dense) {
        int n = dense.size();
        int nonzeroCount = countNonZeros(dense, n);
        int[] rowPointers = new int[n + 1];
        int[] columnIndices = new int[nonzeroCount];
        double[] values = new double[nonzeroCount];
        fill(dense, n, rowPointers, columnIndices, values);
        return CsrMatrix.adopt(n, rowPointers, columnIndices, values);
    }

    private static int countNonZeros(DenseMatrix dense, int n) {
        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dense.get(i, j) != 0.0) {
                    count++;
                }
            }
        }
        return count;
    }

    private static void fill(DenseMatrix dense, int n, int[] rowPointers,
                             int[] columnIndices, double[] values) {
        int position = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double value = dense.get(i, j);
                if (value != 0.0) {
                    columnIndices[position] = j;
                    values[position] = value;
                    position++;
                }
            }
            rowPointers[i + 1] = position;
        }
    }
}
