package es.ulpgc.bigdata.matrixmul.sparse;

import es.ulpgc.bigdata.matrixmul.DenseMatrix;

/**
 * n x n matrix in compressed sparse row (CSR) format.
 *
 * <p>Storage: {@code values[k]} holds the value of the nonzero at column
 * {@code columnIndices[k]} of the row delimited by
 * {@code rowPointers[r]..rowPointers[r+1]}. Within every row the column
 * indices are strictly increasing (canonical CSR), so instances are
 * immutable value objects. A row with no stored entries contributes nothing
 * to a product, which is exactly the sparsity advantage under study.
 *
 * <p>Byte footprint of one instance: {@code 12 * nonZeros + 4 * (size + 1)}
 * (8-byte float64 values + 4-byte int32 columns + 4-byte int32 row pointers).
 */
public final class CsrMatrix {

    private final int size;
    private final int[] rowPointers;
    private final int[] columnIndices;
    private final double[] values;

    private CsrMatrix(int size, int[] rowPointers, int[] columnIndices, double[] values) {
        this.size = size;
        this.rowPointers = rowPointers;
        this.columnIndices = columnIndices;
        this.values = values;
    }

    /**
     * Adopt the three CSR arrays without copying; the caller must not mutate
     * them afterwards.
     *
     * @throws IllegalArgumentException if the canonical-CSR invariants fail
     */
    public static CsrMatrix adopt(int size, int[] rowPointers, int[] columnIndices, double[] values) {
        CsrMatrix matrix = new CsrMatrix(size, rowPointers, columnIndices, values);
        matrix.validate();
        return matrix;
    }

    public int size() {
        return size;
    }

    public long nonZeros() {
        return values.length;
    }

    public int rowStart(int row) {
        return rowPointers[row];
    }

    public int rowEnd(int row) {
        return rowPointers[row + 1];
    }

    public int columnAt(int position) {
        return columnIndices[position];
    }

    public double valueAt(int position) {
        return values[position];
    }

    /** Row-major sum over the stored entries; equals the dense checksum (implicit zeros add nothing). */
    public double checksum() {
        double total = 0.0;
        for (double value : values) {
            total += value;
        }
        return total;
    }

    /** Materialise the dense equivalent; used only for matched dense comparisons, never inside a timed region. */
    public DenseMatrix toDense() {
        double[][] rows = new double[size][size];
        for (int row = 0; row < size; row++) {
            for (int position = rowPointers[row]; position < rowPointers[row + 1]; position++) {
                rows[row][columnIndices[position]] = values[position];
            }
        }
        return DenseMatrix.wrap(rows);
    }

    /** Theoretical CSR storage in bytes: 12 bytes per nonzero plus 4 bytes per row pointer. */
    public long theoreticalStorageBytes() {
        return 12L * values.length + 4L * (size + 1);
    }

    private void validate() {
        if (rowPointers.length != size + 1) {
            throw new IllegalArgumentException("rowPointers must have size + 1 entries");
        }
        if (columnIndices.length != values.length) {
            throw new IllegalArgumentException("columnIndices and values must have equal length");
        }
        if (rowPointers[0] != 0 || rowPointers[size] != values.length) {
            throw new IllegalArgumentException("rowPointers must start at 0 and end at the nonzero count");
        }
        for (int row = 0; row < size; row++) {
            if (rowPointers[row] > rowPointers[row + 1]) {
                throw new IllegalArgumentException("rowPointers must be non-decreasing");
            }
            int previousColumn = -1;
            for (int position = rowPointers[row]; position < rowPointers[row + 1]; position++) {
                int column = columnIndices[position];
                if (column <= previousColumn || column >= size) {
                    throw new IllegalArgumentException(
                            "columns within row " + row + " must be strictly increasing in [0, size)");
                }
                previousColumn = column;
            }
        }
    }
}
