package es.ulpgc.bigdata.matrixmul;

/**
 * n x n matrix of IEEE-754 binary64 values in row-major order.
 *
 * <p>Preconditions for every accessor: {@code 0 <= i, j < size()}. Instances
 * behave as immutable value objects except that {@link #set} lets benchmark
 * code fill a pre-allocated output buffer.
 */
public final class DenseMatrix {

    private final int size;
    private final double[][] rows;

    private DenseMatrix(int size, double[][] rows) {
        this.size = size;
        this.rows = rows;
    }

    /** Wrap {@code values} without copying; the caller must not mutate it afterwards. */
    public static DenseMatrix wrap(double[][] values) {
        int size = values.length;
        for (double[] row : values) {
            if (row.length != size) {
                throw new IllegalArgumentException("matrix must be square");
            }
        }
        return new DenseMatrix(size, values);
    }

    /** Allocate an size x size matrix filled with zeroes. */
    public static DenseMatrix zeros(int size) {
        return new DenseMatrix(size, new double[size][size]);
    }

    public int size() {
        return size;
    }

    /** Row {@code i} in row-major storage; callers must not mutate the returned array. */
    public double[] row(int i) {
        checkIndex(i);
        return rows[i];
    }

    public double get(int i, int j) {
        checkIndex(i);
        checkIndex(j);
        return rows[i][j];
    }

    public void set(int i, int j, double value) {
        checkIndex(i);
        checkIndex(j);
        rows[i][j] = value;
    }

    /** Sum of all entries in row-major order (deterministic, used as checksum). */
    public double checksum() {
        double total = 0.0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                total += rows[i][j];
            }
        }
        return total;
    }

    private void checkIndex(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("index " + index + " outside [0, " + size + ")");
        }
    }
}
