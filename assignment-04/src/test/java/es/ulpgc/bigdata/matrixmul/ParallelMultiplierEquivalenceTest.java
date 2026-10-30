package es.ulpgc.bigdata.matrixmul;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Random;

import org.junit.jupiter.api.Test;

/**
 * Correctness of the thread-parallel blocked kernel against its sequential
 * counterpart: identity, zero, a known 3x3 answer, random seeded matrices,
 * and the thread-count invariance of the checksum that evidences safe access
 * to the shared output.
 */
class ParallelMultiplierEquivalenceTest {

    private static final double TOLERANCE = 1e-12;
    private static final int[] THREAD_COUNTS = {1, 2, 3, 4, 8};

    @Test
    void parallelMatchesSequentialOnIdentity() {
        assertMatchesSequential(new int[] {1}, identity(8));
    }

    @Test
    void parallelMatchesSequentialOnZeroMatrices() {
        assertMatchesSequential(new int[] {1, 2, 5}, DenseMatrix.zeros(5));
    }

    @Test
    void parallelReproducesKnownThreeByThreeAnswer() {
        DenseMatrix a = DenseMatrix.wrap(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
        });
        DenseMatrix b = DenseMatrix.wrap(new double[][] {
                {9, 8, 7},
                {6, 5, 4},
                {3, 2, 1},
        });
        double[][] expected = {
                {30, 24, 18},
                {84, 69, 54},
                {138, 114, 90},
        };
        for (int threads : THREAD_COUNTS) {
            DenseMatrix out = DenseMatrix.zeros(3);
            try (ParallelBlockedMultiplier multiplier = new ParallelBlockedMultiplier(64, threads)) {
                multiplier.multiplyInto(a, b, out);
            }
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    assertEquals(expected[i][j], out.get(i, j), TOLERANCE,
                            "threads=" + threads + " entry " + i + "," + j);
                }
            }
        }
    }

    @Test
    void parallelMatchesSequentialOnRandomMatricesAcrossSizesAndThreadCounts() {
        for (int size : new int[] {1, 2, 7, 16, 33, 64, 65, 130}) {
            DenseMatrix a = random(size, new Random(42L + size));
            DenseMatrix b = random(size, new Random(43L + size));
            assertMatchesSequential(new int[] {1, 2, 3, 4, 8}, a, b);
        }
    }

    @Test
    void checksumIsInvariantAcrossThreadCounts() {
        int size = 65;
        DenseMatrix a = random(size, new Random(7L));
        DenseMatrix b = random(size, new Random(8L));
        double sequentialChecksum = multiplySequential(a, b).checksum();
        for (int threads : THREAD_COUNTS) {
            assertEquals(sequentialChecksum, multiplyParallel(a, b, threads).checksum(), 0.0,
                    "checksum changed with threads=" + threads);
        }
    }

    @Test
    void parallelRejectsNonPositiveThreadCounts() {
        assertThrows(IllegalArgumentException.class, () -> new ParallelBlockedMultiplier(64, 0));
        assertThrows(IllegalArgumentException.class, () -> new ParallelBlockedMultiplier(64, -1));
    }

    @Test
    void parallelRejectsNonPositiveBlockSize() {
        assertThrows(IllegalArgumentException.class, () -> new ParallelBlockedMultiplier(0, 2));
    }

    @Test
    void parallelRejectsMismatchedSizes() {
        try (ParallelBlockedMultiplier multiplier = new ParallelBlockedMultiplier(2, 2)) {
            assertThrows(IllegalArgumentException.class,
                    () -> multiplier.multiplyInto(
                            random(3, new Random(1)), random(4, new Random(2)), DenseMatrix.zeros(3)));
        }
    }

    private static void assertMatchesSequential(int[] threadCounts, DenseMatrix a) {
        assertMatchesSequential(threadCounts, a, a);
    }

    private static void assertMatchesSequential(int[] threadCounts, DenseMatrix a, DenseMatrix b) {
        double[][] expected = entryWise(multiplySequential(a, b));
        for (int threads : threadCounts) {
            double[][] actual = entryWise(multiplyParallel(a, b, threads));
            for (int i = 0; i < a.size(); i++) {
                for (int j = 0; j < a.size(); j++) {
                    assertEquals(expected[i][j], actual[i][j], TOLERANCE,
                            "n=" + a.size() + " threads=" + threads + " entry " + i + "," + j);
                }
            }
        }
    }

    private static DenseMatrix multiplySequential(DenseMatrix a, DenseMatrix b) {
        DenseMatrix out = DenseMatrix.zeros(a.size());
        new BlockedMultiplier(64).multiplyInto(a, b, out);
        return out;
    }

    private static DenseMatrix multiplyParallel(DenseMatrix a, DenseMatrix b, int threads) {
        DenseMatrix out = DenseMatrix.zeros(a.size());
        try (ParallelBlockedMultiplier multiplier = new ParallelBlockedMultiplier(64, threads)) {
            multiplier.multiplyInto(a, b, out);
        }
        return out;
    }

    private static double[][] entryWise(DenseMatrix matrix) {
        double[][] values = new double[matrix.size()][matrix.size()];
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix.size(); j++) {
                values[i][j] = matrix.get(i, j);
            }
        }
        return values;
    }

    private static DenseMatrix identity(int size) {
        DenseMatrix matrix = DenseMatrix.zeros(size);
        for (int i = 0; i < size; i++) {
            matrix.set(i, i, 1.0);
        }
        return matrix;
    }

    private static DenseMatrix random(int size, Random seed) {
        double[][] values = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                values[i][j] = seed.nextDouble();
            }
        }
        return DenseMatrix.wrap(values);
    }
}
