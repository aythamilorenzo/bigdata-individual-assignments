package es.ulpgc.bigdata.matrixmul;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Random;

import org.junit.jupiter.api.Test;

/**
 * Correctness of the optimised dense kernels against the validated i-j-k
 * baseline: identity, zero, a known 3x3 answer and random matrices.
 */
class DenseMultiplierEquivalenceTest {

    private static final double TOLERANCE = 1e-12;

    @Test
    void ikjMatchesBaselineOnIdentity() {
        assertMatchesBaseline(new IkjLoopMultiplier(), identity(8));
    }

    @Test
    void blockedMatchesBaselineOnIdentity() {
        assertMatchesBaseline(new BlockedMultiplier(3), identity(8));
    }

    @Test
    void ikjMatchesBaselineOnZeroMatrices() {
        assertMatchesBaseline(new IkjLoopMultiplier(), DenseMatrix.zeros(5));
    }

    @Test
    void blockedMatchesBaselineOnZeroMatrices() {
        assertMatchesBaseline(new BlockedMultiplier(2), DenseMatrix.zeros(5));
    }

    @Test
    void ikjReproducesKnownThreeByThreeAnswer() {
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
        DenseMatrix out = DenseMatrix.zeros(3);
        new IkjLoopMultiplier().multiplyInto(a, b, out);
        double[][] expected = {
                {30, 24, 18},
                {84, 69, 54},
                {138, 114, 90},
        };
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expected[i][j], out.get(i, j), TOLERANCE, "entry " + i + "," + j);
            }
        }
    }

    @Test
    void allOptimisedKernelsMatchBaselineOnRandomMatrices() {
        for (int size : new int[] {1, 2, 7, 16, 33}) {
            DenseMatrix a = random(size, new Random(42L + size));
            DenseMatrix b = random(size, new Random(43L + size));
            MatrixMultiplier[] optimised = {
                    new IkjLoopMultiplier(),
                    new BlockedMultiplier(1),
                    new BlockedMultiplier(4),
                    new BlockedMultiplier(64),
            };
            for (MatrixMultiplier multiplier : optimised) {
                assertMatchesBaseline(multiplier, a, b);
            }
        }
    }

    @Test
    void blockedMultiplierRejectsNonPositiveBlockSize() {
        assertThrows(IllegalArgumentException.class, () -> new BlockedMultiplier(0));
    }

    @Test
    void kernelsRejectMismatchedSizes() {
        assertThrows(IllegalArgumentException.class,
                () -> new IkjLoopMultiplier().multiplyInto(
                        random(3, new Random(1)), random(4, new Random(2)), DenseMatrix.zeros(3)));
        assertThrows(IllegalArgumentException.class,
                () -> new BlockedMultiplier(2).multiplyInto(
                        random(3, new Random(1)), random(3, new Random(2)), DenseMatrix.zeros(4)));
    }

    private static void assertMatchesBaseline(MatrixMultiplier multiplier, DenseMatrix a) {
        assertMatchesBaseline(multiplier, a, a);
    }

    private static void assertMatchesBaseline(MatrixMultiplier multiplier, DenseMatrix a, DenseMatrix b) {
        DenseMatrix expected = DenseMatrix.zeros(a.size());
        new TripleLoopMultiplier().multiplyInto(a, b, expected);
        DenseMatrix actual = DenseMatrix.zeros(a.size());
        multiplier.multiplyInto(a, b, actual);
        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < a.size(); j++) {
                assertEquals(expected.get(i, j), actual.get(i, j), TOLERANCE,
                        "n=" + a.size() + " entry " + i + "," + j + " of "
                                + multiplier.getClass().getSimpleName());
            }
        }
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
