package es.ulpgc.bigdata.matrixmul;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Correctness suite for the Java kernel.
 *
 * <p>Identity, zero and known-answer cases are exact in binary64; the random
 * n = 5 case compares against the shared NumPy reference files with the
 * assignment tolerances rtol = 1e-9 and atol = 1e-12.
 */
class TripleLoopMultiplierTest {

    private static final double RTOL = 1e-9;
    private static final double ATOL = 1e-12;
    private static final int REFERENCE_SIZE = 5;
    private static final Path REFERENCE_DIR = Path.of("data", "reference");

    private final TripleLoopMultiplier multiplier = new TripleLoopMultiplier();

    @Test
    @DisplayName("identity matrix leaves the other operand unchanged")
    void identityMatrixProductPreservesOperand() {
        DenseMatrix matrixA = randomMatrix(4, 7L);
        DenseMatrix identity = DenseMatrix.zeros(4);
        for (int i = 0; i < 4; i++) {
            identity.set(i, i, 1.0);
        }

        DenseMatrix out = DenseMatrix.zeros(4);
        multiplier.multiplyInto(matrixA, identity, out);
        assertMatricesEqualExactly(matrixA, out);

        multiplier.multiplyInto(identity, matrixA, out);
        assertMatricesEqualExactly(matrixA, out);
    }

    @Test
    @DisplayName("any matrix times the zero matrix is zero")
    void zeroMatrixProductIsZero() {
        DenseMatrix matrixA = randomMatrix(4, 11L);
        DenseMatrix zero = DenseMatrix.zeros(4);
        DenseMatrix out = DenseMatrix.zeros(4);

        multiplier.multiplyInto(matrixA, zero, out);
        multiplier.multiplyInto(zero, matrixA, out);

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                assertEquals(0.0, out.get(i, j), "entry (" + i + "," + j + ")");
            }
        }
    }

    @Test
    @DisplayName("3x3 known-answer case")
    void knownAnswerThreeByThree() {
        DenseMatrix matrixA = DenseMatrix.wrap(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });
        DenseMatrix matrixB = DenseMatrix.wrap(new double[][] {
                {9, 8, 7},
                {6, 5, 4},
                {3, 2, 1}
        });
        double[][] expected = {
                {30, 24, 18},
                {84, 69, 54},
                {138, 114, 90}
        };

        DenseMatrix out = DenseMatrix.zeros(3);
        multiplier.multiplyInto(matrixA, matrixB, out);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expected[i][j], out.get(i, j), 0.0);
            }
        }
    }

    @Test
    @DisplayName("random n=5 matches the shared NumPy reference within tolerance")
    void matchesSharedNumPyReference() throws IOException {
        DenseMatrix matrixA = loadReference("matrixA_5.bin");
        DenseMatrix matrixB = loadReference("matrixB_5.bin");
        DenseMatrix expected = loadReference("expectedC_5.bin");

        DenseMatrix out = DenseMatrix.zeros(REFERENCE_SIZE);
        multiplier.multiplyInto(matrixA, matrixB, out);

        for (int i = 0; i < REFERENCE_SIZE; i++) {
            for (int j = 0; j < REFERENCE_SIZE; j++) {
                int row = i;
                int column = j;
                double reference = expected.get(row, column);
                double tolerance = ATOL + RTOL * Math.abs(reference);
                assertTrue(Math.abs(out.get(row, column) - reference) <= tolerance,
                        () -> "entry (" + row + "," + column + "): got " + out.get(row, column)
                                + ", expected " + reference);
            }
        }
    }

    private static DenseMatrix randomMatrix(int size, long seed) {
        Random random = new Random(seed);
        double[][] rows = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                rows[i][j] = random.nextDouble();
            }
        }
        return DenseMatrix.wrap(rows);
    }

    private static DenseMatrix loadReference(String fileName) throws IOException {
        byte[] bytes = Files.readAllBytes(REFERENCE_DIR.resolve(fileName));
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        double[][] rows = new double[REFERENCE_SIZE][REFERENCE_SIZE];
        for (int i = 0; i < REFERENCE_SIZE; i++) {
            for (int j = 0; j < REFERENCE_SIZE; j++) {
                rows[i][j] = buffer.getDouble();
            }
        }
        return DenseMatrix.wrap(rows);
    }

    private static void assertMatricesEqualExactly(DenseMatrix expected, DenseMatrix actual) {
        assertEquals(expected.size(), actual.size());
        for (int i = 0; i < expected.size(); i++) {
            for (int j = 0; j < expected.size(); j++) {
                assertEquals(expected.get(i, j), actual.get(i, j),
                        "entry (" + i + "," + j + ")");
            }
        }
    }
}
