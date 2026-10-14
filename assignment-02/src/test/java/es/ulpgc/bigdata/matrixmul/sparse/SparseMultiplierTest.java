package es.ulpgc.bigdata.matrixmul.sparse;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.Test;

import es.ulpgc.bigdata.matrixmul.DenseMatrix;
import es.ulpgc.bigdata.matrixmul.TripleLoopMultiplier;

/**
 * Correctness of the CSR pipeline: conversion, canonical form and the
 * Gustavson kernel against the validated dense i-j-k baseline on matched
 * inputs.
 */
class SparseMultiplierTest {

    private static final double TOLERANCE = 1e-12;

    private final DenseToCsrConverter converter = new DenseToCsrConverter();

    @Test
    void conversionDropsZerosAndKeepsRowStructure() {
        DenseMatrix dense = DenseMatrix.wrap(new double[][] {
                {0, 2, 0},
                {0, 0, 0},
                {3, 0, 4},
        });
        CsrMatrix csr = converter.convert(dense);
        assertEquals(3, csr.size());
        assertEquals(3, csr.nonZeros());
        // row 0 stores only the 2 at column 1: positions [0, 1)
        assertEquals(0, csr.rowStart(0));
        assertEquals(1, csr.rowEnd(0));
        assertEquals(1, csr.columnAt(0));
        assertEquals(2.0, csr.valueAt(0), TOLERANCE);
        // row 1 is empty: positions [1, 1)
        assertEquals(1, csr.rowStart(1));
        assertEquals(1, csr.rowEnd(1));
        // row 2 stores 3 at column 0 and 4 at column 2: positions [1, 3)
        assertEquals(1, csr.rowStart(2));
        assertEquals(0, csr.columnAt(1));
        assertEquals(3.0, csr.valueAt(1), TOLERANCE);
        assertEquals(2, csr.columnAt(2));
        assertEquals(4.0, csr.valueAt(2), TOLERANCE);
        assertEquals(3, csr.rowEnd(2));
    }

    @Test
    void conversionRoundTripsThroughToDense() {
        DenseMatrix dense = randomWithZeros(9, new Random(7));
        assertEquals(0.0, diff(converter.convert(dense).toDense(), dense), TOLERANCE);
    }

    @Test
    void gustavsonMatchesDenseBaselineOnRandomSparseMatrices() {
        for (int size : new int[] {1, 2, 5, 17, 40}) {
            for (double density : new double[] {0.01, 0.1, 0.5, 1.0}) {
                DenseMatrix a = randomWithDensity(size, density, new Random(100L + size));
                DenseMatrix b = randomWithDensity(size, density, new Random(200L + size));
                assertMatchesDenseBaseline(a, b);
            }
        }
    }

    @Test
    void productReportsTheDensificationOfTheOutput() {
        // A and B each hold a single nonzero per row; the product stays sparse.
        DenseMatrix a = DenseMatrix.wrap(new double[][] {
                {0, 5},
                {6, 0},
        });
        DenseMatrix b = DenseMatrix.wrap(new double[][] {
                {0, 7},
                {8, 0},
        });
        CsrMatrix product = new GustavsonMultiplier(2).multiply(converter.convert(a), converter.convert(b));
        assertEquals(2, product.nonZeros());
        assertEquals(40.0, product.valueAt(0), TOLERANCE);
        assertEquals(42.0, product.valueAt(1), TOLERANCE);
    }

    @Test
    void csrAdoptionRejectsUnsortedColumns() {
        assertThrows(IllegalArgumentException.class,
                () -> CsrMatrix.adopt(2, new int[] {0, 2, 2}, new int[] {1, 0}, new double[] {1, 2}));
    }

    @Test
    void csrAdoptionRejectsOutOfRangeColumns() {
        assertThrows(IllegalArgumentException.class,
                () -> CsrMatrix.adopt(2, new int[] {0, 1, 1}, new int[] {5}, new double[] {1}));
    }

    @Test
    void gustavsonRejectsMismatchedSizes() {
        CsrMatrix small = converter.convert(DenseMatrix.zeros(2));
        CsrMatrix large = converter.convert(DenseMatrix.zeros(3));
        assertThrows(IllegalArgumentException.class, () -> new GustavsonMultiplier(2).multiply(small, large));
    }

    @Test
    void theoreticalStorageMatchesTheDocumentedFormula() {
        DenseMatrix dense = randomWithDensity(10, 0.5, new Random(3));
        CsrMatrix csr = converter.convert(dense);
        assertEquals(12 * csr.nonZeros() + 4 * (csr.size() + 1), csr.theoreticalStorageBytes());
        assertTrue(csr.nonZeros() > 0);
    }

    private void assertMatchesDenseBaseline(DenseMatrix a, DenseMatrix b) {
        DenseMatrix expected = DenseMatrix.zeros(a.size());
        new TripleLoopMultiplier().multiplyInto(a, b, expected);
        CsrMatrix actual = new GustavsonMultiplier(a.size())
                .multiply(converter.convert(a), converter.convert(b));
        assertEquals(0.0, diff(expected, actual.toDense()), TOLERANCE,
                "n=" + a.size() + " mismatch vs dense baseline");
    }

    private static double diff(DenseMatrix a, DenseMatrix b) {
        double worst = 0.0;
        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < a.size(); j++) {
                worst = Math.max(worst, Math.abs(a.get(i, j) - b.get(i, j)));
            }
        }
        return worst;
    }

    private static DenseMatrix randomWithZeros(int size, Random seed) {
        return randomWithDensity(size, 0.3, seed);
    }

    private static DenseMatrix randomWithDensity(int size, double density, Random seed) {
        double[][] values = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (seed.nextDouble() < density) {
                    values[i][j] = seed.nextDouble();
                }
            }
        }
        return DenseMatrix.wrap(values);
    }
}
