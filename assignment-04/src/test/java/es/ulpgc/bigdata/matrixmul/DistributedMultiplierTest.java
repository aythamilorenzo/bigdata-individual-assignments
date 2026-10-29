package es.ulpgc.bigdata.matrixmul;

import es.ulpgc.bigdata.matrixmul.distributed.DistributedMultiplier;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for distributed matrix multiplication.
 * Validates correctness against sequential BlockedMultiplier.
 */
public class DistributedMultiplierTest {

    private static JavaSparkContext sc;

    @BeforeAll
    static void setupSpark() {
        SparkConf conf = new SparkConf()
            .setAppName("DistributedMultiplierTest")
            .setMaster("local[2]")
            .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        sc = new JavaSparkContext(conf);
    }

    @AfterAll
    static void teardownSpark() {
        if (sc != null) {
            sc.close();
        }
    }

    @Test
    public void testIdentityMatrix() {
        int n = 10;
        DenseMatrix identity = createIdentityMatrix(n);
        DenseMatrix a = createRandomMatrix(n, 42);
        
        DistributedMultiplier distributed = new DistributedMultiplier(sc, 2);
        DenseMatrix result = distributed.multiply(a, identity);
        
        assertMatricesEqual(a, result, 1e-10);
    }

    @Test
    public void testZeroMatrix() {
        int n = 10;
        DenseMatrix zero = DenseMatrix.zeros(n);
        DenseMatrix a = createRandomMatrix(n, 42);
        
        DistributedMultiplier distributed = new DistributedMultiplier(sc, 2);
        DenseMatrix result = distributed.multiply(a, zero);
        
        assertMatricesEqual(zero, result, 1e-10);
    }

    @Test
    public void testSmallMatrix() {
        int n = 50;
        DenseMatrix a = createRandomMatrix(n, 42);
        DenseMatrix b = createRandomMatrix(n, 43);
        
        BlockedMultiplier sequential = new BlockedMultiplier(64);
        DenseMatrix expected = sequential.multiply(a, b);
        
        DistributedMultiplier distributed = new DistributedMultiplier(sc, 2);
        DenseMatrix result = distributed.multiply(a, b);
        
        assertMatricesEqual(expected, result, 1e-10);
    }

    @Test
    public void testPartitioning1Worker() {
        int n = 100;
        DenseMatrix a = createRandomMatrix(n, 42);
        DenseMatrix b = createRandomMatrix(n, 43);
        
        BlockedMultiplier sequential = new BlockedMultiplier(64);
        DenseMatrix expected = sequential.multiply(a, b);
        
        DistributedMultiplier distributed = new DistributedMultiplier(sc, 1);
        DenseMatrix result = distributed.multiply(a, b);
        
        assertMatricesEqual(expected, result, 1e-10);
    }

    @Test
    public void testPartitioning4Workers() {
        int n = 100;
        DenseMatrix a = createRandomMatrix(n, 42);
        DenseMatrix b = createRandomMatrix(n, 43);
        
        BlockedMultiplier sequential = new BlockedMultiplier(64);
        DenseMatrix expected = sequential.multiply(a, b);
        
        DistributedMultiplier distributed = new DistributedMultiplier(sc, 4);
        DenseMatrix result = distributed.multiply(a, b);
        
        assertMatricesEqual(expected, result, 1e-10);
    }

    @Test
    public void testChecksumInvariance() {
        int n = 100;
        DenseMatrix a = createRandomMatrix(n, 42);
        DenseMatrix b = createRandomMatrix(n, 43);
        
        DistributedMultiplier distributed1 = new DistributedMultiplier(sc, 1);
        DistributedMultiplier distributed2 = new DistributedMultiplier(sc, 4);
        
        DenseMatrix result1 = distributed1.multiply(a, b);
        DenseMatrix result2 = distributed2.multiply(a, b);
        
        assertEquals(result1.checksum(), result2.checksum(), 1e-10,
            "Checksum should be invariant across different partition counts");
    }

    private DenseMatrix createIdentityMatrix(int n) {
        DenseMatrix identity = DenseMatrix.zeros(n);
        for (int i = 0; i < n; i++) {
            identity.set(i, i, 1.0);
        }
        return identity;
    }

    private DenseMatrix createRandomMatrix(int n, long seed) {
        double[][] values = new double[n][n];
        java.util.Random rng = new java.util.Random(seed);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                values[i][j] = rng.nextDouble();
            }
        }
        return DenseMatrix.wrap(values);
    }

    private void assertMatricesEqual(DenseMatrix expected, DenseMatrix actual, double tolerance) {
        assertEquals(expected.size(), actual.size(), "Matrix sizes must match");
        int n = expected.size();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                assertEquals(expected.get(i, j), actual.get(i, j), tolerance,
                    String.format("Mismatch at (%d, %d)", i, j));
            }
        }
    }
}
