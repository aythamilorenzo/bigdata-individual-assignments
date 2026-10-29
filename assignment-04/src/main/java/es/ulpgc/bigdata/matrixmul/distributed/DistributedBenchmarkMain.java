package es.ulpgc.bigdata.matrixmul.distributed;

import es.ulpgc.bigdata.matrixmul.BlockedMultiplier;
import es.ulpgc.bigdata.matrixmul.DenseMatrix;
import es.ulpgc.bigdata.matrixmul.InputMatrixLoader;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Main entry point for distributed matrix multiplication benchmarks.
 * 
 * <p>Measures timing breakdown: partition, broadcast, compute, collect.
 * Outputs results to CSV format for analysis.
 */
public class DistributedBenchmarkMain {

    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println("Usage: DistributedBenchmarkMain <size> <input-dir> <output-csv>");
            System.err.println("Example: DistributedBenchmarkMain 2000 data/inputs results/distributed_time_raw.csv");
            System.exit(1);
        }

        int n = Integer.parseInt(args[0]);
        Path inputDir = Path.of(args[1]);
        String outputCsv = args[2];
        
        // Spark configuration
        SparkConf conf = new SparkConf()
            .setAppName("MatrixMultiplicationBenchmark")
            .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .set("spark.kryoserializer.buffer.max", "256m")
            .set("spark.sql.shuffle.partitions", "8");
        
        // If running locally, use local[*] by default
        if (!conf.contains("spark.master")) {
            conf.setMaster("local[*]");
        }
        
        try (JavaSparkContext sc = new JavaSparkContext(conf)) {
            System.out.println("Spark Master: " + conf.get("spark.master"));
            System.out.println("Running benchmark with n=" + n);
            
            // Load input matrices
            System.out.println("Loading input matrices...");
            InputMatrixLoader loader = new InputMatrixLoader();
            DenseMatrix a = loader.load(inputDir.resolve("matrixA_" + n + ".bin"), n);
            DenseMatrix b = loader.load(inputDir.resolve("matrixB_" + n + ".bin"), n);
            
            // Sequential baseline for comparison
            System.out.println("Computing sequential baseline...");
            BlockedMultiplier sequential = new BlockedMultiplier(64);
            DenseMatrix cSequential = sequential.multiply(a, b);
            long checksumExpected = computeChecksum(cSequential);
            System.out.println("Sequential checksum: " + checksumExpected);
            
            // Run distributed benchmarks with different worker counts
            int[] workerCounts = {1, 2, 4, 8};
            List<BenchmarkResult> results = new ArrayList<>();
            
            for (int workers : workerCounts) {
                System.out.println("\n=== Testing with " + workers + " workers ===");
                
                // Warmup
                System.out.println("Warmup...");
                for (int i = 0; i < 3; i++) {
                    DistributedMultiplier distributed = new DistributedMultiplier(sc, workers);
                    distributed.multiply(a, b);
                }
                
                // Measured runs
                System.out.println("Running 7 measured iterations...");
                for (int rep = 0; rep < 7; rep++) {
                    TimingBreakdown timing = runTimedBenchmark(sc, a, b, workers, checksumExpected);
                    results.add(new BenchmarkResult(n, workers, rep + 1, timing));
                    
                    System.out.printf("  Rep %d: total=%.3fs, compute=%.3fs, overhead=%.3fs%n",
                        rep + 1,
                        timing.totalMs / 1000.0,
                        timing.computeMs / 1000.0,
                        (timing.totalMs - timing.computeMs) / 1000.0);
                }
            }
            
            // Write results to CSV
            writeResultsToCsv(results, outputCsv);
            System.out.println("\nResults written to: " + outputCsv);
            
        } catch (Exception e) {
            System.err.println("Error during benchmark: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    private static TimingBreakdown runTimedBenchmark(JavaSparkContext sc, DenseMatrix a, DenseMatrix b, 
                                                      int workers, long expectedChecksum) {
        TimingBreakdown timing = new TimingBreakdown();
        
        // Total end-to-end timing
        long startTotal = System.nanoTime();
        
        // Create distributed multiplier
        DistributedMultiplier distributed = new DistributedMultiplier(sc, workers);
        
        // Compute
        long startCompute = System.nanoTime();
        DenseMatrix c = distributed.multiply(a, b);
        long endCompute = System.nanoTime();
        
        long endTotal = System.nanoTime();
        
        // Verify correctness
        long checksumActual = computeChecksum(c);
        if (checksumActual != expectedChecksum) {
            throw new RuntimeException("Checksum mismatch! Expected " + expectedChecksum + 
                                     " but got " + checksumActual);
        }
        
        timing.totalMs = (endTotal - startTotal) / 1_000_000.0;
        timing.computeMs = (endCompute - startCompute) / 1_000_000.0;
        
        return timing;
    }
    
    private static long computeChecksum(DenseMatrix matrix) {
        long checksum = 0;
        int n = matrix.size();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                checksum += Double.doubleToLongBits(matrix.get(i, j));
            }
        }
        return checksum;
    }
    
    private static void writeResultsToCsv(List<BenchmarkResult> results, String outputPath) throws IOException {
        Path path = Paths.get(outputPath);
        path.toFile().getParentFile().mkdirs();
        
        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(path.toFile())))) {
            // Header
            writer.println("size,workers,repetition,total_ms,compute_ms,overhead_ms,timestamp");
            
            // Data rows
            for (BenchmarkResult result : results) {
                writer.printf("%d,%d,%d,%.3f,%.3f,%.3f,%s%n",
                    result.size,
                    result.workers,
                    result.repetition,
                    result.timing.totalMs,
                    result.timing.computeMs,
                    result.timing.totalMs - result.timing.computeMs,
                    Instant.now().toString());
            }
        }
    }
    
    private static class TimingBreakdown {
        double totalMs;
        double computeMs;
    }
    
    private static class BenchmarkResult {
        final int size;
        final int workers;
        final int repetition;
        final TimingBreakdown timing;
        
        BenchmarkResult(int size, int workers, int repetition, TimingBreakdown timing) {
            this.size = size;
            this.workers = workers;
            this.repetition = repetition;
            this.timing = timing;
        }
    }
}
