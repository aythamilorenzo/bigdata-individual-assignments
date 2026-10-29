package es.ulpgc.bigdata.matrixmul.distributed;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import es.ulpgc.bigdata.matrixmul.DenseMatrix;
import es.ulpgc.bigdata.matrixmul.InputMatrixLoader;
import es.ulpgc.bigdata.matrixmul.BlockedMultiplier;
import es.ulpgc.bigdata.matrixmul.benchmark.BenchmarkHarness;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Simple runner for local testing without spark-submit.
 * Tests distributed multiplication with different thread counts.
 */
public class LocalRunner {
    
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: LocalRunner <size> <output-dir>");
            System.err.println("Example: LocalRunner 1000 data/results");
            System.exit(1);
        }
        
        int size = Integer.parseInt(args[0]);
        String outputDir = args[1];
        
        // Load input matrices
        InputMatrixLoader loader = new InputMatrixLoader();
        Path inputDir = Paths.get("data/inputs");
        DenseMatrix a = loader.load(inputDir.resolve("matrixA_" + size + ".bin"), size);
        DenseMatrix b = loader.load(inputDir.resolve("matrixB_" + size + ".bin"), size);
        
        System.out.println("Loaded matrices: " + size + "x" + size);
        
        // Compute sequential baseline
        System.out.println("Computing sequential baseline...");
        BlockedMultiplier sequential = new BlockedMultiplier(64);
        DenseMatrix cSeq = sequential.multiply(a, b);
        double seqChecksum = cSeq.checksum();
        System.out.println("Sequential checksum: " + seqChecksum);
        
        // Test different configurations
        int[] threadCounts = {1, 2, 4, 8, 12};
        String timestamp = Instant.now().toString().replace(":", "-").substring(0, 19);
        String csvFile = outputDir + "/local_" + size + "_" + timestamp + ".csv";
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(csvFile))) {
            writer.println("size,threads,total_ms,compute_ms,overhead_ms,checksum_match");
            
            for (int threads : threadCounts) {
                System.out.println("\n=== Testing with " + threads + " threads ===");
                
                // Configure Spark
                SparkConf conf = new SparkConf()
                    .setAppName("MatrixMultiplication-Local-" + threads)
                    .setMaster("local[" + threads + "]")
                    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                    .set("spark.kryoserializer.buffer.max", "256m")
                    .set("spark.driver.memory", "2g");
                
                try (JavaSparkContext sc = new JavaSparkContext(conf)) {
                    DistributedMultiplier distributed = new DistributedMultiplier(sc, threads);
                    
                    // Warmup
                    System.out.println("Warmup...");
                    for (int i = 0; i < 2; i++) {
                        distributed.multiply(a, b);
                    }
                    
                    // Measure
                    System.out.println("Measuring...");
                    long totalTime = 0;
                    long computeTime = 0;
                    double checksum = 0;
                    
                    for (int i = 0; i < 3; i++) {
                        long startTotal = System.nanoTime();
                        DenseMatrix c = distributed.multiply(a, b);
                        long endTotal = System.nanoTime();
                        
                        totalTime += (endTotal - startTotal) / 1_000_000;
                        checksum = c.checksum();
                    }
                    
                    totalTime /= 3;
                    computeTime = totalTime; // Simplified for now
                    boolean match = Math.abs(checksum - seqChecksum) < 0.001;
                    
                    System.out.printf("Total time: %d ms%n", totalTime);
                    System.out.printf("Checksum match: %s%n", match);
                    
                    writer.printf("%d,%d,%d,%d,%d,%s%n",
                        size, threads, totalTime, computeTime, totalTime - computeTime, match);
                }
            }
            
            System.out.println("\n✓ Results written to: " + csvFile);
        }
    }
}
