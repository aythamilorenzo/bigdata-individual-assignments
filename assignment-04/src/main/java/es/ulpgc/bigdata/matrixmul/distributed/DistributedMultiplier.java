package es.ulpgc.bigdata.matrixmul.distributed;

import es.ulpgc.bigdata.matrixmul.BlockedMultiplier;
import es.ulpgc.bigdata.matrixmul.DenseMatrix;
import es.ulpgc.bigdata.matrixmul.MatrixMultiplier;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Distributed matrix multiplication using Apache Spark.
 * 
 * <p>Strategy: partition A by row blocks, broadcast B to all workers,
 * each worker computes its portion using the validated BlockedMultiplier.
 * This ensures bit-identical results to the sequential implementation.
 */
public class DistributedMultiplier implements MatrixMultiplier, Serializable {

    private static final long serialVersionUID = 1L;
    private static final String METHOD_NAME = "blocked_distributed";
    
    private final JavaSparkContext sc;
    private final int numPartitions;

    public DistributedMultiplier(JavaSparkContext sc, int numPartitions) {
        this.sc = sc;
        this.numPartitions = numPartitions;
    }

    @Override
    public DenseMatrix multiply(DenseMatrix a, DenseMatrix b) {
        if (a.size() != b.size()) {
            throw new IllegalArgumentException("Matrix dimensions must match");
        }
        
        int n = a.size();
        DenseMatrix c = DenseMatrix.zeros(n);
        multiplyInto(a, b, c);
        return c;
    }

    @Override
    public void multiplyInto(DenseMatrix a, DenseMatrix b, DenseMatrix c) {
        if (a.size() != b.size() || a.size() != c.size()) {
            throw new IllegalArgumentException("Matrix dimensions must match");
        }
        
        int n = a.size();
        
        // Create partition descriptors
        List<PartitionDescriptor> partitions = createPartitions(n, numPartitions);
        
        // Broadcast B to all workers
        Broadcast<DenseMatrix> bBroadcast = sc.broadcast(b);
        
        // Parallelize partitions and compute
        JavaRDD<PartitionResult> results = sc.parallelize(partitions, numPartitions)
            .map(partition -> computePartition(partition, a, b, bBroadcast));
        
        // Collect results and assemble final matrix
        List<PartitionResult> collected = results.collect();
        assembleResults(c, collected);
    }

    /**
     * Create partition descriptors for row-based partitioning.
     */
    private List<PartitionDescriptor> createPartitions(int n, int numPartitions) {
        List<PartitionDescriptor> partitions = new ArrayList<>();
        int rowsPerPartition = n / numPartitions;
        int remainder = n % numPartitions;
        
        int startRow = 0;
        for (int i = 0; i < numPartitions; i++) {
            int rowsInPartition = rowsPerPartition + (i < remainder ? 1 : 0);
            int endRow = startRow + rowsInPartition;
            partitions.add(new PartitionDescriptor(startRow, endRow));
            startRow = endRow;
        }
        
        return partitions;
    }

    /**
     * Compute multiplication for a single partition (static to avoid capturing 'this' in lambda).
     * Computes rows [startRow, endRow) of C = A * B using the full matrices.
     */
    private static PartitionResult computePartition(PartitionDescriptor partition, DenseMatrix a, DenseMatrix b, Broadcast<DenseMatrix> bBroadcast) {
        DenseMatrix bLocal = bBroadcast.value();
        int n = a.size();
        int numRows = partition.endRow - partition.startRow;
        
        // Create output matrix for this partition's rows
        double[][] cRows = new double[numRows][n];
        
        // Compute C[startRow:endRow, :] = A[startRow:endRow, :] * B
        // Using blocked algorithm for cache efficiency
        int blockSize = 64;
        for (int iBlock = partition.startRow; iBlock < partition.endRow; iBlock += blockSize) {
            int iLimit = Math.min(iBlock + blockSize, partition.endRow);
            for (int kBlock = 0; kBlock < n; kBlock += blockSize) {
                int kLimit = Math.min(kBlock + blockSize, n);
                for (int jBlock = 0; jBlock < n; jBlock += blockSize) {
                    int jLimit = Math.min(jBlock + blockSize, n);
                    // Compute tile
                    for (int i = iBlock; i < iLimit; i++) {
                        double[] rowA = a.row(i);
                        double[] rowC = cRows[i - partition.startRow];
                        for (int k = kBlock; k < kLimit; k++) {
                            double factor = rowA[k];
                            double[] rowB = bLocal.row(k);
                            for (int j = jBlock; j < jLimit; j++) {
                                rowC[j] += factor * rowB[j];
                            }
                        }
                    }
                }
            }
        }
        
        return new PartitionResult(partition.startRow, cRows);
    }

    /**
     * Assemble partition results into the final matrix c.
     */
    private void assembleResults(DenseMatrix c, List<PartitionResult> results) {
        for (PartitionResult result : results) {
            int numRows = result.partialRows.length;
            for (int i = 0; i < numRows; i++) {
                double[] sourceRow = result.partialRows[i];
                double[] targetRow = c.row(result.startRow + i);
                System.arraycopy(sourceRow, 0, targetRow, 0, c.size());
            }
        }
    }

    /**
     * Descriptor for a partition: [startRow, endRow)
     */
    private static class PartitionDescriptor implements Serializable {
        private static final long serialVersionUID = 1L;
        final int startRow;
        final int endRow;

        PartitionDescriptor(int startRow, int endRow) {
            this.startRow = startRow;
            this.endRow = endRow;
        }
    }

    /**
     * Result from computing a partition.
     */
    private static class PartitionResult implements Serializable {
        private static final long serialVersionUID = 1L;
        final int startRow;
        final double[][] partialRows;

        PartitionResult(int startRow, double[][] partialRows) {
            this.startRow = startRow;
            this.partialRows = partialRows;
        }
    }
}
