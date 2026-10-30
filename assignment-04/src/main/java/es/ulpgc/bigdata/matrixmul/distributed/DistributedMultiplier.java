package es.ulpgc.bigdata.matrixmul.distributed;

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
 * <p>Strategy: A is partitioned by contiguous row blocks carried in the task
 * descriptors (each task receives only its own rows of A), B is broadcast to
 * all workers and each worker computes its rows of C with the validated
 * cache-blocked tile order. Every output row has exactly one writer and
 * accumulates over k in the same ascending order as the sequential and
 * thread-pool kernels, so all three methods are element-wise identical.
 *
 * <p>Each multiplication can be measured with {@link #measure}, which reports
 * the driver-side wall clock of its stages (broadcast of B, job execution,
 * driver-side assembly) plus the compute time summed across workers and the
 * payload returned by the tasks. The stages are NOT additive with the summed
 * worker compute time: workers run concurrently and their compute happens
 * inside the job stage, so the summed compute time may exceed the job wall
 * clock and is reported as a separate quantity.
 */
public class DistributedMultiplier implements MatrixMultiplier, Serializable {

    private static final long serialVersionUID = 1L;
    private static final int BLOCK_SIZE = 64;

    private final JavaSparkContext sc;
    private final int partitions;

    /**
     * Cost of one multiplication. All values are nanoseconds except
     * {@code taskCount} and {@code resultPayloadBytes}.
     */
    public record PhaseTimings(
            long broadcastNs,
            long jobWallNs,
            long assembleNs,
            long taskComputeNs,
            long taskCount,
            long resultPayloadBytes) {

        /** Driver-side end to end cost of one multiplication. */
        public long endToEndNs() {
            return broadcastNs + jobWallNs + assembleNs;
        }
    }

    /** One multiplication with its phase decomposition. */
    public record MeasuredMultiplication(DenseMatrix result, PhaseTimings timings) {
    }

    public DistributedMultiplier(JavaSparkContext sc, int partitions) {
        this.sc = sc;
        this.partitions = partitions;
    }

    @Override
    public DenseMatrix multiply(DenseMatrix a, DenseMatrix b) {
        return measure(a, b).result();
    }

    @Override
    public void multiplyInto(DenseMatrix a, DenseMatrix b, DenseMatrix out) {
        MatrixMultiplier.requireCompatibleSizes(a, b, out);
        copyRows(multiply(a, b), out);
    }

    /** Run one multiplication and capture the cost of each stage. */
    public MeasuredMultiplication measure(DenseMatrix a, DenseMatrix b) {
        if (a.size() != b.size()) {
            throw new IllegalArgumentException("matrix dimensions must match");
        }
        int n = a.size();
        List<PartitionBlock> blocks = createBlocks(a, n, partitions);

        long broadcastStart = System.nanoTime();
        Broadcast<DenseMatrix> bBroadcast = sc.broadcast(b);
        long broadcastNs = System.nanoTime() - broadcastStart;

        JavaRDD<PartitionResult> rdd = sc.parallelize(blocks, partitions)
                .map(block -> computeBlock(block, bBroadcast));

        long jobStart = System.nanoTime();
        List<PartitionResult> collected = rdd.collect();
        long jobWallNs = System.nanoTime() - jobStart;

        long assembleStart = System.nanoTime();
        DenseMatrix out = DenseMatrix.zeros(n);
        assemble(out, collected);
        long assembleNs = System.nanoTime() - assembleStart;

        long taskComputeNs = 0L;
        long resultPayloadBytes = 0L;
        for (PartitionResult part : collected) {
            taskComputeNs += part.taskComputeNs;
            resultPayloadBytes += (long) part.rows.length * n * Double.BYTES;
        }

        PhaseTimings timings = new PhaseTimings(
                broadcastNs, jobWallNs, assembleNs,
                taskComputeNs, collected.size(), resultPayloadBytes);
        return new MeasuredMultiplication(out, timings);
    }

    /**
     * Split A into contiguous row blocks, one per task; block boundaries
     * differ by at most one row block. Each block carries its own rows of A,
     * so a task only ever receives the rows it multiplies.
     */
    private List<PartitionBlock> createBlocks(DenseMatrix a, int n, int count) {
        List<PartitionBlock> blocks = new ArrayList<>();
        int rowsPerBlock = n / count;
        int remainder = n % count;
        int startRow = 0;
        for (int i = 0; i < count; i++) {
            int endRow = startRow + rowsPerBlock + (i < remainder ? 1 : 0);
            double[][] rows = new double[endRow - startRow][n];
            for (int r = startRow; r < endRow; r++) {
                System.arraycopy(a.row(r), 0, rows[r - startRow], 0, n);
            }
            blocks.add(new PartitionBlock(startRow, rows));
            startRow = endRow;
        }
        return blocks;
    }

    /**
     * Compute rows of C = A_block x B on the worker, timing the tile loops so
     * the driver can report compute cost summed across workers. Static to
     * avoid capturing the multiplier (and its SparkContext) in the task
     * closure.
     */
    private static PartitionResult computeBlock(PartitionBlock block,
                                                Broadcast<DenseMatrix> bBroadcast) {
        DenseMatrix b = bBroadcast.value();
        int numRows = block.rowsA.length;
        if (numRows == 0) {
            return new PartitionResult(block.startRow, new double[0][0], 0L);
        }
        int n = block.rowsA[0].length;
        double[][] rows = new double[numRows][n];

        long computeStart = System.nanoTime();
        for (int iLocal = 0; iLocal < numRows; iLocal += BLOCK_SIZE) {
            int iLimit = Math.min(iLocal + BLOCK_SIZE, numRows);
            for (int kBlock = 0; kBlock < n; kBlock += BLOCK_SIZE) {
                int kLimit = Math.min(kBlock + BLOCK_SIZE, n);
                for (int jBlock = 0; jBlock < n; jBlock += BLOCK_SIZE) {
                    int jLimit = Math.min(jBlock + BLOCK_SIZE, n);
                    for (int i = iLocal; i < iLimit; i++) {
                        double[] rowA = block.rowsA[i];
                        double[] rowC = rows[i];
                        for (int k = kBlock; k < kLimit; k++) {
                            double factor = rowA[k];
                            double[] rowB = b.row(k);
                            for (int j = jBlock; j < jLimit; j++) {
                                rowC[j] += factor * rowB[j];
                            }
                        }
                    }
                }
            }
        }
        long computeNs = System.nanoTime() - computeStart;

        return new PartitionResult(block.startRow, rows, computeNs);
    }

    private static void assemble(DenseMatrix out, List<PartitionResult> parts) {
        for (PartitionResult part : parts) {
            for (int i = 0; i < part.rows.length; i++) {
                double[] source = part.rows[i];
                double[] target = out.row(part.startRow + i);
                System.arraycopy(source, 0, target, 0, out.size());
            }
        }
    }

    private static void copyRows(DenseMatrix source, DenseMatrix target) {
        for (int i = 0; i < source.size(); i++) {
            double[] sourceRow = source.row(i);
            double[] targetRow = target.row(i);
            System.arraycopy(sourceRow, 0, targetRow, 0, source.size());
        }
    }

    /** One contiguous row block of A with its starting row in the output. */
    private static class PartitionBlock implements Serializable {
        private static final long serialVersionUID = 1L;
        final int startRow;
        final double[][] rowsA;

        PartitionBlock(int startRow, double[][] rowsA) {
            this.startRow = startRow;
            this.rowsA = rowsA;
        }
    }

    private static class PartitionResult implements Serializable {
        private static final long serialVersionUID = 1L;
        final int startRow;
        final double[][] rows;
        final long taskComputeNs;

        PartitionResult(int startRow, double[][] rows, long taskComputeNs) {
            this.startRow = startRow;
            this.rows = rows;
            this.taskComputeNs = taskComputeNs;
        }
    }
}
