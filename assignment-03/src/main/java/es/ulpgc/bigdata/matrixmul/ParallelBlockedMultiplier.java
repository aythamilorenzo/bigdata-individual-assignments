package es.ulpgc.bigdata.matrixmul;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Parallel cache-blocked multiplication: the i-tiles of
 * {@link BlockedMultiplier} are partitioned across a fixed pool of threads.
 *
 * <p>Partitioning rule: tile {@code t} of {@code ceil(n / blockSize)} is owned
 * by thread {@code t * threads / tiles} (contiguous ranges, sizes differing by
 * at most one tile). Each owner zeroes the rows of its own tiles and then runs
 * the exact {@link BlockedMultiplier#multiplyTile} computation, so every output
 * entry is written by exactly one thread and accumulates over k in the same
 * ascending order as the sequential kernel: results are element-wise identical
 * to {@link BlockedMultiplier} and no synchronisation of the arithmetic is
 * needed (partitioned writes only, never a shared location).
 *
 * <p>The worker threads are created and prestarted in the constructor, outside
 * the timed region of the benchmark: thread creation is setup, like allocating
 * the output buffer. The timed region of {@link #multiplyInto} therefore
 * contains the per-multiplication dispatch and join of the p tasks plus the
 * shared tile kernel — the framework overhead that the p = 1 configuration
 * makes visible.
 */
public final class ParallelBlockedMultiplier implements MatrixMultiplier, AutoCloseable {

    private final BlockedMultiplier kernel;
    private final int threads;
    private final ThreadPoolExecutor pool;

    /** Parallel blocked kernel with {@code threads >= 1} workers and the given tile side. */
    public ParallelBlockedMultiplier(int blockSize, int threads) {
        if (threads < 1) {
            throw new IllegalArgumentException("threads must be >= 1, got " + threads);
        }
        this.kernel = new BlockedMultiplier(blockSize);
        this.threads = threads;
        this.pool = new ThreadPoolExecutor(threads, threads,
                0L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<>());
        this.pool.prestartAllCoreThreads();
    }

    public int threads() {
        return threads;
    }

    public int blockSize() {
        return kernel.blockSize();
    }

    @Override
    public void multiplyInto(DenseMatrix a, DenseMatrix b, DenseMatrix out) {
        MatrixMultiplier.requireCompatibleSizes(a, b, out);
        int n = a.size();
        int blockSize = kernel.blockSize();
        int tiles = (n + blockSize - 1) / blockSize;
        List<Future<?>> futures = new ArrayList<>(threads);
        for (int thread = 0; thread < threads; thread++) {
            int fromTile = thread * tiles / threads;
            int toTile = (thread + 1) * tiles / threads;
            futures.add(pool.submit(() -> {
                for (int tile = fromTile; tile < toTile; tile++) {
                    runTile(a, b, out, n, blockSize, tile);
                }
            }));
        }
        awaitAll(futures);
    }

    /** Zero and multiply one i-tile; the only writer of its rows of {@code out}. */
    private static void runTile(DenseMatrix a, DenseMatrix b, DenseMatrix out,
                                int n, int blockSize, int tile) {
        int iBlock = tile * blockSize;
        int iLimit = Math.min(iBlock + blockSize, n);
        for (int i = iBlock; i < iLimit; i++) {
            Arrays.fill(out.row(i), 0.0);
        }
        for (int kBlock = 0; kBlock < n; kBlock += blockSize) {
            int kLimit = Math.min(kBlock + blockSize, n);
            for (int jBlock = 0; jBlock < n; jBlock += blockSize) {
                int jLimit = Math.min(jBlock + blockSize, n);
                BlockedMultiplier.multiplyTile(a, b, out, iBlock, iLimit, kBlock, kLimit,
                        jBlock, jLimit);
            }
        }
    }

    private static void awaitAll(List<Future<?>> futures) {
        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while awaiting parallel tiles",
                        exception);
            } catch (ExecutionException exception) {
                throw new IllegalStateException("parallel tile failed", exception.getCause());
            }
        }
    }

    /** Release the worker threads; the multiplier must not be used afterwards. */
    @Override
    public void close() {
        pool.shutdown();
    }
}
