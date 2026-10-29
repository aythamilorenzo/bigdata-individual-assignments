package es.ulpgc.bigdata.matrixmul.benchmark;

import es.ulpgc.bigdata.matrixmul.BlockedMultiplier;
import es.ulpgc.bigdata.matrixmul.MatrixMultiplier;
import es.ulpgc.bigdata.matrixmul.ParallelBlockedMultiplier;

/**
 * Registry mapping the declared method names of the campaign CSV schema to
 * their implementations, so the CLI and the CSV writer share one vocabulary.
 *
 * <p>Assignment 3 compares exactly two families built on the same kernel:
 * the sequential baseline and its thread-parallel distribution.
 */
public final class Multipliers {

    public static final String METHOD_BLOCKED_SEQUENTIAL = "blocked_ijk_sequential";
    public static final String METHOD_BLOCKED_PARALLEL = "blocked_ijk_parallel";

    /** Tile side of the blocked kernel: 64 doubles x 64 doubles = 32 KB per tile of b. */
    public static final int BLOCK_SIZE = 64;

    private Multipliers() {
    }

    /**
     * Multiplier for {@code method}; the parallel family needs the declared
     * thread count, the sequential family ignores it (it always runs on one
     * thread). The caller owns {@code AutoCloseable} multipliers and must
     * close them after the measurement.
     */
    public static MatrixMultiplier create(String method, int threads) {
        return switch (method) {
            case METHOD_BLOCKED_SEQUENTIAL -> new BlockedMultiplier(BLOCK_SIZE);
            case METHOD_BLOCKED_PARALLEL -> new ParallelBlockedMultiplier(BLOCK_SIZE, threads);
            default -> throw new IllegalArgumentException("unknown method: " + method);
        };
    }

    public static boolean isParallel(String method) {
        return METHOD_BLOCKED_PARALLEL.equals(method);
    }

    /** Release resources of multipliers that own any (thread pools); no-op otherwise. */
    public static void shutdown(MatrixMultiplier multiplier) {
        if (multiplier instanceof AutoCloseable closeable) {
            try {
                closeable.close();
            } catch (Exception exception) {
                throw new IllegalStateException("cannot shut down multiplier", exception);
            }
        }
    }
}
