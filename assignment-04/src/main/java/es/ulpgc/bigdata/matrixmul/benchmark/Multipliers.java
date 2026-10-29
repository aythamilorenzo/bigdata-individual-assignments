package es.ulpgc.bigdata.matrixmul.benchmark;

import es.ulpgc.bigdata.matrixmul.BlockedMultiplier;
import es.ulpgc.bigdata.matrixmul.MatrixMultiplier;

/**
 * Registry mapping the declared method names of the campaign CSV schema to
 * their implementations, so the CLI and the CSV writer share one vocabulary.
 *
 * <p>Assignment 4 focuses on distributed execution: the sequential baseline
 * is retained for comparison, and the distributed multiplier is created
 * separately with a SparkContext.
 */
public final class Multipliers {

    public static final String METHOD_BLOCKED_SEQUENTIAL = "blocked_ijk_sequential";

    /** Tile side of the blocked kernel: 64 doubles x 64 doubles = 32 KB per tile of b. */
    public static final int BLOCK_SIZE = 64;

    private Multipliers() {
    }

    /**
     * Multiplier for {@code method}. Only the sequential baseline is created here;
     * the distributed multiplier requires a SparkContext and is created separately.
     */
    public static MatrixMultiplier create(String method, int threads) {
        return switch (method) {
            case METHOD_BLOCKED_SEQUENTIAL -> new BlockedMultiplier(BLOCK_SIZE);
            default -> throw new IllegalArgumentException("unknown method: " + method);
        };
    }

    public static boolean isSequential(String method) {
        return METHOD_BLOCKED_SEQUENTIAL.equals(method);
    }
}
