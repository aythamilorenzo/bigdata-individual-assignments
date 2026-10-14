package es.ulpgc.bigdata.matrixmul.benchmark;

import es.ulpgc.bigdata.matrixmul.BlockedMultiplier;
import es.ulpgc.bigdata.matrixmul.IkjLoopMultiplier;
import es.ulpgc.bigdata.matrixmul.MatrixMultiplier;
import es.ulpgc.bigdata.matrixmul.TripleLoopMultiplier;

/**
 * Registry mapping the declared method names of the campaign CSV schema to
 * their implementations, so the CLI and the CSV writer share one vocabulary.
 */
public final class Multipliers {

    public static final String METHOD_IJK = "triple_loop_ijk_dense";
    public static final String METHOD_IKJ = "triple_loop_ikj_dense";
    public static final String METHOD_BLOCKED = "blocked_ijk_dense";
    public static final String METHOD_SPARSE_CSR = "sparse_csr_gustavson";

    /** Tile side of the blocked kernel: 64 doubles x 64 doubles = 32 KB per tile of b. */
    public static final int BLOCK_SIZE = 64;

    private Multipliers() {
    }

    /** Dense multiplier for {@code method}; throws on unknown names. */
    public static MatrixMultiplier dense(String method) {
        return switch (method) {
            case METHOD_IJK -> new TripleLoopMultiplier();
            case METHOD_IKJ -> new IkjLoopMultiplier();
            case METHOD_BLOCKED -> new BlockedMultiplier(BLOCK_SIZE);
            default -> throw new IllegalArgumentException("unknown dense method: " + method);
        };
    }

    public static boolean isDense(String method) {
        return !METHOD_SPARSE_CSR.equals(method);
    }
}
