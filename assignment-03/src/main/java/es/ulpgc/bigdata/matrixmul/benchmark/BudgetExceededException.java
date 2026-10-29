package es.ulpgc.bigdata.matrixmul.benchmark;

/** Raised when a single multiplication exceeds the declared 60 s time budget. */
public final class BudgetExceededException extends RuntimeException {

    private final long probeNs;

    public BudgetExceededException(long probeNs) {
        super("single multiplication took " + probeNs / 1e9 + " s");
        this.probeNs = probeNs;
    }

    public long probeNs() {
        return probeNs;
    }
}
