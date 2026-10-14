package es.ulpgc.bigdata.matrixmul.benchmark;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import es.ulpgc.bigdata.matrixmul.sparse.CsrMatrix;
import es.ulpgc.bigdata.matrixmul.sparse.SparseMatrixMultiplier;

/**
 * Kernel-only timing harness for the sparse multiplier.
 *
 * <p>Same protocol as {@link BenchmarkHarness}: a probe multiplication
 * decides the time budget and the batch size, warm-up samples are recorded
 * with {@code warmup=true}, and every multiplication inside a timed sample
 * reuses the multiplier's pre-allocated buffers. The timer surrounds only
 * {@link SparseMatrixMultiplier#multiply}, which builds the output CSR
 * structure (its size is data-dependent, so it cannot be pre-allocated).
 */
public final class SparseBenchmarkHarness {

    private final SparseMatrixMultiplier multiplier;
    private CsrMatrix lastResult;

    public SparseBenchmarkHarness(SparseMatrixMultiplier multiplier) {
        this.multiplier = multiplier;
    }

    /** Result of one benchmark run: every raw sample, the batch size and the last product. */
    public record Measurement(List<TimingSample> samples, int batchSize, CsrMatrix result) {

        public Measurement {
            samples = Collections.unmodifiableList(new ArrayList<>(samples));
        }
    }

    public Measurement measure(CsrMatrix a, CsrMatrix b, int warmups, int repetitions) {
        long probeNs = timeSingle(a, b);
        if (probeNs > BenchmarkHarness.BUDGET_NS) {
            throw new BudgetExceededException(probeNs);
        }
        int batchSize = BenchmarkHarness.chooseBatchSize(probeNs);

        List<TimingSample> samples = new ArrayList<>();
        for (int repetition = 1; repetition <= warmups; repetition++) {
            samples.add(sample(repetition, true, a, b, batchSize));
        }
        for (int repetition = 1; repetition <= repetitions; repetition++) {
            samples.add(sample(repetition, false, a, b, batchSize));
        }
        return new Measurement(samples, batchSize, lastResult);
    }

    private TimingSample sample(int repetition, boolean warmup,
                                CsrMatrix a, CsrMatrix b, int batchSize) {
        long start = System.nanoTime();
        for (int copy = 0; copy < batchSize; copy++) {
            lastResult = multiplier.multiply(a, b);
        }
        long elapsedNs = System.nanoTime() - start;
        double perMultiplyNs = elapsedNs / (double) batchSize;
        return new TimingSample(repetition, warmup, batchSize, perMultiplyNs, lastResult.checksum());
    }

    private long timeSingle(CsrMatrix a, CsrMatrix b) {
        long start = System.nanoTime();
        lastResult = multiplier.multiply(a, b);
        return System.nanoTime() - start;
    }
}
