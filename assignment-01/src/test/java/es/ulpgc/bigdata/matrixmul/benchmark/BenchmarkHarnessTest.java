package es.ulpgc.bigdata.matrixmul.benchmark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class BenchmarkHarnessTest {

    @Test
    void largeMultiplicationsUseSingleSampleBatches() {
        assertEquals(1, BenchmarkHarness.chooseBatchSize(5_000_000_000L));
        assertEquals(1, BenchmarkHarness.chooseBatchSize(BenchmarkHarness.BATCH_THRESHOLD_NS));
    }

    @Test
    void shortMultiplicationsAreBatchedUpToTheThreshold() {
        assertEquals(2, BenchmarkHarness.chooseBatchSize(1_000_000L));
        assertEquals(4, BenchmarkHarness.chooseBatchSize(500_000L));
    }

    @Test
    void batchSizeNeverExceedsTheDeclaredCap() {
        assertEquals(BenchmarkHarness.MAX_BATCH, BenchmarkHarness.chooseBatchSize(1L));
    }
}
