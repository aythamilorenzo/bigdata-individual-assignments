package es.ulpgc.bigdata.matrixmul.benchmark;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;

/**
 * Peak Java heap usage from {@code MemoryPoolMXBean.getPeakUsage()}.
 *
 * <p>This is a managed-heap quantity and is reported separately from the
 * process peak working set; the two must not be compared as if identical.
 */
public final class PeakHeap {

    private PeakHeap() {
    }

    /** Reset the recorded peak of every heap pool before a measurement run. */
    public static void reset() {
        for (MemoryPoolMXBean pool : heapPools()) {
            pool.resetPeakUsage();
        }
    }

    /** Sum of the heap pools' peak usage in bytes since the last {@link #reset()}. */
    public static long usedBytes() {
        long total = 0L;
        for (MemoryPoolMXBean pool : heapPools()) {
            total += pool.getPeakUsage().getUsed();
        }
        return total;
    }

    private static Iterable<MemoryPoolMXBean> heapPools() {
        return ManagementFactory.getMemoryPoolMXBeans().stream()
                .filter(pool -> pool.getType() == MemoryType.HEAP)
                .toList();
    }
}
