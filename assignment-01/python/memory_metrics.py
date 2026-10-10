"""Peak resident set size of the current process from /proc/self/status (VmHWM).

The C, Java and Go runners read the very same kernel counter, so the four
memory figures share one metric (peak RSS) and one tool. This is a physical
memory quantity and is not comparable with theoretical array storage or with
the Java peak heap as if they were identical.
"""

from __future__ import annotations

PROC_STATUS = "/proc/self/status"
PEAK_RSS_FIELD = "VmHWM:"


def peak_rss_bytes() -> int:
    """Return the process peak resident set size in bytes."""
    with open(PROC_STATUS, encoding="ascii") as handle:
        for line in handle:
            if line.startswith(PEAK_RSS_FIELD):
                kibibytes = int(line.split()[1])
                return kibibytes * 1024
    raise RuntimeError(f"{PEAK_RSS_FIELD} not found in {PROC_STATUS}")
