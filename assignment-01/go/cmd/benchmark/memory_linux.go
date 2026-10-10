//go:build linux

package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// peakResidentMemoryBytes reads the VmHWM watermark from /proc/self/status,
// the same kernel counter reported by the C, Java and Python runners.
func peakResidentMemoryBytes() (uint64, error) {
	file, err := os.Open("/proc/self/status")
	if err != nil {
		return 0, fmt.Errorf("open /proc/self/status: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "VmHWM:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0, fmt.Errorf("unexpected VmHWM line: %q", line)
		}
		kibibytes, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			return 0, fmt.Errorf("parse VmHWM value: %w", err)
		}
		return kibibytes * 1024, nil
	}
	if err := scanner.Err(); err != nil {
		return 0, fmt.Errorf("read /proc/self/status: %w", err)
	}
	return 0, fmt.Errorf("VmHWM not found in /proc/self/status")
}
