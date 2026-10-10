// Command benchmark runs the Go triple-loop measurement with the CLI contract
// shared by the Python, Java, C and Go runners:
//
//	benchmark --inputs DIR --size N --time-csv FILE --memory-csv FILE
//	          --warmup W --repetitions R
//
// Exit codes: 0 ok, 2 invalid arguments, 3 time budget exceeded,
// 4 missing input files.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"es.ulpgc.bigdata.matrixmul/tripleloop"
)

const (
	language           = "go"
	method             = "triple_loop_ijk_dense"
	numericType        = "float64"
	batchThresholdNs   = 2_000_000
	budgetNs           = 60_000_000_000
	maxBatch           = 100_000
	exitOK             = 0
	exitInvalidArgs    = 2
	exitBudgetExceeded = 3
	exitMissingInput   = 4
)

const timeHeader = "size,language,method,numeric_type,repetition,warmup,batch_size,time_ns_per_multiply,checksum"
const memoryHeader = "size,language,metric,tool,value_bytes,theoretical_bytes"

func main() {
	os.Exit(run(os.Args[1:]))
}

// run executes one benchmark configuration and returns the process exit code.
func run(arguments []string) int {
	flags := flag.NewFlagSet("benchmark", flag.ContinueOnError)
	inputs := flags.String("inputs", "", "directory with matrixA_N.bin / matrixB_N.bin")
	size := flags.Int("size", 0, "side length n of the square matrices")
	timeCSV := flags.String("time-csv", "", "raw timing CSV (appended)")
	memoryCSV := flags.String("memory-csv", "", "raw memory CSV (appended)")
	warmup := flags.Int("warmup", 3, "warm-up repetitions excluded from analysis")
	repetitions := flags.Int("repetitions", 7, "measured repetitions")
	if err := flags.Parse(arguments); err != nil {
		return exitInvalidArgs
	}
	if *inputs == "" || *size <= 0 || *timeCSV == "" || *memoryCSV == "" {
		flags.Usage()
		return exitInvalidArgs
	}

	matrixA, err := loadMatrix(filepath.Join(*inputs, fmt.Sprintf("matrixA_%d.bin", *size)), *size)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return exitMissingInput
	}
	matrixB, err := loadMatrix(filepath.Join(*inputs, fmt.Sprintf("matrixB_%d.bin", *size)), *size)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return exitMissingInput
	}
	out := make([]float64, (*size)*(*size))

	probeNs := timeMultiply(matrixA, matrixB, out, *size)
	if probeNs.Nanoseconds() > budgetNs {
		fmt.Printf("budget-exceeded n=%d probe_s=%.1f\n", *size, probeNs.Seconds())
		return exitBudgetExceeded
	}
	batchSize := chooseBatchSize(probeNs)

	rows := make([][]string, 0, *warmup+*repetitions)
	for repetition := 1; repetition <= *warmup+*repetitions; repetition++ {
		start := time.Now()
		for copy := 0; copy < batchSize; copy++ {
			tripleloop.MultiplyInto(matrixA, matrixB, out, *size)
		}
		perMultiply := float64(time.Since(start).Nanoseconds()) / float64(batchSize)
		rows = append(rows, timeRow(*size, repetition, repetition <= *warmup, batchSize,
			perMultiply, tripleloop.Checksum(out)))
	}

	if err := appendRows(*timeCSV, timeHeader, rows); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return exitInvalidArgs
	}
	peakRss, err := peakResidentMemoryBytes()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return exitInvalidArgs
	}
	theoretical := int64(24) * int64(*size) * int64(*size)
	memoryRows := [][]string{
		{fmt.Sprint(*size), language, "peak_rss", "/proc/self/status VmHWM", fmt.Sprint(peakRss), fmt.Sprint(theoretical)},
		{fmt.Sprint(*size), language, "theoretical_array_storage", "analytical", fmt.Sprint(theoretical), fmt.Sprint(theoretical)},
	}
	if err := appendRows(*memoryCSV, memoryHeader, memoryRows); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return exitInvalidArgs
	}
	fmt.Printf("go n=%d batch=%d checksum=%.17g\n", *size, batchSize, tripleloop.Checksum(out))
	return exitOK
}

func loadMatrix(path string, size int) ([]float64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cannot read %s: %w", path, err)
	}
	expected := size * size * 8
	if len(raw) != expected {
		return nil, fmt.Errorf("%s: expected %d bytes, got %d", path, expected, len(raw))
	}
	values := make([]float64, size*size)
	for index := range values {
		bits := binary.LittleEndian.Uint64(raw[index*8:])
		values[index] = math.Float64frombits(bits)
	}
	return values, nil
}

func timeMultiply(a, b, out []float64, size int) time.Duration {
	start := time.Now()
	tripleloop.MultiplyInto(a, b, out, size)
	return time.Since(start)
}

func chooseBatchSize(probe time.Duration) int {
	if probe <= 0 {
		fmt.Fprintf(os.Stderr, "non-positive probe duration %v; defaulting to max batch\n", probe)
		return maxBatch
	}
	if probe.Nanoseconds() >= batchThresholdNs {
		return 1
	}
	batch := (batchThresholdNs + probe.Nanoseconds() - 1) / probe.Nanoseconds()
	if batch > maxBatch {
		batch = maxBatch
	}
	return int(batch)
}

func timeRow(size int, repetition int, warmup bool, batchSize int, perMultiplyNs float64, checksum float64) []string {
	warmupText := "false"
	if warmup {
		warmupText = "true"
	}
	return []string{
		fmt.Sprint(size), language, method, numericType,
		fmt.Sprint(repetition), warmupText, fmt.Sprint(batchSize),
		fmt.Sprintf("%.3f", perMultiplyNs), fmt.Sprintf("%.17g", checksum),
	}
}

func appendRows(path string, header string, rows [][]string) error {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("cannot create %s: %w", dir, err)
		}
	}
	needsHeader := true
	if info, err := os.Stat(path); err == nil && info.Size() > 0 {
		needsHeader = false
	}
	file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("cannot open %s: %w", path, err)
	}
	defer file.Close()
	if needsHeader {
		if _, err := fmt.Fprintln(file, header); err != nil {
			return err
		}
	}
	for _, row := range rows {
		if _, err := fmt.Fprintln(file, strings.Join(row, ",")); err != nil {
			return err
		}
	}
	return nil
}
