package tripleloop

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const (
	referenceSize = 5
	rtol          = 1e-9
	atol          = 1e-12
)

func TestIdentityMatrixPreservesOperand(t *testing.T) {
	n := 4
	matrixA := []float64{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 1.1, 1.2,
		1.3, 1.4, 1.5, 1.6,
	}
	identity := make([]float64, n*n)
	for i := 0; i < n; i++ {
		identity[i*n+i] = 1.0
	}
	out := make([]float64, n*n)

	MultiplyInto(matrixA, identity, out, n)
	assertExact(t, matrixA, out)

	MultiplyInto(identity, matrixA, out, n)
	assertExact(t, matrixA, out)
}

func TestZeroMatrixProductIsZero(t *testing.T) {
	n := 4
	matrixA := make([]float64, n*n)
	zero := make([]float64, n*n)
	out := make([]float64, n*n)
	for index := range matrixA {
		matrixA[index] = float64(index)
	}

	MultiplyInto(matrixA, zero, out, n)
	for index, value := range out {
		if value != 0.0 {
			t.Fatalf("entry %d: got %v, want 0", index, value)
		}
	}
}

func TestKnownAnswerThreeByThree(t *testing.T) {
	matrixA := []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	matrixB := []float64{
		9, 8, 7,
		6, 5, 4,
		3, 2, 1,
	}
	expected := []float64{
		30, 24, 18,
		84, 69, 54,
		138, 114, 90,
	}
	out := make([]float64, 9)

	MultiplyInto(matrixA, matrixB, out, 3)
	assertExact(t, expected, out)
}

func TestMatchesSharedNumPyReference(t *testing.T) {
	dir := filepath.Join("..", "..", "data", "reference")
	matrixA := loadReference(t, filepath.Join(dir, "matrixA_5.bin"))
	matrixB := loadReference(t, filepath.Join(dir, "matrixB_5.bin"))
	expected := loadReference(t, filepath.Join(dir, "expectedC_5.bin"))
	out := make([]float64, referenceSize*referenceSize)

	MultiplyInto(matrixA, matrixB, out, referenceSize)

	for index := range out {
		tolerance := atol + rtol*math.Abs(expected[index])
		if math.Abs(out[index]-expected[index]) > tolerance {
			t.Fatalf("entry %d: got %v, want %v (tolerance %v)",
				index, out[index], expected[index], tolerance)
		}
	}
}

func assertExact(t *testing.T, expected, actual []float64) {
	t.Helper()
	if len(expected) != len(actual) {
		t.Fatalf("length mismatch: %d vs %d", len(expected), len(actual))
	}
	for index := range expected {
		if expected[index] != actual[index] {
			t.Fatalf("entry %d: got %v, want %v", index, actual[index], expected[index])
		}
	}
}

func loadReference(t *testing.T, path string) []float64 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("cannot read %s: %v", path, err)
	}
	values := make([]float64, len(raw)/8)
	for index := range values {
		values[index] = math.Float64frombits(binary.LittleEndian.Uint64(raw[index*8:]))
	}
	return values
}
