// Package tripleloop provides the basic dense i-j-k triple-loop multiplication
// used as the Assignment 1 baseline. Matrices are flat row-major slices of
// float64 (IEEE-754 binary64), matching the storage layout of the C and Java
// implementations.
package tripleloop

// MultiplyInto computes out = a * b for three flat row-major n x n matrices.
// It overwrites every entry of out and must not alias a or b.
func MultiplyInto(a, b, out []float64, n int) {
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			total := 0.0
			for k := 0; k < n; k++ {
				total += a[i*n+k] * b[k*n+j]
			}
			out[i*n+j] = total
		}
	}
}

// Checksum returns the row-major left-to-right sum of out; the same order the
// other language runners use, so checksums are directly comparable.
func Checksum(out []float64) float64 {
	total := 0.0
	for _, value := range out {
		total += value
	}
	return total
}
