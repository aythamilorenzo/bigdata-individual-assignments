package es.ulpgc.bigdata.matrixmul;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/** Loads shared little-endian float64 row-major matrix files produced by the Python generator. */
public final class InputMatrixLoader {

    /** Read an n x n matrix from {@code path}; the file must hold exactly n*n little-endian doubles. */
    public DenseMatrix load(Path path, int size) {
        long expectedBytes = (long) size * size * Double.BYTES;
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            if (channel.size() != expectedBytes) {
                throw new IllegalArgumentException(
                        path + ": expected " + expectedBytes + " bytes, got " + channel.size());
            }
            ByteBuffer buffer = ByteBuffer.allocate((int) expectedBytes).order(ByteOrder.LITTLE_ENDIAN);
            while (buffer.hasRemaining()) {
                if (channel.read(buffer) < 0) {
                    throw new IOException("unexpected end of file in " + path);
                }
            }
            buffer.flip();
            double[][] rows = new double[size][size];
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    rows[i][j] = buffer.getDouble();
                }
            }
            return DenseMatrix.wrap(rows);
        } catch (IOException exception) {
            throw new IllegalStateException("cannot read " + path, exception);
        }
    }
}
