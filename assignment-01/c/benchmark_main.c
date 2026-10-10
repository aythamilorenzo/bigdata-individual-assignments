#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "triple_loop_mul.h"

#define BATCH_THRESHOLD_NS 2000000L
#define BUDGET_NS 60000000000L
#define MAX_BATCH 100000

#define EXIT_OK 0
#define EXIT_INVALID_ARGUMENTS 2
#define EXIT_BUDGET_EXCEEDED 3
#define EXIT_MISSING_INPUT 4

#define TIME_HEADER \
    "size,language,method,numeric_type,repetition,warmup,batch_size," \
    "time_ns_per_multiply,checksum\n"
#define MEMORY_HEADER "size,language,metric,tool,value_bytes,theoretical_bytes\n"

typedef struct {
    const char *inputs;
    int size;
    const char *time_csv;
    const char *memory_csv;
    int warmup;
    int repetitions;
} arguments_t;

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s --inputs DIR --size N --time-csv FILE --memory-csv FILE "
            "--warmup W --repetitions R\n",
            program);
}

static int parse_arguments(int argc, char **argv, arguments_t *out) {
    memset(out, 0, sizeof(*out));
    out->warmup = 3;
    out->repetitions = 7;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "--inputs") == 0) {
            out->inputs = argv[i + 1];
        } else if (strcmp(argv[i], "--size") == 0) {
            out->size = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--time-csv") == 0) {
            out->time_csv = argv[i + 1];
        } else if (strcmp(argv[i], "--memory-csv") == 0) {
            out->memory_csv = argv[i + 1];
        } else if (strcmp(argv[i], "--warmup") == 0) {
            out->warmup = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--repetitions") == 0) {
            out->repetitions = atoi(argv[i + 1]);
        } else {
            return -1;
        }
    }
    if (out->inputs == NULL || out->size <= 0 || out->time_csv == NULL ||
        out->memory_csv == NULL) {
        return -1;
    }
    return 0;
}

static int64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static size_t peak_rss_bytes(void) {
    FILE *status = fopen("/proc/self/status", "r");
    if (status == NULL) {
        return 0;
    }
    char line[256];
    long long kibibytes = 0;
    while (fgets(line, sizeof(line), status) != NULL) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            sscanf(line + 6, "%lld", &kibibytes);
            break;
        }
    }
    fclose(status);
    return (size_t)kibibytes * 1024UL;
}

static double *load_matrix(const char *path, int size) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    size_t elements = (size_t)size * (size_t)size;
    double *matrix = (double *)malloc(elements * sizeof(double));
    if (matrix == NULL) {
        fclose(file);
        fprintf(stderr, "out of memory loading %s\n", path);
        return NULL;
    }
    size_t read = fread(matrix, sizeof(double), elements, file);
    fclose(file);
    if (read != elements) {
        fprintf(stderr, "%s: expected %zu doubles, got %zu\n", path, elements, read);
        free(matrix);
        return NULL;
    }
    return matrix;
}

static int64_t timed_multiply(const double *a, const double *b, double *out, int size) {
    int64_t start = monotonic_ns();
    triple_loop_mul_ijk(a, b, out, (size_t)size);
    return monotonic_ns() - start;
}

static double checksum_row_major(const double *out, size_t elements) {
    double total = 0.0;
    for (size_t index = 0; index < elements; index++) {
        total += out[index];
    }
    return total;
}

static int file_needs_header(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    int first = fgetc(file);
    fclose(file);
    return first == EOF;
}

static int append_memory_rows(const arguments_t *args, size_t peak_rss) {
    FILE *file = fopen(args->memory_csv, "a");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s: %s\n", args->memory_csv, strerror(errno));
        return -1;
    }
    if (file_needs_header(args->memory_csv)) {
        fputs(MEMORY_HEADER, file);
    }
    long long theoretical = 24LL * args->size * args->size;
    fprintf(file, "%d,c,peak_rss,/proc/self/status VmHWM,%zu,%lld\n",
            args->size, peak_rss, theoretical);
    fprintf(file, "%d,c,theoretical_array_storage,analytical,%lld,%lld\n",
            args->size, theoretical, theoretical);
    fclose(file);
    return 0;
}

int main(int argc, char **argv) {
    arguments_t args;
    if (parse_arguments(argc, argv, &args) != 0) {
        usage(argv[0]);
        return EXIT_INVALID_ARGUMENTS;
    }

    char path_a[1024];
    char path_b[1024];
    snprintf(path_a, sizeof(path_a), "%s/matrixA_%d.bin", args.inputs, args.size);
    snprintf(path_b, sizeof(path_b), "%s/matrixB_%d.bin", args.inputs, args.size);

    double *matrix_a = load_matrix(path_a, args.size);
    if (matrix_a == NULL) {
        return EXIT_MISSING_INPUT;
    }
    double *matrix_b = load_matrix(path_b, args.size);
    if (matrix_b == NULL) {
        free(matrix_a);
        return EXIT_MISSING_INPUT;
    }
    size_t elements = (size_t)args.size * (size_t)args.size;
    double *out = (double *)malloc(elements * sizeof(double));
    if (out == NULL) {
        fprintf(stderr, "out of memory allocating the output matrix\n");
        free(matrix_a);
        free(matrix_b);
        return EXIT_BUDGET_EXCEEDED;
    }

    int64_t probe_ns = timed_multiply(matrix_a, matrix_b, out, args.size);
    if (probe_ns > BUDGET_NS) {
        fprintf(stderr, "budget exceeded at n=%d: %.1f s\n", args.size, probe_ns / 1e9);
        free(matrix_a);
        free(matrix_b);
        free(out);
        return EXIT_BUDGET_EXCEEDED;
    }
    int batch_size = 1;
    if (probe_ns < BATCH_THRESHOLD_NS) {
        long batch = (BATCH_THRESHOLD_NS + probe_ns - 1) / probe_ns;
        if (batch > MAX_BATCH) {
            batch = MAX_BATCH;
        }
        batch_size = (int)batch;
    }

    FILE *time_file = fopen(args.time_csv, "a");
    if (time_file == NULL) {
        fprintf(stderr, "cannot open %s: %s\n", args.time_csv, strerror(errno));
        free(matrix_a);
        free(matrix_b);
        free(out);
        return EXIT_INVALID_ARGUMENTS;
    }
    if (file_needs_header(args.time_csv)) {
        fputs(TIME_HEADER, time_file);
    }
    double checksum = 0.0;
    for (int repetition = 1; repetition <= args.warmup + args.repetitions; repetition++) {
        int64_t start = monotonic_ns();
        for (int copy = 0; copy < batch_size; copy++) {
            triple_loop_mul_ijk(matrix_a, matrix_b, out, (size_t)args.size);
        }
        int64_t elapsed_ns = monotonic_ns() - start;
        checksum = checksum_row_major(out, elements);
        int is_warmup = repetition <= args.warmup;
        fprintf(time_file,
                "%d,c,triple_loop_ijk_dense,float64,%d,%s,%d,%.3f,%.17g\n",
                args.size, is_warmup ? repetition : repetition - args.warmup,
                is_warmup ? "true" : "false", batch_size,
                elapsed_ns / (double)batch_size, checksum);
    }
    fclose(time_file);

    size_t peak_rss = peak_rss_bytes();
    if (append_memory_rows(&args, peak_rss) != 0) {
        free(matrix_a);
        free(matrix_b);
        free(out);
        return EXIT_INVALID_ARGUMENTS;
    }
    printf("c n=%d batch=%d checksum=%.17g\n", args.size, batch_size, checksum);

    free(matrix_a);
    free(matrix_b);
    free(out);
    return EXIT_OK;
}
