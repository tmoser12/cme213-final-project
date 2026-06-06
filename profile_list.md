I need you to run the script to collect profiles and timings (run without --profile flag) for all of the 
kernels in each of the kernels/ directories, for both models. All the timings should be collected for the
same set of workloads:

CONFIGS = [
    (1, 1),      # Auto-regressive decoding phase
    (1, 128),    # Short prompt
    (2, 128),    # Batched short prompt
    (8, 128),
    (8, 512),    # Medium prompt
    (16, 1024),  # Long batched prompt
    (1, 1024),
]

These are the configs that I want to collect timings on. The exception is the attention decode kernel, which
should keep its current configs:

CONFIGS = [
    (1, 1, 0), (1, 1, 127), (1, 1, 1024), (1, 1, 4096), (1, 1, 16383),
    (1, 2, 1024), (1, 4, 127), (1, 5, 4096), (1, 8, 16383), (1, 8, 0),
    (2, 1, 1024), (2, 5, 1024),
]

The profiling workloads can remain untouched.