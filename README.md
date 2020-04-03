# Matrix Multiplication

This implementation evaluates,

* A naive cpu version,
* A naive gpu version and,
* A shared memory optimized gpu version.

## Usage

`make`

`./matmul <matrix size>`

BLOCK_SIZE can be set by altering the Makefile.

## Kernel Details

* serial_matmul: sequential triply nested loop implementation.
* naive_matmul: every thread computes one element in the output matrix.
* shared_matmul: titled implementation that make use of GPU shared memory.
* run_serial: run the serial version and produce timing info.
* run_naive: run the naive GPU version and produce timing info.
* run_shared: run the optimized GPU version and produce timing info.
* check: check the result for correctness (for ones matrices).  