#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

void serial_matmul(float * a, float * b, float * c, int N)
{
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            float sum = 0;
            for (int k = 0; k < N; k++){
                sum += a[i*N+k]*b[k*N+j];
            }
            c[i*N+j] = sum;
            
        }
    }
}

__global__
void naive_matmul(float * a, float * b, float * c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // #row 
    int col = blockIdx.x * blockDim.x + threadIdx.x; // #col
    float sum = 0.0; // holding the dot product of row and col
    if( col < N && row < N) 
    {
        // compute the dot product
        for(int i = 0; i < N; i++)
        {
            sum += a[row * N + i] * b[i * N + col];
        }
        // write the value back to global memory
        c[row * N + col] = sum;
    }
}

__global__
void shared_matmul(float * a, float * b, float * c, int N)
{   
    // tiles in the shared memory
    __shared__ float ds_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float ds_b[BLOCK_SIZE][BLOCK_SIZE];
    // thread and block ids
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x;  int by = blockIdx.y;
    // rows and cols
    int row = by * blockDim.y + ty; 
    int col = bx * blockDim.x + tx;
    // dot product of row and col
    float val = 0.0;
    // for each block
    for (int r = 0; r < (N+BLOCK_SIZE-1)/BLOCK_SIZE; ++r){
        if (row < N && (r * BLOCK_SIZE + tx) < N)
            ds_a[ty][tx] = a[row * N + (r * BLOCK_SIZE + tx)];
        else
            ds_a[ty][tx] = 0.0;
        
        if (col < N && (r * BLOCK_SIZE + ty) < N)
            ds_b[ty][tx] = b[(r * BLOCK_SIZE + ty) * N + col];
        else
            ds_b[ty][tx] = 0.0;
        __syncthreads(); // sync threads after copying to shared memory
        
        for (int i = 0; i < BLOCK_SIZE; ++i)
            val += ds_a[ty][i] * ds_b[i][tx];
        __syncthreads(); // sync threads after computing the dot product
    }
    // write the value to global memory
    if (row < N && col < N)
        c[row * N + col] = val;
}

void init(float *& h_a, float *& h_b, float *& h_c, int N)
{
    // inputs
    cudaMallocHost((void **)&h_a, sizeof(float)*N*N);
    cudaMallocHost((void **)&h_b, sizeof(float)*N*N);
    // init input
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            h_a[i*N+j] = 1.0;
            h_b[i*N+j] = 1.0;
        }
    }
    // output
    cudaMallocHost((void **)&h_c, sizeof(float)*N*N);
    // init output
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            h_c[i*N+j] = 0.0;
        }
    }
}

bool check(float * c, int N){
    // check the result of ones matrices multiplication
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if (c[i*N+j] != float(N))
                return false;
        }
    }
    return true;
}

void run_serial(int N)
{
    // input and output
    float *h_a, *h_b, *h_c; 
    init(h_a, h_b, h_c, N);
    clock_t t;
    t = clock();
    serial_matmul(h_a, h_b, h_c, N);
    t = clock() - t;
    double elapsed_time = ((double)t*1000)/CLOCKS_PER_SEC; 
    // check
    if (check(h_c, N))
        cout << "CPU Serial: Check OK, Time: " << elapsed_time << " ms" << endl;
    else 
        cout << "CPU Serial: Check Failed, Time: " << elapsed_time << " ms" << endl;
}

void run_naive(int N)
{
    // input and output
    float *h_a, *h_b, *h_c; 
    init(h_a, h_b, h_c, N);

    // device memory allocate
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(float)*N*N);
    cudaMalloc((void **) &d_b, sizeof(float)*N*N);
    cudaMalloc((void **) &d_c, sizeof(float)*N*N);

    // gpu params
    dim3 dimGrid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    // events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // start gpu time
    cudaEventRecord(start, 0);
    // memcpy
    cudaMemcpy(d_a, h_a, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    // launch kernel
    naive_matmul<<<dimGrid, dimBlock>>>(d_a,d_b,d_c,N);
    // memcpy back
    cudaMemcpy(h_c, d_c, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // stop gpu time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_time; 
    // compute time elapse on GPU computing
    cudaEventElapsedTime(&elapsed_time, start, stop);
    // check
    if (check(h_c, N))
        cout << "GPU Naive: Check OK, Time: " << elapsed_time << " ms" << endl;
    else 
        cout << "GPU Naive: Check Failed, Time: " << elapsed_time << " ms" << endl;
}

void run_shared(int N)
{
    // input and output
    float *h_a, *h_b, *h_c; 
    init(h_a, h_b, h_c, N);

    // device memory allocate
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(float)*N*N);
    cudaMalloc((void **) &d_b, sizeof(float)*N*N);
    cudaMalloc((void **) &d_c, sizeof(float)*N*N);

    // gpu params
    dim3 dimGrid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    // events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // start gpu time
    cudaEventRecord(start, 0);
    // memcpy
    cudaMemcpy(d_a, h_a, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    // launch kernel
    shared_matmul<<<dimGrid, dimBlock>>>(d_a,d_b,d_c,N);
    // memcpy back
    cudaMemcpy(h_c, d_c, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // stop gpu time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_time; 
    // compute time elapse on GPU computing
    cudaEventElapsedTime(&elapsed_time, start, stop);
    // check
    if (check(h_c, N))
        cout << "GPU Shared: Check OK, Time: " << elapsed_time << " ms" << endl;
    else 
        cout << "GPU Shared: Check Failed, Time: " << elapsed_time << " ms" << endl;
}

int main(int argc, char ** argv)
{
    if (argc < 2){
        cout << "Usage: ./matmul <size of matrix>" << endl;
        return -1;
    }
    // size
    int N = atoi(argv[1]);
    run_serial(N);
    run_naive(N);
    run_shared(N);

    return 0;
}