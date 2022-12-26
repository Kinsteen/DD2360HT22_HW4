
#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <fstream>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len)
{
  //@@ Insert code to implement vector addition here
  int id = threadIdx.x + blockDim.x * blockIdx.x;

  if (id < len)
  {
    out[id] = in1[id] + in2[id];
  }
}

//@@ Insert code to implement timer start
double startTimer()
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

//@@ Insert code to implement timer stop
double stopTimer(double start, const char* title)
{
  double time = startTimer() - start;
  printf("Timer for %s: %lf\n", title, time);
  return time;
}

int main(int argc, char **argv)
{
  std::setlocale(LC_ALL, "");
  std::locale::global(std::locale(""));
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  int n_streams = 4;

  //@@ Insert code below to read in inputLength from args
  if (argc > 1)
  {
    inputLength = std::stoi(argv[1]);
  }

  int S_seg = ceil(inputLength / (float) n_streams);

  if (argc > 2) {
    S_seg = std::stoi(argv[2]);
    n_streams = ceil(inputLength / (float) S_seg);
  }

  if (S_seg > inputLength) S_seg = inputLength;

  printf("The input length is %d\n", inputLength);
  printf("S_seg %d\n", S_seg);
  printf("n_streams %d\n", n_streams);

  //@@ Insert code below to allocate Host memory for input and output
  // hostInput1 = (DataType *)malloc(sizeof(DataType) * inputLength);
  // hostInput2 = (DataType *)malloc(sizeof(DataType) * inputLength);
  // hostOutput = (DataType *)malloc(sizeof(DataType) * inputLength);
  // resultRef = (DataType *)malloc(sizeof(DataType) * inputLength);
  cudaMallocHost(&hostInput1, sizeof(DataType) * inputLength);
  cudaMallocHost(&hostInput2, sizeof(DataType) * inputLength);
  cudaMallocHost(&hostOutput, sizeof(DataType) * inputLength);
  cudaMallocHost(&resultRef, sizeof(DataType) * inputLength);

  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  double lower_bound = 0;
  double upper_bound = 10;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::default_random_engine re(std::random_device{}());
  double s = startTimer();
  for (int i = 0; i < inputLength; i++) {
    DataType d1 = unif(re);
    DataType d2 = unif(re);

    hostInput1[i] = d1;
    hostInput2[i] = d2;

    resultRef[i] = d1 + d2;
  }
  stopTimer(s, "CPU result");

  cudaStream_t stream[n_streams];

  for (int i = 0; i < n_streams; i++) {
    cudaStreamCreate(&stream[i]);
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, sizeof(DataType) * inputLength);
  cudaMalloc(&deviceInput2, sizeof(DataType) * inputLength);
  cudaMalloc(&deviceOutput, sizeof(DataType) * inputLength);

  //@@ Initialize the 1D grid and block dimensions here
  int threads_per_block = 512;
  int blocks = ceil(S_seg / (float)threads_per_block);
  printf("TPB: %d\n", threads_per_block);
  printf("Blocks: %d\n", blocks);

  s = startTimer();

  #if 0
  for (int i = 0; i < n_streams; i++) {
    int n = min(S_seg, inputLength - i * S_seg);

    //@@ Insert code to below to Copy memory to the GPU here
    cudaMemcpyAsync(deviceInput1 + i * S_seg, hostInput1 + i * S_seg, sizeof(DataType) * n, cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(deviceInput2 + i * S_seg, hostInput2 + i * S_seg, sizeof(DataType) * n, cudaMemcpyHostToDevice, stream[i]);

    //@@ Launch the GPU Kernel here
    vecAdd<<<blocks, threads_per_block, 0, stream[i]>>>(deviceInput1 + i * S_seg, deviceInput2 + i * S_seg, deviceOutput + i * S_seg, n);

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpyAsync(hostOutput + i * S_seg, deviceOutput + i * S_seg, sizeof(DataType) * n, cudaMemcpyDeviceToHost, stream[i]);
  }

  #else

  for (int i = 0; i < n_streams; i++) {
    int n = min(S_seg, inputLength - i * S_seg);

    //@@ Insert code to below to Copy memory to the GPU here
    cudaMemcpyAsync(deviceInput1 + i * S_seg, hostInput1 + i * S_seg, sizeof(DataType) * n, cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(deviceInput2 + i * S_seg, hostInput2 + i * S_seg, sizeof(DataType) * n, cudaMemcpyHostToDevice, stream[i]);
  }

  for (int i = 0; i < n_streams; i++) {
    int n = min(S_seg, inputLength - i * S_seg);

    //@@ Launch the GPU Kernel here
    vecAdd<<<blocks, threads_per_block, 0, stream[i]>>>(deviceInput1 + i * S_seg, deviceInput2 + i * S_seg, deviceOutput + i * S_seg, n);
  }

  for (int i = 0; i < n_streams; i++) {
    int n = min(S_seg, inputLength - i * S_seg);

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpyAsync(hostOutput + i * S_seg, deviceOutput + i * S_seg, sizeof(DataType) * n, cudaMemcpyDeviceToHost, stream[i]);
  }

  #endif

  cudaDeviceSynchronize();

  double compute_duration = stopTimer(s, "GPU compute");

  std::ofstream outfile;
  outfile.open("result_ex2.txt", std::ios_base::app);
  outfile << inputLength << ";" << S_seg << ";" << n_streams << ";" << compute_duration << std::endl;

  //@@ Insert code below to compare the output with the reference
  bool areVecEqual = true;

  for (int i = 0; i < inputLength; i++) {
    if (hostOutput[i] != resultRef[i]) {
      areVecEqual = false;
    }
  }

  if (areVecEqual) {
    printf("Result are equal!\n");
  } else {
    printf("Results are different\n");
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  for (int i = 0; i < n_streams; i++) {
    cudaStreamDestroy(stream[i]);
  }

  cudaDeviceReset();

  //@@ Free the CPU memory here
  cudaFree(hostInput1);
  cudaFree(hostInput2);
  cudaFree(hostOutput);
  cudaFree(resultRef);
  // free(hostInput1);
  // free(hostInput2);
  // free(hostOutput);
  // free(resultRef);

  return 0;
}
