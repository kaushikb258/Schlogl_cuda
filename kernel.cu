#include "kernel.h"
#include "consts.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cstdlib>

#define TX 32
#define TY 32
#define RAD 1

int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__device__
int idxClip(int idx, int idxMax) {
  return idx >(idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int col, int row, int width, int height) {
  return idxClip(col, width) + idxClip(row, height)*width;
}


__global__
void resetKernel(float *d_u, float *d_v, int w, int h, int setup) {
  const int col = blockIdx.x*blockDim.x + threadIdx.x;
  const int row = blockIdx.y*blockDim.y + threadIdx.y;
  if ((col >= w) || (row >= h)) return;

  float uu, vv;
  int d1;
  float d2;
  const int nislands = 5;
  int xis[nislands] = {w/2, w/4, 3*w/4, w/3, 4*w/5};
  int yis[nislands] = {h/5, 3*h/4, h/2, h/6, h/3};

  const float rad1 = 4.0;
  float A0, B0, Ap, Bp, delta;
  float k1, k3;
  int k;

  if (setup == 0)
  {
   k1 = k1_1;
   k3 = k3_1;
  }
  else
  {
   k1 = k1_2;
   k3 = k3_2;
  }

  A0 = 0.0;
  B0 = km3/k3;
  delta = km3*km3 - 4.0*k1*k1*k3/k2;
  Ap = (km3 + sqrt(delta))/(2.0*k1);
  Bp = (km3 - k1*Ap)/k3;
 

  uu = A0;
  vv = B0;

  for (k=0; k<nislands; k++)
  {
   d1 = (col-xis[k])*(col-xis[k]) + (row-yis[k])*(row-yis[k]);  
   d2 = sqrtf((float)(d1));
   if(d2 <= rad1)
   {
    uu = Ap;
    vv = Bp;
   }
  }

  d_u[row*w + col] = uu;
  d_v[row*w + col] = vv;
}



__global__
void gsKernel(uchar4 *d_out, float *d_u, float *d_v, int w, int h, int setup) {
  extern __shared__ float s_in[];
  float d2udx2, d2udy2, d2vdx2, d2vdy2;   
  float Del2u, Del2v, uu, vv, dudt, dvdt;
  int ij, im1, ip1, jm1, jp1;
  int si, di;

  // global indices
  const int col = threadIdx.x + blockDim.x * blockIdx.x;
  const int row = threadIdx.y + blockDim.y * blockIdx.y;
  if ((col >= w) || (row >= h)) return;
  const int idx = flatten(col, row, w, h);
  // local width and height
  const int s_w = blockDim.x + 2 * RAD;
  const int s_h = blockDim.y + 2 * RAD;
  // local indices
  const int s_col = threadIdx.x + RAD;
  const int s_row = threadIdx.y + RAD;
  const int s_idx = flatten(s_col, s_row, s_w, s_h);
  // assign default color values for d_out (black)
  d_out[idx].x = 0;
  d_out[idx].z = 0;
  d_out[idx].y = 0;
  d_out[idx].w = 255;


  float  k1, k3;

  if (setup == 0)
  {
   k1 = k1_1;
   k3 = k3_1;
  }
  else
  {
   k1 = k1_2;
   k3 = k3_2;
  }



   // For u

  // Load regular cells
  s_in[s_idx] = d_u[idx];

  // Load ghost cells
  if (threadIdx.x < RAD) {
    si = flatten(s_col - RAD, s_row, s_w, s_h);
    di = flatten(col - RAD, row, w, h);
    s_in[si] = d_u[di]; 

    si = flatten(s_col + blockDim.x, s_row, s_w, s_h);
    di = flatten(col + blockDim.x, row, w, h);
    s_in[si] = d_u[di]; 
  }
  if (threadIdx.y < RAD) {
    si = flatten(s_col, s_row - RAD, s_w, s_h);
    di = flatten(col, row - RAD, w, h);
    s_in[si] = d_u[di]; 
    
    si = flatten(s_col, s_row + blockDim.y, s_w, s_h);
    di = flatten(col, row + blockDim.y, w, h);
    s_in[si] = d_u[di];
  }

  __syncthreads();


  // second derivative
  ij = flatten(s_col, s_row, s_w, s_h); 
  im1 = flatten(s_col - 1, s_row, s_w, s_h);
  ip1 = flatten(s_col + 1, s_row, s_w, s_h);  
  jm1 = flatten(s_col, s_row - 1, s_w, s_h);
  jp1 = flatten(s_col, s_row + 1, s_w, s_h);

  d2udx2 = ( s_in[im1] - 2.0*s_in[ij] + s_in[ip1] )/(dx*dx);
  d2udy2 = ( s_in[jm1] - 2.0*s_in[ij] + s_in[jp1] )/(dy*dy);
    
  Del2u = d2udx2 + d2udy2;
  uu = s_in[ij];


    // For v

  // Load regular cells
  s_in[s_idx] = d_v[idx];

  // Load ghost cells
  if (threadIdx.x < RAD) {
    si = flatten(s_col - RAD, s_row, s_w, s_h);
    di = flatten(col - RAD, row, w, h);
    s_in[si] = d_v[di]; 

    si = flatten(s_col + blockDim.x, s_row, s_w, s_h);
    di = flatten(col + blockDim.x, row, w, h);
    s_in[si] = d_v[di]; 
  }
  if (threadIdx.y < RAD) {
    si = flatten(s_col, s_row - RAD, s_w, s_h);
    di = flatten(col, row - RAD, w, h);
    s_in[si] = d_v[di]; 
    
    si = flatten(s_col, s_row + blockDim.y, s_w, s_h);
    di = flatten(col, row + blockDim.y, w, h);
    s_in[si] = d_v[di];
  }

  __syncthreads();


  // second derivative
  ij = flatten(s_col, s_row, s_w, s_h); 
  im1 = flatten(s_col - 1, s_row, s_w, s_h);
  ip1 = flatten(s_col + 1, s_row, s_w, s_h);  
  jm1 = flatten(s_col, s_row - 1, s_w, s_h);
  jp1 = flatten(s_col, s_row + 1, s_w, s_h);

  d2vdx2 = ( s_in[im1] - 2.0*s_in[ij] + s_in[ip1] )/(dx*dx);
  d2vdy2 = ( s_in[jm1] - 2.0*s_in[ij] + s_in[jp1] )/(dy*dy);
    
  Del2v = d2vdx2 + d2vdy2;
  vv = s_in[ij];

  
  __syncthreads();

  dudt = Da*Del2u - k1*uu + k2*uu*uu*vv;
  dvdt = Db*Del2v + km3 - k3*vv - k2*uu*uu*vv;
  
  uu += dudt*dt;
  vv += dvdt*dt;

  d_u[idx] = uu;
  d_v[idx] = vv;

  d_out[idx].x = 255.0/2.0 - 255.0/2.0*tanh((uu-1.0)/0.2);
  d_out[idx].y = 255.0/2.0*tanh((uu-1.0)/0.25) - 255.0/2.0*tanh((uu-2.0)/0.25);
  d_out[idx].z = 255.0/2.0*tanh((uu-2.0)/0.25) - 255.0/2.0*tanh((uu-3.0)/0.25);
  d_out[idx].w = 255.0/2.0 + 255.0/2.0*tanh((uu-3.0)/0.25);

  __syncthreads();
}

void kernelLauncher(uchar4 *d_out, float *d_u, float *d_v, int w, int h,
                    int setup) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize(divUp(w, TX), divUp(h, TY));
  const size_t smSz = (TX + 2 * RAD)*(TY + 2 * RAD)*sizeof(float);
  gsKernel<<<gridSize, blockSize, smSz>>>(d_out, d_u, d_v, w, h, setup);
}

void resetgs(float *d_u, float *d_v, int w, int h, int setup) {
 const dim3 blockSize(TX, TY);
 const dim3 gridSize(divUp(w, TX), divUp(h, TY));
 resetKernel<<<gridSize, blockSize>>>(d_u, d_v, w, h, setup);
}

