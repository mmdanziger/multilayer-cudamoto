#include "Cudamoto2.h"

/*Lacuna double atomicAdd*/
#ifndef  __SM_60_ATOMIC_FUNCTIONS_H__
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

/**
 * Kernels related to reduce operation
 */


__global__ void device_reduce_warp_atomic_kernel_float2(real *in, real* outx, real* outy, int N, int second_net_offset) {
	real sumx=0,sumy=0;
	int j=0;
	for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {

		j = i + second_net_offset;

		sumx+=cos(in[j]);
		sumy+=sin(in[j]);
	}
  for (int offset = warpSize/2; offset > 0; offset /= 2){
	sumx += __shfl_down_sync(0xFFFFFFFF, sumx, offset);
	sumy += __shfl_down_sync(0xFFFFFFFF, sumy, offset);
  }
  if(threadIdx.x%warpSize==0){
    atomicAdd(outx,sumx);
    atomicAdd(outy,sumy);

  }
}


__global__ void device_reduce_kfactor_warp_atomic_kernel_float2(real *in, const uint_t *   offsetList, real* outx, real* outy, int N, int second_net_offset) {
	real sumx=0,sumy=0;
	int j=0,k=0;
	for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {

		j = i + second_net_offset;
		k = offsetList[j+1] - offsetList[j];
		sumx+=k*cos(in[j]);
		sumy+=k*sin(in[j]);
	}
  for (int offset = warpSize/2; offset > 0; offset /= 2){
	sumx += __shfl_down_sync(0xFFFFFFFF, sumx, offset);
	sumy += __shfl_down_sync(0xFFFFFFFF, sumy, offset);
  }
  if(threadIdx.x%warpSize==0){
    atomicAdd(outx,sumx);
    atomicAdd(outy,sumy);

  }
}

__global__ void device_reduce_sis_kfactor_warp_atomic_kernel_float2(real *in, const uint_t *   offsetList, real* outx, int N, int second_net_offset) {
	real sumx=0;
	int j=0,k=0;
	for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
		j = i + second_net_offset;
		k = offsetList[j+1] - offsetList[j];
		sumx+=k*in[j];
	}
  for (int offset = warpSize/2; offset > 0; offset /= 2){
	sumx += __shfl_down_sync(0xFFFFFFFF, sumx, offset);


  }
  if(threadIdx.x%warpSize==0){
    atomicAdd(outx,sumx);
  }
}



__global__ void device_reduce_warp_atomic_kernel_1norm(real *in, real* out, int N) {
	real sum=0;

	for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
		sum+=abs(in[i]);

	}
  for (int offset = warpSize/2; offset > 0; offset /= 2){
	sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }
  if(threadIdx.x%warpSize==0){
    atomicAdd(out,sum);
  }
}


__global__ void r_local_euma_step( const uint_t N, const real h, const real __restrict__ lambda1, const real __restrict__ lambda2, const uint_t couple_to,
		const int interaction_type, const uint_t * __restrict__ adjList, const uint_t * __restrict__ offsetList,
		const real * __restrict__ theta, const real * __restrict__ omega, real * __restrict__ k1)
{
	const uint_t this_idx = blockIdx.x * blockDim.x + threadIdx.x;// threadIdx.x;
	if(this_idx < 2*N){
		const uint_t this_offset = offsetList[this_idx];
		const real this_k = offsetList[this_idx+1] - this_offset;
		const real this_theta = theta[this_idx], this_omega = omega[this_idx];
		real r_local=0,neighbor_phase=0,x=0,y=0,tmp_phase,new_k1=0,cos_factor=0;

		const uint_t net_idx = this_idx < N ? 0 : 1;

			//Calculate standard kuramoto term
			for(uint_t j=0; j<this_k; j++){
				new_k1 += sinf(theta[adjList[this_offset + j]] - this_theta);
			}
			if(this_idx%N < couple_to){
				const uint_t other_idx = net_idx == 0? (this_idx + N) : this_idx -N;
				const uint_t other_offset = offsetList[other_idx];
				const uint_t other_k = offsetList[other_idx + 1] - offsetList[other_idx];
				/* TO CHECK r_i = avg( exp(i*theta_j - psi_i)) instead of avg(exp(i*theta_j)) with autoaveraging //
								//Get arg of average phase difference
								for(uint_t j=0; j<other_k; j++){
									tmp_phase = theta[adjList[other_offset + j]] - neighbor_phase;
									x += cosf(tmp_phase);
									y += sinf(tmp_phase);
								}
				 */
				//Get arg of average phase difference
				for(uint_t j=0; j<other_k; j++){
					tmp_phase = theta[adjList[other_offset + j]];
					x += cosf(tmp_phase);
					y += sinf(tmp_phase);
				}

				r_local = other_k>0? sqrt(x*x + y*y) / other_k : 0;
#ifdef COS_ONLY
				neighbor_phase = other_k>0? atan2f(y,x) : 0;
				cos_factor = abs(cosf(this_theta - neighbor_phase));
				r_local = cos_factor;
#elif defined(SQRT_RCOS)
				neighbor_phase = other_k>0? atan2f(y,x) : 0;
				cos_factor = abs(cosf(this_theta - neighbor_phase));
				r_local = sqrt(r_local*cos_factor);
#elif defined(RCOS)
				neighbor_phase = other_k>0? atan2f(y,x) : 0;
				cos_factor = abs(cosf(this_theta - neighbor_phase));
				r_local *= cos_factor;
#endif
				if(interaction_type == 1)
					new_k1 *= r_local;
				else if(interaction_type == 2)
					new_k1 *= (1 - r_local);
				else if(interaction_type == 3){
					if(net_idx == 0)
						new_k1 *= r_local;
					else
						new_k1 *= (1 - r_local);
				}

		}
		real lambda = net_idx==0? lambda1 : lambda2;
		new_k1 *= lambda;
		new_k1 += this_omega;
		new_k1 *= h;
		k1[this_idx] = new_k1;
	}
}

__global__ void mixed_system_euma_step( const uint_t N, const real h, const real __restrict__ lambda1, const real __restrict__ lambda2, const uint_t couple_to,
		const int interaction_type, const uint_t * __restrict__ adjList, const uint_t * __restrict__ offsetList,
		const real * __restrict__ theta, const real * __restrict__ omega, real * __restrict__ k1)
{
	const uint_t this_idx = blockIdx.x * blockDim.x + threadIdx.x;// threadIdx.x;
	if(this_idx < 2*N){
		const uint_t this_offset = offsetList[this_idx];
		const real this_k = offsetList[this_idx+1] - this_offset;
		const real this_theta = theta[this_idx], this_omega = omega[this_idx];
		real r_local=0,neighbor_phase=0,x=0,y=0,tmp_phase=0,new_k1=0;

		const uint_t net_idx = this_idx < N ? 0 : 1;
		if(net_idx == 0){ //calculate standard kuramoto modulated by disease level in other net
			//Calculate standard kuramoto term
			for(uint_t j=0; j<this_k; j++){
				new_k1 += sinf(theta[adjList[this_offset + j]] - this_theta);
			}
			if(this_idx%N < couple_to){
				const uint_t other_idx = net_idx == 0? (this_idx + N) : this_idx -N;
				const uint_t other_offset = offsetList[other_idx];
				const uint_t other_k = offsetList[other_idx + 1] - offsetList[other_idx];

				//Get average state of neighbors in other net
				for(uint_t j=0; j<other_k; j++){
					tmp_phase+=theta[adjList[other_offset + j]];
				}
				r_local = other_k>0? tmp_phase / other_k : 0;
				if(interaction_type == 1)
					new_k1 *= r_local;
				else if(interaction_type == 2)
					new_k1 *= (1 - r_local);
				else if(interaction_type == 3){
					if(net_idx == 0)
						new_k1 *= r_local;
					else
						new_k1 *= (1 - r_local);
				}

			}
		} else {

			//Calculate standard SIS term
			for(uint_t j=0; j<this_k; j++){
							new_k1 += theta[adjList[this_offset + j]];
						}
			new_k1*=(1 - this_theta);
			if(this_idx%N < couple_to){
				const uint_t other_idx = net_idx == 0? (this_idx + N) : this_idx -N;
				const uint_t other_offset = offsetList[other_idx];
				const uint_t other_k = offsetList[other_idx + 1] - offsetList[other_idx];
#ifdef AVERAGE_NEIGHBORS
				//Get neighbor's average phase
				for(uint_t j=0; j<other_k; j++){
					neighbor_phase +=  theta[adjList[other_offset + j]];
					//neighbor_phase += tmp_phase;
				}
				//Averaged
				neighbor_phase=other_k>0? neighbor_phase/other_k : neighbor_phase;
#endif
				//Get arg of average phase difference
				for(uint_t j=0; j<other_k; j++){
					tmp_phase = theta[adjList[other_offset + j]] - neighbor_phase;
					x += cosf(tmp_phase);
					y += sinf(tmp_phase);
				}
				r_local = other_k>0? sqrt(x*x + y*y) / other_k : 0;
				if(interaction_type == 1)
					new_k1 *= r_local;
				else if(interaction_type == 2)
					new_k1 *= (1 - r_local);
				else if(interaction_type == 3){
					if(net_idx == 0)
						new_k1 *= r_local;
					else
						new_k1 *= (1 - r_local);
				}

			}
		}

		real lambda = net_idx==0? lambda1 : lambda2;
		new_k1 *= lambda;
		if (net_idx ==0){
			new_k1 += this_omega;
		}else{
			new_k1 -= this_theta; //SIS has a -x self-term
		}
		new_k1 *= h;
		k1[this_idx] = new_k1;
	}
}


__global__ void r_local_euma_step_weighted_links( const uint_t N, const real h, const real __restrict__ lambda1, const real __restrict__ lambda2,
		const uint_t couple_to, const int interaction_type, const real * __restrict__ weightList, const uint_t * __restrict__ adjList, const uint_t * __restrict__ offsetList,
		const real * __restrict__ theta, const real * __restrict__ omega, real * __restrict__ k1)
{
	const uint_t this_idx = blockIdx.x * blockDim.x + threadIdx.x;// threadIdx.x;
	if(this_idx < 2*N){
		const uint_t this_offset = offsetList[this_idx];
		const real this_k = offsetList[this_idx+1] - this_offset;
		const real this_theta = theta[this_idx], this_omega = omega[this_idx];
		real r_local=0,neighbor_phase=0,x=0,y=0,tmp_phase,new_k1=0;

		const uint_t net_idx = this_idx < N ? 0 : 1;

			//Calculate standard kuramoto term
			for(uint_t j=0; j<this_k; j++){
				new_k1 += weightList[this_offset + j] * sinf(theta[adjList[this_offset + j]] - this_theta);
			}
			if(this_idx%N < couple_to){
				const uint_t other_idx = net_idx == 0? (this_idx + N) : this_idx -N;
				const uint_t other_offset = offsetList[other_idx];
				const uint_t other_k = offsetList[other_idx + 1] - offsetList[other_idx];

				//Get neighbor's average phase
				for(uint_t j=0; j<other_k; j++){
					neighbor_phase +=  theta[adjList[other_offset + j]];
					//neighbor_phase += tmp_phase;
				}
				//Averaged
				neighbor_phase=other_k>0? neighbor_phase/other_k : neighbor_phase;
				//Get arg of average phase difference
				for(uint_t j=0; j<other_k; j++){
					tmp_phase = theta[adjList[other_offset + j]] - neighbor_phase;
					x += cosf(tmp_phase);
					y += sinf(tmp_phase);
				}
				r_local = other_k>0? sqrt(x*x + y*y) / other_k : 0;
				if(interaction_type == 1)
					new_k1 *= r_local;
				else if(interaction_type == 2)
					new_k1 *= (1 - r_local);
				else if(interaction_type == 3){
					if(net_idx == 0)
						new_k1 *= r_local;
					else
						new_k1 *= (1 - r_local);
				}

		}
		real lambda = net_idx==0? lambda1 : lambda2;
		new_k1 *= lambda;
		new_k1 += this_omega;
		new_k1 *= h;
		k1[this_idx] = new_k1;
	}
}

__global__ void r_local_euma_step_directed_links( const uint_t N, const real h, const real __restrict__ lambda1, const real __restrict__ lambda2,
		const short * __restrict__ interaction_type_vector, const uint_t * __restrict__ adjList, const uint_t * __restrict__ offsetList,
		const real * __restrict__ theta, const real * __restrict__ omega, real * __restrict__ k1)
{
	const uint_t this_idx = blockIdx.x * blockDim.x + threadIdx.x;// threadIdx.x;
	const uint_t this_offset = offsetList[this_idx];
	const real this_k = offsetList[this_idx+1] - this_offset;
	const real this_theta = theta[this_idx], this_omega = omega[this_idx];
	real r_local=0,neighbor_phase=0,x=0,y=0,tmp_phase,new_k1=0;
	const int interaction_type = static_cast<int>(interaction_type_vector[this_idx]);
	const uint_t net_idx = this_idx < N ? 0 : 1;

		//Calculate standard kuramoto term
		for(uint_t j=0; j<this_k; j++){
			new_k1 += sinf(theta[adjList[this_offset + j]] - this_theta);
		}
		if(interaction_type){//if there is any interaction term to this node
			const uint_t other_idx = net_idx == 0? (this_idx + N) : this_idx -N;
			const uint_t other_offset = offsetList[other_idx];
			const uint_t other_k = offsetList[other_idx + 1] - offsetList[other_idx];

			//Get neighbor's average phase
			for(uint_t j=0; j<other_k; j++){
				neighbor_phase +=  theta[adjList[other_offset + j]];
				//neighbor_phase += tmp_phase;
			}
			//Averaged
			neighbor_phase=other_k>0? neighbor_phase/other_k : neighbor_phase;
			//Get arg of average phase difference
			for(uint_t j=0; j<other_k; j++){
				tmp_phase = theta[adjList[other_offset + j]] - neighbor_phase;
				x += cosf(tmp_phase);
				y += sinf(tmp_phase);
			}
			r_local = other_k>0? sqrt(x*x + y*y) / other_k : 0;
			if(interaction_type == 1){
				new_k1 *= r_local;
			}else if(interaction_type == 2){
				new_k1 *= (1 - r_local);
			}


	}
	real lambda = net_idx==0? lambda1 : lambda2;
	new_k1 *= lambda;
	new_k1 += this_omega;
	new_k1 *= h;
	k1[this_idx] = new_k1;

}

__global__ void vectorPlusEqualsMod2Pi(real * __restrict__ y, const real * __restrict__ x){
    uint_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    y[idx] = fmod(y[idx] + x[idx],static_cast<real>(TWO_PI_DEF));
}

__global__ void vectorPlusEqualsMod2PiFactor(real * __restrict__ y, const real * __restrict__ x, const real factor){
    uint_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    y[idx] = fmod(y[idx] + factor*x[idx],static_cast<real>(TWO_PI_DEF));
}

__global__ void vectorSumMod2PiFactor(real * __restrict__ y, const real * __restrict__ x, const real * __restrict__ d, const real factor){
    uint_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    y[idx] = fmod(x[idx] + factor*d[idx],static_cast<real>(TWO_PI_DEF));
}




__global__ void vectorDifference(real * __restrict__ d, const real * __restrict__ y, const real * __restrict__ x, const int N){
    uint_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx <N){
    real diff = y[idx] - x[idx];
/**
 * Periodicity of two pi means differences cannot be longer than pi.  If so, go the other way.
 */
    if(diff > PI_DEF){
    	diff-=TWO_PI_DEF;
    }else if (diff < -PI_DEF){
    	diff+=TWO_PI_DEF;
    }

    d[idx] = diff;
    }
}

