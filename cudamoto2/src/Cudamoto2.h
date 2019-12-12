/*
 * Cudamoto2.h
 *
 *  Created on: Dec 30, 2015
 *      Author: micha
 */

#ifndef CUDAMOTO2_H_
#define CUDAMOTO2_H_

#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
//#include <thrust/reduce.h>
#include <cstdint>
#include <fstream>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <chrono>
#include <exception>
#include <iomanip>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include "helper_string.h"
#include "helper_cuda.h"
//#include <helper_functions.h>
#include <curand_kernel.h>
#include <curand.h>

using real = float;
using real2 = float2;
using real3 = float3;
using int_t = int32_t;
using uint_t = uint32_t;
/**/
#include "utilities.hpp"
using uvector = std::vector<uint_t>;
using uvectorvec = std::vector< std::vector<uint_t> >;
using rvector = std::vector<real>;
using std::vector;

//using thrust::device_vector;
//using thrust::host_vector;

#define PI_DEF 3.14159265359
#define TWO_PI_DEF 6.283185307179586
//#define INTERDEPENDENT 1
//#define COMPETITIVE 2

struct criticalValues{
	real lambda_f;
	real lambda_b;
	real deltar_f;
	real deltar_b;
};

struct cudamoto2Result{
	uint_t N;
	real k1;
	real k2;
	real lambda1;
	real lambda2;
	uint_t interaction_type;
	real r1_init,r2_init;
	real r1_final,r2_final;
	real f,h,tfinal;
	uint_t random_id;
};

struct initialConditionResult{
	real lambda1,lambda2;
	real2 r1r2_00,r1r2_10,r1r2_11;

};
template<typename T> T& operator<<(T& stream, initialConditionResult& res){
		stream << "{\"lambda\" : ["<<res.lambda1 <<", "<<res.lambda2 << "], "
				"\"r_00\" : ["<<res.r1r2_00.x <<","<<res.r1r2_00.y<<"], "
				"\"r_10\" : ["<<res.r1r2_10.x <<","<<res.r1r2_10.y<<"], "
				"\"r_11\" : ["<<res.r1r2_11.x <<","<<res.r1r2_11.y<<"] }\n";
	return stream;
	}

class Cudamoto2 {
public:

public:
	Cudamoto2(uint_t N, real k1, real k2, real f, real h);
	Cudamoto2(std::string edge_list_fname);
	virtual ~Cudamoto2();
	static constexpr int INTERDEPENDENT = 1;
	static constexpr int COMPETITIVE = 2;
	static constexpr int HYBRID = 3;
	static constexpr int MIXED = 4;
	static constexpr int UNIFORM = 1;
	static constexpr int CAUCHY = 2;
	static constexpr int GAUSSIAN = 3;
	void generate_random_networks();
	void generate_exponential_length_networks();
	void generate_scale_free_networks();
	void initialize_oscillator_frequencies_and_phases();
	void randomize_oscillator_phases(int netidx);
	void zero_oscillator_phases(int netidx);
	void set_initial_conditions(int2 r1r2_init);
	void set_initial_conditions(real2 r1r2_init);
	void load_weights(std::string edge_list_fname);
	void allocate_and_copy_to_device();
	void generate_omega();
	template  <typename T> void cuda_free_if_alloced(T var);
	void copy_omega_to_device();
	void copy_network_to_device();
	void integrate_one_step();
	std::vector<real2> scan_lambda_values(real fixed_net_lambda, int fixed_net_idx, real lambda_init, real lambda_final, real lambda_step, real t_final);
	std::vector<real3> scan_lambda_values_conserved_sum(real lambda_sum, real lambda_init, real lambda_final, real lambda_step, real t_final);
	criticalValues find_lambda_fb(real fixed_net_lambda, int fixed_net_idx, real precision, real t_final);
	void scan_both_lambda_values();
	vector< vector<real2> > scan_both_lambda_values_symmetric(real lambda_init, real lambda_final, real lambda_step, real t_final);
	void set_lambda(real lambda1, real lambda2);
	void set_zeta(real zeta1, real zeta2){zeta={zeta1,zeta2};}
	void set_sfgamma(real sfgamma1, real sfgamma2){sfgamma={sfgamma1,sfgamma2};}
	void set_f(real f_){ f = f_; if(isdirected){assign_directed_interactions();}}
	void generate_initial_conditions();
	real2 integrate_to_time(real t_final);
	void local_attack(int rh, int net_idx, int target_r);
	vector<real2> generate_history_to_time(real t_final);
	vector<real2> generate_history_to_time(real t_final, real stop_condition);
	real2 generate_result(int2 r1r2_init, real t_final);
	vector<real2> generate_history_changing_lambda(real lambda_init, real lambda_final, real lambda_step, real t_final);
	vector<real2> make_L_lambda_path_vector(real2 lambda_0, real2 lambda_1, real2 lambda_2, int total_steps);
	vector<real2> make_straight_lambda_path_vector(real2 lambda_0, real2 lambda_1, int total_steps);
	/**
	 * Lyapunov measurement functions
	 */
	void prepare_lyapunov_measurements();
	void switch_active_theta(int zero_if_primary);
	real get_primary_secondary_difference();
	void renormalize_secondary(real dist, real d0);
	void lyapunov_scratchpad();

	void make_interdependent(){interaction_type = INTERDEPENDENT;}
	void make_competitive(){interaction_type = COMPETITIVE;}
	void make_hybrid(){interaction_type = HYBRID;}
	void make_uniform(){natural_distribution = UNIFORM;}
	void make_cauchy(){natural_distribution = CAUCHY;}
	void make_directed(){isdirected =1;assign_directed_interactions();}
	void make_mixed(){ismixed=1;}
	void assign_directed_interactions();
	void set_directed(int directed_state) {isdirected = directed_state;}
	void set_distribution(int type_id){natural_distribution = type_id;}
	int get_distribution(){return natural_distribution;}
	void set_cauchy_tail_parameter(real parameter) {frequency_distribution_parameter = parameter;}
	void set_interaction(int type_id) {interaction_type = type_id;}
	void set_use_global_op(int yesno){isglobalOP = yesno;}
	int get_interaction(){return interaction_type;}
	real2 get_r_device();
	real2 get_r_host();
	uint_t get_N(){return N;}
	const rvector& get_theta(){ return theta;}
	void copy_theta_to_host();
	void copy_theta_to_device();

	template <typename Stream> void output_theta(Stream& os);
private:
	uint_t N;
	uint_t threads;
	rvector kbar; //vec of len 2 for each net
	rvector lambda; //vec of len 2 for each net
	rvector zeta; //vec of len 2 for each net
	rvector sfgamma;
	uvector flatAdjList;
	uvector flatOffsetList;
	rvector flatWeightList;
	rvector theta;
	rvector omega;
	std::vector<short> interaction_type_vector;
	std::mt19937 gen;
	real f;
	real h;
	uint_t *d_flatAdjList = nullptr;
	uint_t *d_flatOffsetList = nullptr;
	real *d_theta = nullptr;
	real *d_omega = nullptr;
	real *d_k1 = nullptr;
	uint_t *d_offsetList = nullptr;
	uint_t *d_adjList = nullptr;
	real *d_weightList = nullptr;
	real *d_x = nullptr, *d_y = nullptr, *h_x = nullptr, *h_y = nullptr; // for calculating r = sqrt(x*x + y*y) / N
	real *d_theta_init_upup = nullptr, *d_theta_init_updown = nullptr, *d_theta_init_downup = nullptr, *d_theta_init_downdown = nullptr;
	short *d_interaction_type_vector = nullptr;
	dim3 doubleNetNodeBlock;
	dim3 threadsPerBlock;
	int interaction_type,natural_distribution,isdirected,isweighted,ismixed,isglobalOP;
	real frequency_distribution_parameter;
	real *d_theta_primary = nullptr, *d_theta_secondary = nullptr, *d_secondary_diff = nullptr, *d_secondary_dist = nullptr;
};

template <typename T>
void Cudamoto2::cuda_free_if_alloced(T var){
	if(var != nullptr){
		checkCudaErrors(cudaFree(var));
		var = nullptr;
	}

}


template <typename Stream>
void Cudamoto2::output_theta(Stream & os){
	bool firstline=true;
	os << "[";
	for(auto theta_val : theta){
		if(firstline){
			os << theta_val << std::endl;
			firstline=false;
		} else {
			os << "," << theta_val << std::endl;
		}

		}
	os << "]\n";


}

#endif /* CUDAMOTO2_H_ */
