/*
 * Cudamoto2.cpp
 *
 *  Created on: Dec 30, 2015
 *      Author: micha
 */

#include "Cudamoto2.h"
#include "cudamoto2_kernels.cuh"
#include "adjlistgen.hpp"

constexpr int verbosity = 0;

Cudamoto2::Cudamoto2(uint_t N, real kbar1, real kbar2, real f, real h) :N(N),kbar(2),lambda(2),zeta(2),theta(2*N),omega(2*N),f(f),h(h)
{
	isdirected = 0;
	isweighted = 0;
	ismixed = 0;
	isglobalOP=0;
	kbar = {kbar1, kbar2};
	frequency_distribution_parameter = 1;
	natural_distribution = UNIFORM;
	zeta = {2, 2};
	gen.seed(time(0));
	make_interdependent();
	lambda = {0, 0};
	generate_random_networks();
	//generate_double_random(N,kbar1,kbar2,flatAdjList,flatOffsetList);
	//generate_double_exponential(N,kbar1,kbar2,zeta[0],zeta[1],flatAdjList,flatOffsetList);
	//generate_exponential_length_networks();
	initialize_oscillator_frequencies_and_phases();
	allocate_and_copy_to_device();
	threads = N > 1<<9 ? 1<<9 : N;
	doubleNetNodeBlock = dim3(2* N / threads);
	threadsPerBlock = dim3(threads);
	checkCudaErrors(cudaMalloc((void ** )&d_x, sizeof(real)));
	checkCudaErrors(cudaMalloc((void ** )&d_y, sizeof(real)));
	h_x = (real*) malloc(sizeof(real));
	h_y = (real*) malloc(sizeof(real));
	if(verbosity)
		std::cout << "Initialization complete.\n";
}

Cudamoto2::Cudamoto2(std::string edge_list_fname){

	isdirected = 0;
	isweighted = 1;
	frequency_distribution_parameter = 1;
	natural_distribution = UNIFORM;
	zeta = {2, 2};
	gen.seed(time(0));
	make_interdependent();
	lambda = {0, 0};
	load_weights(edge_list_fname);
	initialize_oscillator_frequencies_and_phases();
	allocate_and_copy_to_device();
	checkCudaErrors(cudaMalloc((void ** )&d_x, sizeof(real)));
	checkCudaErrors(cudaMalloc((void ** )&d_y, sizeof(real)));
	h_x = (real*) malloc(sizeof(real));
	h_y = (real*) malloc(sizeof(real));
	if(verbosity)
		std::cout << "Initialization complete.\n";
}


void Cudamoto2::generate_random_networks(){

//for(int net_idx=0; net_idx<2; ++net_idx){
//	uvectorvec adjacency_list(N);
//	uint_t s, t, num_links = 0, link_count = 0;
//	while (num_links < N * kbar[net_idx] / 2) {
//		do {
//			s = randint(N);
//			t = randint(N);
//		} while (s == t);
//		if (std::find(adjacency_list[s].begin(), adjacency_list[s].end(), t)
//				== adjacency_list[s].end()) {
//			adjacency_list[s].push_back(t);
//			adjacency_list[t].push_back(s);
//			num_links++;
//		}
//	}
//
//uvector ordering(N);
//for (uint_t i = 0; i < N; i++) {
//	ordering[i] = i;
//}
//uvector inverse_ordering(N);
//for (uint_t i = 0; i < N; i++) {
//	inverse_ordering[ordering[i]] = i;
//}
////std::cout << "Added " <<num_links << " links\n";
//uint lastNumLinks=0, offsetCorrection=0;
//if (net_idx == 0){
//	flatAdjList.resize( num_links * 2);
//	flatOffsetList.resize( N + 1);
//} else {
//	lastNumLinks = flatAdjList.size();
//	offsetCorrection = N;
//	flatAdjList.resize(lastNumLinks + num_links * 2);
//	flatOffsetList.resize(2*N + 1);
//}
//uint_t block_number;
//for (uint_t i = 0; i < N; i++) {
//	block_number = inverse_ordering[i];
//	flatOffsetList[offsetCorrection + i] = lastNumLinks + link_count;
////	std::cout << "Offset["<<i<<"] = "<<link_count<<"\n";
//	for (uint_t j = 0; j < adjacency_list[block_number].size(); j++) {
//		flatAdjList[link_count + lastNumLinks] = ordering[adjacency_list[block_number][j]] + offsetCorrection;
//		link_count++;
//
//	}
//}
//flatOffsetList[N + offsetCorrection] = link_count + lastNumLinks;
//
//}

	generate_double_random(N,kbar[0],kbar[1],flatAdjList,flatOffsetList);
	copy_network_to_device();

}

void Cudamoto2::generate_exponential_length_networks(){

	generate_double_exponential(N,kbar[0],kbar[1],zeta[0],zeta[1],flatAdjList,flatOffsetList);
	copy_network_to_device();

}

void Cudamoto2::generate_scale_free_networks(){

	generate_double_scale_free(N,sfgamma[0],sfgamma[1],flatAdjList,flatOffsetList);
	copy_network_to_device();
}

void Cudamoto2::load_weights(std::string edge_list_fname){

	WeightedListGen wl(edge_list_fname);
	wl.load_weighted_adjacency_list();
	wl.copy_weighted_adjacency_list(flatAdjList,flatOffsetList,flatWeightList);
	N = (flatOffsetList.size() - 1 ) /2;
	if(N < 1<<9){
		threads = 1<<static_cast<int>(std::log2(static_cast<real>(N)));
	} else {
		threads = 1<<9;
	}
	isweighted=1;
	theta.resize(2*N);
	omega.resize(2*N);
	doubleNetNodeBlock = dim3((2* N + threads - 1)/ threads); //<-- have to round up by one
	threadsPerBlock = dim3(threads);
	generate_omega();
	allocate_and_copy_to_device();
	copy_network_to_device();
}

void Cudamoto2::initialize_oscillator_frequencies_and_phases(){
	std::uniform_real_distribution<real> uniform_dist(-frequency_distribution_parameter,frequency_distribution_parameter);
	std::cauchy_distribution<real> cauchy_dist(0,frequency_distribution_parameter);

	for (uint_t i = 0; i < 2*N; i++) {
		omega[i] = frequency_distribution_parameter * (2 * randreal() - 1);
		if(ismixed && i >= N){
			theta[i] = 0.001*randreal(); //corresponding state of randomized for SIS is zero
		} else {
		theta[i] = PI_DEF * (2 * randreal() - 1);
		}
	}

}

void Cudamoto2::generate_omega(){
	std::uniform_real_distribution<real> uniform_dist(-frequency_distribution_parameter,frequency_distribution_parameter);
	std::cauchy_distribution<real> cauchy_dist(0,frequency_distribution_parameter);
	std::normal_distribution<real> gaussian_dist(0,frequency_distribution_parameter);

	switch(natural_distribution){
		case UNIFORM:
			for (uint_t i = 0; i < 2*N; i++) {
				omega[i] = uniform_dist(gen);
			}
			break;
		case CAUCHY:
			for (uint_t i = 0; i < 2*N; i++) {
				omega[i] = cauchy_dist(gen);
			}
			break;
		case GAUSSIAN:
			for (uint_t i = 0; i < 2*N; i++) {
				omega[i] = gaussian_dist(gen);
			}
			break;
		default:
			std::cerr << "Bad natural frequency distribution. \n";
			break;
	}
}

void Cudamoto2::copy_omega_to_device(){

	checkCudaErrors(
			cudaMemcpy(d_omega, &omega[0], sizeof(real) * 2 * N, cudaMemcpyHostToDevice));
}

void Cudamoto2::randomize_oscillator_phases(int netidx){
	for (uint_t i = netidx*N; i  < (netidx+1)*N; i++) {
		if(ismixed && i >= N){
			theta[i] = 0.001*randreal(); //corresponding state of randomized for SIS is zero
		} else {
			theta[i] = PI_DEF * (2 * randreal() - 1);
		}
	}
	checkCudaErrors(
				cudaMemcpy(d_theta + netidx*N, &theta[netidx*N], sizeof(real) * N, cudaMemcpyHostToDevice));
}

void Cudamoto2::zero_oscillator_phases(int netidx){
	if(ismixed && netidx==1){
		checkCudaErrors(cudaMemsetAsync(d_theta + N, 0.999999, N*sizeof(real)));
		getLastCudaError("Failed to zero phases on device.");
	} else {
		checkCudaErrors(cudaMemsetAsync(d_theta +netidx*N, 0, N*sizeof(real)));
		getLastCudaError("Failed to zero phases on device.");
	}
}


void Cudamoto2::set_initial_conditions(int2 r1r2_init){
	switch(r1r2_init.x){
		case 0:
			randomize_oscillator_phases(0);
			break;
		case 1:
			zero_oscillator_phases(0);
			break;
		default:
			std::cerr<<"Initial condition not supported.\n";
			break;
	}
	switch(r1r2_init.y){
			case 0:
				randomize_oscillator_phases(1);
				break;
			case 1:
				zero_oscillator_phases(1);
				break;
			default:
				std::cerr<<"Initial condition not supported.\n";
				break;
		}
}

void Cudamoto2::set_initial_conditions(real2 r1r2_init){
	std::uniform_real_distribution<real> rand_phase(-TWO_PI_DEF,TWO_PI_DEF);
	std::uniform_real_distribution<real> random_float(0,1);

	for(int i=0; i<N; ++i){
		if(random_float(gen) < r1r2_init.x)
			theta[i] = 0;
		else
			theta[i] = rand_phase(gen);
	}

	for(int i=N; i<2*N; ++i){
		if(random_float(gen) < r1r2_init.y)
			theta[i] = ismixed? 0.99999 : 0;
		else
			theta[i] = ismixed? 0.00001 : rand_phase(gen);
	}
	checkCudaErrors(
			cudaMemcpy(d_theta, &theta[0], sizeof(real) * 2 * N, cudaMemcpyHostToDevice));
	if(verbosity){
		std::cout << "Initial conditions set (float)\n";
	}
}

void Cudamoto2::allocate_and_copy_to_device(){

	/*if(d_offsetList != nullptr)
		checkCudaErrors(cudaFree(d_offsetList));
	if(d_adjList != nullptr)
		checkCudaErrors(cudaFree(d_adjList));
	*/
	//cuda_free_if_alloced(d_offsetList);
	//cuda_free_if_alloced(d_adjList);
	//cuda_free_if_alloced(d_theta);
	//cuda_free_if_alloced(d_omega);
	//cuda_free_if_alloced(d_k1);

	if(d_theta == nullptr)
		checkCudaErrors(cudaMalloc((void ** )&d_theta, sizeof(real) * 2 * N));
	if(d_omega == nullptr)
		checkCudaErrors(cudaMalloc((void ** )&d_omega, sizeof(real) * 2 * N));
	if(d_k1 == nullptr)
		checkCudaErrors(cudaMalloc((void ** )&d_k1, sizeof(real) * 2 * N));
//	checkCudaErrors(cudaMalloc((void ** )&d_offsetList, sizeof(uint_t) * (2 * N + 1)));
//	checkCudaErrors(cudaMalloc((void ** )&d_adjList, sizeof(uint_t) * flatAdjList.size()));
//	checkCudaErrors(
//			cudaMemcpy(d_offsetList, &flatOffsetList[0], sizeof(uint_t) * (2 * N + 1),
//					cudaMemcpyHostToDevice));
//	checkCudaErrors(
//			cudaMemcpy(d_adjList, &flatAdjList[0], sizeof(uint_t) * (flatAdjList.size()),
//					cudaMemcpyHostToDevice));
	//copy_network_to_device();
	//std::cout << "cudaMalloc finished.\n";
	//checkCudaErrors(cudaMalloc((void **)&d_flatEdgeList, sizeof(uint_t)*numlinks));
	checkCudaErrors(
			cudaMemcpy(d_theta, &theta[0], sizeof(real) * 2 * N, cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(d_omega, &omega[0], sizeof(real) * 2 * N, cudaMemcpyHostToDevice));
	checkCudaErrors(
		    cudaMemsetAsync(d_k1, 0, 2*N*sizeof(real)));
	//checkCudaErrors(cudaMemcpy(d_flatEdgeList,flatEdgeList, sizeof(uint_t)*(numlinks),cudaMemcpyHostToDevice));
	//std::cout << "cudaMemcpy finished\n";
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("Failed to initialize data on device.");
}

void Cudamoto2::set_lambda(real lambda1, real lambda2){
	lambda[0] = lambda1;
	lambda[1] = lambda2;
}

void Cudamoto2::assign_directed_interactions(){
	interaction_type_vector.resize(2*N);
	std::uniform_real_distribution<real> uniform_real(0,1);
	switch(interaction_type){
	case INTERDEPENDENT:
	case COMPETITIVE:	{
		short short_interaction_type = static_cast<short>(interaction_type);
		for(uint_t i = 0; i < 2*N; ++i){
				interaction_type_vector[i] = uniform_real(gen) < f? short_interaction_type : 0;
			}
		break;}
	case HYBRID:
	{
		short short_interaction_type = static_cast<short>(INTERDEPENDENT);
		for(uint_t i = 0; i < N; ++i){
			interaction_type_vector[i] = uniform_real(gen) < f? short_interaction_type : 0;
		}
		short_interaction_type = static_cast<short>(COMPETITIVE);
		for(uint_t i = 0; i < N; ++i){
			interaction_type_vector[i] = uniform_real(gen) < f? short_interaction_type : 0;
		}
		break;
	}
	case MIXED:{
		short interdep_type = static_cast<short>(INTERDEPENDENT);
		short comp_type = static_cast<short>(COMPETITIVE);
		for(uint_t i = 0; i < 2*N; ++i){
			interaction_type_vector[i] = uniform_real(gen) < f? interdep_type: comp_type;
		}
		break;

	}
	default:
		std::fill(interaction_type_vector.begin(), interaction_type_vector.end(), 0);
		break;


	}

	if(d_interaction_type_vector == nullptr){
		checkCudaErrors(cudaMalloc((void ** )&d_interaction_type_vector, sizeof(short) * 2 * N));
	}
	checkCudaErrors(
			cudaMemcpy(d_interaction_type_vector, &interaction_type_vector[0], sizeof(short) * 2 * N, cudaMemcpyHostToDevice));

}

void Cudamoto2::integrate_one_step(){
if(isweighted){
	r_local_euma_step_weighted_links<<<doubleNetNodeBlock,threadsPerBlock>>>(N,h,lambda[0],lambda[1],f*N,interaction_type,d_weightList,d_adjList,d_offsetList,d_theta,d_omega,d_k1);
} else if(isdirected){
	r_local_euma_step_directed_links<<<doubleNetNodeBlock,threadsPerBlock>>>(N,h,lambda[0],lambda[1],d_interaction_type_vector,d_adjList,d_offsetList,d_theta,d_omega,d_k1);
} else if(ismixed){
	mixed_system_euma_step<<<doubleNetNodeBlock,threadsPerBlock>>>(N,h,lambda[0],lambda[1],f*N,interaction_type,d_adjList,d_offsetList,d_theta,d_omega,d_k1);
} else {
	r_local_euma_step<<<doubleNetNodeBlock,threadsPerBlock>>>(N,h,lambda[0],lambda[1],f*N,interaction_type,d_adjList,d_offsetList,d_theta,d_omega,d_k1);
}
	getLastCudaError("euma_step execution failed");
	vectorPlusEqualsMod2Pi<<<doubleNetNodeBlock,threadsPerBlock>>>(d_theta,d_k1);
	getLastCudaError("vector plus equals execution failed");
//	std::cout << "one step\n";
}

real2 Cudamoto2::integrate_to_time(real t_final){
	real r1=0,r2=0,t=0;
	uint_t ignore_first = 200;
	vector<real2> rhist_this_run;
	uint_t step_count=0;
	while(t<t_final){
		integrate_one_step();
		t+=h;
		step_count++;
		if(step_count > ignore_first)
			rhist_this_run.push_back(get_r_device());
	}
	for(auto r1r2_it = rhist_this_run.begin(); r1r2_it != rhist_this_run.end(); r1r2_it++){
		r1 += r1r2_it->x;
		r2 += r1r2_it->y;
	}
	r1 /= rhist_this_run.size();
	r2 /= rhist_this_run.size();
	return makereal2(r1,r2);
}

vector<real2> Cudamoto2::generate_history_to_time(real t_final){
	vector<real2> rhist_this_run;
	real t=0;
	while(t<t_final){
		integrate_one_step();
		t+=h;
		rhist_this_run.push_back(get_r_device());
	}
	return rhist_this_run;

}

vector<real2> Cudamoto2::generate_history_to_time(real t_final, real stop_condition){
	vector<real2> rhist_this_run;
	int averaging_window = 500;
	real t=0;
	real oldxsum=0,oldysum=0,newxsum=0,newysum=0;
	while(t<t_final){
		integrate_one_step();
		t+=h;
		auto thisr = get_r_device();
		rhist_this_run.push_back(thisr);
		if(thisr.x < 0.03 && thisr.y < 0.03)
			break;

		if(stop_condition < 0)
			continue;
		if(rhist_this_run.size() > 2*averaging_window){
			oldxsum -= (rhist_this_run.rbegin() + 2*averaging_window )->x;
			oldysum -= (rhist_this_run.rbegin() + 2*averaging_window )->y;
			real2 tomove = *(rhist_this_run.rbegin() + averaging_window);
			oldxsum += tomove.x;
			oldysum += tomove.y;
			newxsum -= tomove.x;
			newysum -= tomove.y;
			newxsum += thisr.x;
			newysum += thisr.y;
			if(abs(newysum - oldysum)/averaging_window < stop_condition && (abs(newxsum - oldxsum)/averaging_window < stop_condition)){
				break;
			}
		}
		else if( rhist_this_run.size() < averaging_window){
			oldxsum += thisr.x;
			oldysum += thisr.y;
		}
		else if( rhist_this_run.size() < 2*averaging_window){
			newxsum += thisr.x;
			newysum += thisr.y;
		}


	}
	return rhist_this_run;

}


real2 Cudamoto2::generate_result(int2 r1r2_init, real t_final){

	set_initial_conditions(r1r2_init);

	auto r1r2_final = integrate_to_time(t_final);

	return r1r2_final;

}

/*
 * Function to find critical lambda in both directions using binary search, with one network having fixed lambda
 * fixed_net_lambda : const value of lambda to fix other network at
 * fixed_net_idx : index of network to be fixed (0 or 1)
 * precision : precision of lambda_c to find 1e-4 appears to be maximal reliable precision
 */
criticalValues Cudamoto2::find_lambda_fb(real fixed_net_lambda, int fixed_net_idx, real precision, real t_final){
	criticalValues output;
	output.deltar_b = 0; output.deltar_f = 0; output.lambda_b=0; output.lambda_f=0;
	real* d_theta_init;
	rvector h_theta_init(2*N);
	//int varying_net_idx = (fixed_net_idx+1)%2;
	for(auto & val: h_theta_init)
		val = PI_DEF*(2*randreal() - 1);
	checkCudaErrors(cudaMalloc((void ** )&d_theta_init, sizeof(real) * 2 * N));
	checkCudaErrors(
				cudaMemcpy(d_theta_init, &h_theta_init[0], sizeof(real) * 2 * N, cudaMemcpyHostToDevice));
	//real lambda_above = 1, lambda_below=0, current_lambda=0, r_above=0,t=0;
	//bool phaseTwo=false, init_is_synced=false;//phaseTwo is finding lambda_b, when we switch we want init synced
	lambda[fixed_net_idx] = fixed_net_lambda;
//	rvector rhist;
//	while(true){
//		if (phaseTwo && !init_is_synced){
//			//TODO: make synced and copy to theta_init
//			init_is_synced = true;
//			continue;
//		}
//		current_lambda = 0.5* ( lambda_above + lambda_below);
//		lambda[varying_net_idx] = current_lambda;
//		//TODO: figure out how many transitions to look for, then write this function.
//
//	}


	return output;
}

std::vector<real2> Cudamoto2::scan_lambda_values(real fixed_net_lambda, int fixed_net_idx, real lambda_init, real lambda_final, real lambda_step, real t_final){
	bool going_down=false;
	int varying_net_idx = (fixed_net_idx+1)%2;
	lambda[fixed_net_idx] = fixed_net_lambda;
	real current_lambda = lambda_init;
	std::vector<real2> rhist_this_run;
	std::vector<real2> rhist_all_runs;
	std::vector<real> lambda_hist;
	int ignore_first = 200;
	real r1=0,r2=0,t=0;

	while(true){
		lambda[varying_net_idx] = current_lambda;
		t=0;
		rhist_this_run.clear();
		while(t<t_final){
			integrate_one_step();
			t+=h;
			rhist_this_run.push_back(get_r_device());
		}
		r1=0;
		r2=0;
		for(auto r1r2_it = rhist_this_run.begin() + ignore_first; r1r2_it != rhist_this_run.end(); r1r2_it++){
			r1 += r1r2_it->x;
			r2 += r1r2_it->y;
		}
		r1 /= (rhist_this_run.size() - ignore_first);
		r2 /= (rhist_this_run.size() - ignore_first);

		/*
		 * Record results
		 */

		//lambda_hist.push_back(current_lambda);
		rhist_all_runs.push_back(makereal2(r1,r2));


		/*
		 * Change lambda or break
		 */
		if (!going_down && current_lambda + lambda_step > lambda_final)
			going_down=true;
		if (going_down)
			current_lambda -= lambda_step;
		else
			current_lambda += lambda_step;
		if(going_down && current_lambda < lambda_init){
			break;
		}
	}
	return rhist_all_runs;
}

std::vector<real3> Cudamoto2::scan_lambda_values_conserved_sum(real lambda_sum, real lambda_init, real lambda_final, real lambda_step, real t_final){
	bool going_down=false;

	lambda[0] = lambda_init;
	lambda[1] = lambda_sum - lambda_init;
	real current_lambda = lambda_init;
	std::vector<real2> rhist_this_run;
	std::vector<real3> rhist_all_runs;
	std::vector<real> lambda_hist;
	int ignore_first = 200;
	real r1=0,r2=0,t=0;
	while(true){
		lambda[0] = current_lambda;
		lambda[1] = lambda_sum - current_lambda;
		t=0;
		rhist_this_run.clear();
		while(t<t_final){
			integrate_one_step();
			t+=h;
			rhist_this_run.push_back(get_r_device());
		}
		r1=0;
		r2=0;
		for(auto r1r2_it = rhist_this_run.begin() + ignore_first; r1r2_it != rhist_this_run.end(); r1r2_it++){
			r1 += r1r2_it->x;
			r2 += r1r2_it->y;
		}
		r1 /= (rhist_this_run.size() - ignore_first);
		r2 /= (rhist_this_run.size() - ignore_first);

		std::cout << std::setprecision(7)  << std::fixed << current_lambda << ", " <<lambda_sum - current_lambda << " : " << r1 << ", "<<r2<<std::endl;
		std::cout.flush();
		/*
		 * Record results
		 */

		//lambda_hist.push_back(current_lambda);
		rhist_all_runs.push_back(makereal3(current_lambda,r1,r2));


		/*
		 * Change lambda or break
		 */
		if (!going_down && current_lambda + lambda_step > lambda_final)
			going_down=true;
		if (going_down)
			current_lambda -= lambda_step;
		else
			current_lambda += lambda_step;
		if(going_down && current_lambda < lambda_init){
			break;
		}
	}
	return rhist_all_runs;
}

vector< vector<real2> > Cudamoto2::scan_both_lambda_values_symmetric(real lambda_init, real lambda_final, real lambda_step, real t_final){
	vector< vector<real2> >  outer_lambda_scan;
	for(real outer_lambda = lambda_init; outer_lambda < lambda_final; outer_lambda +=lambda_step ){
		auto inner_lambda_r_scan = scan_lambda_values(outer_lambda,0,lambda_init,lambda_final,lambda_step,t_final);
		outer_lambda_scan.push_back(inner_lambda_r_scan);
	}
	return outer_lambda_scan;
}

real2 Cudamoto2::get_r_device(){

	int threads = 256;
	int blocks = min((N + threads - 1) / threads, 2048);
	real2 rvec;
	for(int net_idx=0; net_idx<2; net_idx++){
		real r;
		if(ismixed && net_idx ==1){
			checkCudaErrors(cudaMemsetAsync(d_x, real(0), sizeof(real)));
			device_reduce_sis_kfactor_warp_atomic_kernel_float2<<<blocks,threads>>>(d_theta,d_offsetList,d_x,N,net_idx*N);
			checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(real), cudaMemcpyDeviceToHost));
			real normFactor = N*kbar[net_idx];
			r = *h_x / normFactor;

		}else{
			checkCudaErrors(cudaMemsetAsync(d_x, real(0), sizeof(real)));
			checkCudaErrors(cudaMemsetAsync(d_y, real(0), sizeof(real)));
			// std::cout << "r_dev memset complete\n";
			if(isglobalOP)
				device_reduce_warp_atomic_kernel_float2<<<blocks,threads>>>(d_theta,d_x,d_y,N,net_idx*N);
			else
				device_reduce_kfactor_warp_atomic_kernel_float2<<<blocks,threads>>>(d_theta,d_offsetList,d_x,d_y,N,net_idx*N);
			// std::cout << "r_dev kernel complete\n";
			checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(real), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
			// std::cout << "r_dev memcpy complete\n";
			getLastCudaError("r_device execution failed");
			real normFactor = isglobalOP ? N : N*kbar[net_idx];
			r = sqrt((*h_x) * (*h_x) + (*h_y) * (*h_y)) / normFactor;
		}
		if(net_idx == 0)
			rvec.x = r;
		else
			rvec.y = r;

	}
	return rvec;


}

void Cudamoto2::copy_theta_to_host(){

	checkCudaErrors(cudaMemcpy(&theta[0], d_theta, sizeof(real)*2*N, cudaMemcpyDeviceToHost));
	getLastCudaError("copy_theta failed");


}

void Cudamoto2::local_attack(int rh, int net_idx, int target_r) {
	int L = sqrt(N);
	int center_point = (L/2)*L + L/2;
	if (rh >= L/2){
		if(target_r == 1)
			zero_oscillator_phases(net_idx);
		else
			randomize_oscillator_phases(net_idx);

		return;
	}

	copy_theta_to_host();

	int attack_size  = 0;
	std::uniform_real_distribution<real> randphase(-PI_DEF, PI_DEF);
	for(int dx = -rh; dx<= rh; dx++){
		for(int dy = -rh; dy<= rh; dy++){
//			std::cout << "("<<dx<<","<<dy<<")\n";
				if(dx*dx + dy*dy <  rh*rh)
				{
					theta[(center_point + dx*L +dy%L) + net_idx*N ] = target_r ==1? 0 : randphase(gen);
					attack_size++;
				}
			}
	}
	copy_theta_to_device();
	//std::cout << "Attack: " << attack_size << " nodes (rh = " << rh << ") attacked to state " << target_r << " in net " << net_idx << ".\n";
}

void Cudamoto2::copy_network_to_device() {
	cuda_free_if_alloced(d_offsetList);
	cuda_free_if_alloced(d_adjList);
	cuda_free_if_alloced(d_weightList);


	checkCudaErrors(cudaMalloc((void ** )&d_offsetList, sizeof(uint_t) * (2 * N + 1)));
	checkCudaErrors(cudaMalloc((void ** )&d_adjList, sizeof(uint_t) * flatAdjList.size()));
	checkCudaErrors(
			cudaMemcpy(d_offsetList, &flatOffsetList[0], sizeof(uint_t) * (2 * N + 1),
					cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(d_adjList, &flatAdjList[0], sizeof(uint_t) * (flatAdjList.size()),
					cudaMemcpyHostToDevice));
	if(isweighted){
		checkCudaErrors(cudaMalloc((void ** )&d_weightList, sizeof(real) * flatWeightList.size()));
		checkCudaErrors(
				cudaMemcpy(d_weightList, &flatWeightList[0], sizeof(real) * (flatWeightList.size()),
						cudaMemcpyHostToDevice));
	}
}

vector<real2> Cudamoto2::generate_history_changing_lambda(real lambda_init,
		real lambda_final, real lambda_step, real t_final) {
	vector<real2> rhist_this_run;
	real t=0;
	for(real lambda = lambda_init; lambda<=lambda_final; lambda+=lambda_step){
		set_lambda(lambda,lambda);
		t=0;
		while(t<t_final){
			integrate_one_step();
			t+=h;
			rhist_this_run.push_back(get_r_device());
		}
	}
	return rhist_this_run;

}

vector<real2> Cudamoto2::make_L_lambda_path_vector(real2 lambda_0, real2 lambda_1, real2 lambda_2, int total_steps){

	vector<real2> lambda_path;
	lambda_path.push_back(lambda_0);
	real slope_y = ( lambda_1.y - lambda_0.y );
	real slope_x = ( lambda_1.x - lambda_0.x) ;
	real d1 = sqrt((lambda_0.x - lambda_1.x)*(lambda_0.x - lambda_1.x) + (lambda_0.y - lambda_1.y)*(lambda_0.y - lambda_1.y));
	real d2 = sqrt((lambda_2.x - lambda_1.x)*(lambda_2.x - lambda_1.x) + (lambda_2.y - lambda_1.y)*(lambda_2.y - lambda_1.y));
	int first_steps = std::round(total_steps * (d1 / (d1 + d2)));
	for(int i =1; i<=first_steps; i++)
		lambda_path.push_back(makereal2(lambda_0.x + (i +0.0 )/(first_steps) * slope_x , lambda_0.y + (i +0.0 )/(first_steps) * slope_y));
	slope_y = ( lambda_2.y - lambda_1.y );
	slope_x = ( lambda_2.x - lambda_1.x) ;
	int second_steps = total_steps - first_steps;
	for(int i =1; i<=second_steps; i++)
		lambda_path.push_back(makereal2(lambda_1.x + (i +0.0 )/(second_steps) * slope_x , lambda_1.y + (i +0.0 )/(second_steps) * slope_y));
	return lambda_path;

}

vector<real2> Cudamoto2::make_straight_lambda_path_vector(real2 lambda_0, real2 lambda_1, int total_steps){

	vector<real2> lambda_path;
	lambda_path.push_back(lambda_0);
	real slope_y = (lambda_1.y - lambda_0.y);
	real slope_x = (lambda_1.x - lambda_0.x);

	for(int i =1; i<=total_steps; i++)
		lambda_path.push_back(makereal2(lambda_0.x + (i +0.0 )/(total_steps) * slope_x , lambda_0.y + (i +0.0 )/(total_steps) * slope_y));
	return lambda_path;

}

void Cudamoto2::prepare_lyapunov_measurements(){

	checkCudaErrors(cudaMalloc((void ** )&d_secondary_dist, sizeof(real)));
	checkCudaErrors(cudaMalloc((void ** )&d_theta_primary, sizeof(real) * 2 * N));
	checkCudaErrors(cudaMalloc((void ** )&d_theta_secondary, sizeof(real) * 2 * N));
	checkCudaErrors(cudaMalloc((void ** )&d_secondary_diff, sizeof(real) * 2 * N));

}

void Cudamoto2::switch_active_theta(int zero_if_primary){
	if(zero_if_primary == 0){
		d_theta = &d_theta_primary[0];
	} else {
		d_theta = &d_theta_secondary[0];
	}
}

real Cudamoto2::get_primary_secondary_difference(){
	int threads = 256;
	int blocks = min((2*N + threads - 1) / threads, 2048);
	real *dist;
	dist = (real*) malloc(sizeof(real));
	vectorDifference<<<blocks,threads>>>(d_secondary_diff,d_theta_secondary,d_theta_primary,2*N);
	checkCudaErrors(cudaMemsetAsync(d_secondary_dist, real(0), sizeof(real)));
	device_reduce_warp_atomic_kernel_1norm<<<blocks,threads>>>(d_secondary_diff, d_secondary_dist, 2*N);
	checkCudaErrors(cudaMemcpy(dist, d_secondary_dist, sizeof(real), cudaMemcpyDeviceToHost));
	return *dist;

}

void Cudamoto2::renormalize_secondary(real dist, real d0){
	int threads = 256;
	int blocks = min((2*N + threads - 1) / threads, 2048);
	real factor = d0 / dist;
	vectorSumMod2PiFactor<<<blocks,threads>>>(d_theta_secondary,d_theta_primary,d_secondary_diff,factor);
}

void Cudamoto2::lyapunov_scratchpad(){

	std::cout << "[";
	switch_active_theta(0);
	//set_initial_conditions(makereal2(0.5,0.5));
	std::cerr << "Selected primary:\td_theta points to : " << (void *)(d_theta)
					<< " d_theta_primary points to: " << (void *)(d_theta_primary)
					<< " d_theta_secondary points to: " << (void *)(d_theta_secondary)
					<< " d_k1 points to: "<< (void *)(d_k1) <<std::endl;
	checkCudaErrors(cudaMemcpy(&theta[0],d_theta_primary, sizeof(real)*2*N, cudaMemcpyDeviceToHost));
	output_theta(std::cout);
	std::cout << ",\n";
	switch_active_theta(1);
	//set_initial_conditions(makereal2(0.5,0.5));
	std::cerr << "Selected secondary:\td_theta points to : " << (void *)(d_theta)
					<< " d_theta_primary points to: " << (void *)(d_theta_primary)
					<< " d_theta_secondary points to: " << (void *)(d_theta_secondary)
					<< " d_k1 points to: "<< (void *)(d_k1) <<std::endl;
	checkCudaErrors(cudaMemcpy( &theta[0],d_theta_secondary,sizeof(real)*2*N, cudaMemcpyDeviceToHost));
	output_theta(std::cout);
	std::cout << "]";

}

void Cudamoto2::copy_theta_to_device(){

	checkCudaErrors(cudaMemcpy(d_theta, &theta[0],sizeof(real)*2*N, cudaMemcpyHostToDevice));
	getLastCudaError("copy_theta failed");


}


real2 Cudamoto2::get_r_host(){

		copy_theta_to_host();
		real2 rvec;
		for(int net_idx=0; net_idx<2; net_idx++){
			real x=0,y=0,j=0;
			for(int i = 0; i<N; i++){
				j = i + net_idx * N;
				x+=cos(theta[j]);
				y+=sin(theta[j]);
			}
			real r  = sqrt(x*x + y*y) / N;
			if(net_idx==0)
				rvec.x = r;
			else
				rvec.y = r;
		}
	return rvec;
}



Cudamoto2::~Cudamoto2() {
	/*
	checkCudaErrors(cudaFree(d_theta));
	checkCudaErrors(cudaFree(d_omega));
	checkCudaErrors(cudaFree(d_k1));
	checkCudaErrors(cudaFree(d_adjList));
	checkCudaErrors(cudaFree(d_x));
	checkCudaErrors(cudaFree(d_y));
	//TODO: check if used
	if(false){
	try{
	checkCudaErrors(cudaFree(d_theta_init_upup));
	checkCudaErrors(cudaFree(d_theta_init_updown));
	checkCudaErrors(cudaFree(d_theta_init_downup));
	checkCudaErrors(cudaFree(d_theta_init_downdown));
	} catch(std::exception e){
		std::cerr << e.what() << std::endl;
	}}
*/
	cudaDeviceReset();
}

