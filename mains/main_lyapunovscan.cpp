#include "../cudamoto2/src/Cudamoto2.h"
#include <iostream>
#include <fstream>
#include <sstream>

using std::cout;

/**
 * This program scans values of lambda and outputs the final value of the lyapunov exponent.
 */

struct lyapunovResult{
	real lambda1,lambda2;
	real lyapunov;
	real2 ravg;
	real2 rstd;

};
template<typename T> T& operator<<(T& stream, lyapunovResult& res){
		stream 	<< "{\"lambda\" : [" << res.lambda1 <<", "<<res.lambda2 << "], "
				<<  "\"lyapunov\" : " << res.lyapunov << ","
				<<  "\"ravg\" : ["    << res.ravg.x <<", "<<res.ravg.y << "], "
				<<  "\"rstd\" : ["    << res.rstd.x <<", "<<res.rstd.y << "] }\n";

	return stream;
	}

int main(int argc, char** argv){

	srand(time(0));
	uint_t N = argc > 1? 1<<atoi(argv[1]) : 1<<11;
	real kbar = argc >2? atof(argv[2]) : 12;
	real f = argc >3? atof(argv[3]) : 1;
	real lambda_init = argc >4? atof(argv[4]) : 0;
	real lambda_final = argc >5? atof(argv[5]) : 1;
	real lambda_step = argc >6? atof(argv[6]) : 0.02;
	real t_final = argc >7? atof(argv[7]) : 100;
	real d0 = argc >8? atof(argv[8]) : 0.01;
	real h=0.01;
	real dist=0,dist2=0;
	int transient = 500;
	auto interaction_type = Cudamoto2::HYBRID;
  std::stringstream ofilename;

	ofilename << "lyapunov_scan_result_N"<<N<<"_k"<<kbar<<"_f"<<f
				<<"_lambdai"<<lambda_init<<"_lambdaf"<<lambda_final
		    		<<"_x"<<interaction_type<< "_id"
		    		<<rand()%9<<rand()%9<<rand()%9<<rand()%9<<".json";
	std::vector<lyapunovResult> result_vector;
	Cudamoto2 c2(N,kbar,kbar,f,h);
	c2.copy_network_to_device();
	c2.set_interaction(interaction_type);
	c2.prepare_lyapunov_measurements();

	c2.switch_active_theta(0);
	c2.initialize_oscillator_frequencies_and_phases();
	c2.allocate_and_copy_to_device();

	c2.switch_active_theta(1);
	c2.initialize_oscillator_frequencies_and_phases();
	c2.allocate_and_copy_to_device();


	for(real lambda1 = lambda_init; lambda1 < lambda_final; lambda1+=lambda_step){
		for(real lambda2 = lambda_init; lambda2 < lambda_final; lambda2+=lambda_step){
			c2.set_lambda(lambda1,lambda2);
			real t=0,tau=h;
			rvector dvector,d2vector;
			d2vector.push_back(dist);
			vector<real2> rhistory;

			c2.switch_active_theta(0);
			c2.set_initial_conditions(makereal2(0.5,0.5));
			c2.switch_active_theta(1);
			c2.set_initial_conditions(makereal2(0.5,0.5));

			dist = c2.get_primary_secondary_difference();
			//std::cerr << "initital diff: " << dist << std::endl;
			c2.renormalize_secondary(dist,d0);


			while(t<t_final){
				real taut=0;
				while(taut < tau){
					c2.switch_active_theta(0);
					c2.integrate_one_step();
					auto r = c2.get_r_device();
					c2.switch_active_theta(1);
					c2.integrate_one_step();
					rhistory.push_back(r);
					taut+=h;
					t+=h;
				}
				dist = c2.get_primary_secondary_difference();
				c2.renormalize_secondary(dist,d0);
				//std::cerr << "Renormalized " << dist2 << " -> " << dist << "\n";
				dvector.push_back(dist);
				dist2 = c2.get_primary_secondary_difference();
				d2vector.push_back(dist2);


			}
			d2vector.pop_back(); //because the last d2 is unnecessary and makes the containers uneven

			real2 ravg = vector_average(rhistory, transient);
			real2 rstd = vector_std(rhistory, ravg, transient);

			double lyapunov_exponent = 0;

			for(int i=0; i<dvector.size(); i++){
				real d = dvector[i],d2 = d2vector[i];
				lyapunov_exponent += log(d/d2);
			}
			lyapunov_exponent /= t;
			std:: cout	<< "[ "<<lambda1 <<" , " << lambda2 <<"] \t: " <<lyapunov_exponent
						<< "\travg: ["<<ravg.x <<" , "<<ravg.y<<"]"
						<< "\trstd: ["<<rstd.x <<" , "<<rstd.y<<"]" <<std::endl;
			lyapunovResult this_result;
			this_result.lambda1 = lambda1; this_result.lambda2 = lambda2; this_result.lyapunov = lyapunov_exponent;
			this_result.ravg = ravg; this_result.rstd = rstd;
			result_vector.emplace_back(this_result);
		}
	}

	std::ofstream of(ofilename.str());

	of << "[";
	bool firstVal = true;
	for(auto & res : result_vector){
		if(!firstVal){
			of << "," << res;
		} else {
			of << res;
			firstVal = false;
		}
	}
	of << "]\n";
	of.close();
	//std::cerr << "after renormalization: " << dist << std::endl;
/*
 * snippet to dump thetas (1)
	std::ofstream th1of("/tmp/th1.txt");
	std::ofstream th2of("/tmp/th2.txt");

	c2.switch_active_theta(0);
	c2.copy_theta_to_host();
	c2.output_theta(th1of);

	c2.switch_active_theta(1);
	c2.copy_theta_to_host();
	c2.output_theta(th2of);
	th1of.close(); th2of.close();
*/


/*
 *  snippet to dump thetas (2)
	std::ofstream th1oftag("/tmp/th1tag.txt");
	std::ofstream th2oftag("/tmp/th2tag.txt");

	c2.switch_active_theta(0);
	c2.copy_theta_to_host();
	c2.output_theta(th1oftag);

	c2.switch_active_theta(1);
	c2.copy_theta_to_host();
	c2.output_theta(th2oftag);
	th1oftag.close(); th2oftag.close();
*/




return 0;
}
