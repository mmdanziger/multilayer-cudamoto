#include "../cudamoto2/src/Cudamoto2.h"
#include <iostream>
#include <fstream>
using std::cout;

/**
 * This program calculates and outputs the entire evolution of a lyapunov exponent measurement (not just the final value).
 */

int main(int argc, char** argv){

	srand(time(0));
	uint_t N = argc > 1? 1<<atoi(argv[1]) : 1<<11;
	real kbar = argc >2? atof(argv[2]) : 12;
	real f = argc >3? atof(argv[3]) : 1;
	real lambda1 = argc >4? atof(argv[4]) : 0;
	real lambda2 = argc >5? atof(argv[5]) : 1;
	real lambda_step = argc >6? atof(argv[6]) : 0.02;
	real t_final = argc >7? atof(argv[7]) : 100;
	real d0 = argc >8? atof(argv[8]) : 0.01;
	real h=0.01;
	real dist=0,dist2=0;
	auto interaction_type = Cudamoto2::HYBRID;
  std::stringstream ofilename;

	ofilename << "lyapunov_N"<<N<<"_k"<<kbar<<"_f"<<f
				<<"lambdaA"<<lambda1<<"_lambdaB"<<lambda2
		    		<<"_x"<<interaction_type<< "_id"
		    		<<rand()%9<<rand()%9<<rand()%9<<rand()%9<<".json";

	Cudamoto2 c2(N,kbar,kbar,f,h);
	c2.set_lambda(lambda1,lambda2);
	c2.copy_network_to_device();
	c2.set_interaction(interaction_type);
	c2.prepare_lyapunov_measurements();

	c2.switch_active_theta(0);
	c2.initialize_oscillator_frequencies_and_phases();
	c2.allocate_and_copy_to_device();

	c2.switch_active_theta(1);
	c2.initialize_oscillator_frequencies_and_phases();
	c2.allocate_and_copy_to_device();


	dist = c2.get_primary_secondary_difference();

	//std::cerr << "initital diff: " << dist << std::endl;
	c2.renormalize_secondary(dist,d0);
	dist = c2.get_primary_secondary_difference();
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

	real t=0,tau=h;
	rvector dvector,d2vector;
	d2vector.push_back(dist);
	vector<real2> rhistory;
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
	std::ofstream dout(ofilename.str());
	bool firstval=true;
	double lyapunov_exponent = 0;
	dout <<"[";
	for(int i=0; i<dvector.size(); i++){
		real d = dvector[i],d2 = d2vector[i];
		lyapunov_exponent += log(d/d2);
		if(!firstval)
		dout <<",\n"<< "["<<d<<","<<d2<<"]" ;
		else{
			dout << "["<<d<<","<<d2<<"]" ;
			firstval=false;
		}
	}
	dout << "]";
	dout.close();
	lyapunov_exponent /= t;
	std:: cout << "Calculated Lyapunov exponent : " <<lyapunov_exponent << std::endl;


	std::ofstream rout("rout.json");
	firstval=true;

	rout <<"[";
	for(auto r : rhistory){
		if(!firstval)
			rout <<",\n"<< "["<<r.x << " , "<<r.y<<"]";
		else{
			rout <<  "["<<r.x << " , " <<r.y <<"]";
			firstval=false;
		}
	}
	rout << "]";
	rout.close();

return 0;
}
