#include "../cudamoto2/src/Cudamoto2.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <hdf5/serial/H5Cpp.h>

using std::cout;


int main(int argc, char** argv){

	srand(time(0));
	uint_t N = argc > 1? 1<<atoi(argv[1]) : 1<<11;
	real kbar = argc >2? atof(argv[2]) : 6;
	real f = argc >3? atof(argv[3]) : 1;
	real lambda1 = argc >4? atof(argv[4]) : 0;
	real lambda2 = argc >5? atof(argv[5]) : 1;
	int interaction_type = argc >6? atoi(argv[6]) : Cudamoto2::INTERDEPENDENT;
	real t_final = argc >7? atof(argv[7]) : 20000;
	real r_threshold = argc >8? atof(argv[8]) : 0.8;
	real h=0.01;
    int isdirected = 0;
	int r1up=0,r2up=0;
	Cudamoto2 c2(N,kbar,kbar,f,h);
	c2.set_interaction(interaction_type);
	c2.set_distribution(Cudamoto2::UNIFORM);
	c2.set_lambda(lambda1,lambda2);
	c2.initialize_oscillator_frequencies_and_phases();
	c2.copy_network_to_device();
	if(isdirected){
		c2.make_directed();
	}
	rvector init_points{0.5};//1, 0.4, 0.6, 0.8};




//	for(float lambda = lambda1; lambda < lambda2; lambda+=lambda_step){
		c2.set_lambda(lambda1,lambda2);
	    std::stringstream ofilename;

	    ofilename << "single_run_N"<<N<<"_k"<<kbar<<"_f"<<f
	    		<<"lambdaA"<<lambda1<<"_lambdaB"<<lambda2
	    		<<"_x"<<c2.get_interaction()<<"_directed"<<isdirected
	    		<<"_id"<<rand()%9<<rand()%9<<rand()%9<<rand()%9 <<".hdf5";
		auto ofile = H5::H5File(ofilename.str(),H5F_ACC_TRUNC);
		auto r1r2type = H5::CompType(sizeof(real2));
		r1r2type.insertMember("r1", HOFFSET(real2, x), H5::PredType::NATIVE_FLOAT);
		r1r2type.insertMember("r2", HOFFSET(real2, y), H5::PredType::NATIVE_FLOAT);
		int run_count = 0;
	for(auto & r1_init : init_points){
			auto r2_init = 1 - r1_init;
			c2.set_initial_conditions(makereal2(r1_init,r2_init));
			vector<real2> r1r2hist;
			real t=0;
			while(t<t_final){
				c2.integrate_one_step();
				t+=h;
				auto r1r2 = c2.get_r_device();
				if(r1r2.x > r_threshold)
					r1up++;
				if(r1r2.y > r_threshold)
					r2up++;

				r1r2hist.push_back(r1r2);
				//if (!r1r2hist.size()%500)
				//	cout <<t << " : "<< r1r2.x << "\t"<<r1r2.y<<"\n";
			}
			hsize_t dim[] = {r1r2hist.size()};   /* Dataspace dimensions */
			H5::DataSpace dataspace( 1, dim );
			std::stringstream runname;
			runname << "run_N" <<N << "_lambdaA"<<lambda1<< "_lambdaB"<<lambda2<<"_interaction"<<c2.get_interaction()<<"_rAInit"
					<< r1_init <<"_rBInit"<<r2_init<<"_count"<<run_count;
			auto  dataset = ofile.createDataSet(runname.str(),r1r2type,dataspace);
			dataset.write(&r1r2hist[0], r1r2type);
			ofile.flush(H5F_SCOPE_GLOBAL);
			run_count++;
		}

	ofile.close();

//	}
	//cout << r1up << " , " << r2up << " out of " << r1r2hist.size() << " |_| score: " << score << std::endl;
	//cout.flush();

}
