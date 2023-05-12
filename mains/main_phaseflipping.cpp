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
	real lambda_init = argc >4? atof(argv[4]) : 0;
	real lambda_final = argc >5? atof(argv[5]) : 1;
	real lambda_step = argc >6? atof(argv[6]) : 0.02;
	real t_final = argc >7? atof(argv[7]) : 20000;
	real r_threshold = argc >8? atof(argv[8]) : 0.3;
	real h=0.01;

	int Npower_init = 6;
	int Npower_final = 13;
    std::stringstream ofilename;
    auto lambda = lambda_init;
    ofilename << "history_lambda"<<lambda<<rand()%9<<rand()%9<<rand()%9<<rand()%9 <<".hdf5";
	auto ofile = H5::H5File("history.hdf5",H5F_ACC_TRUNC);
	auto r1r2type = H5::CompType(sizeof(real2));
	r1r2type.insertMember("r1", HOFFSET(real2, x), H5::PredType::NATIVE_FLOAT);
	r1r2type.insertMember("r2", HOFFSET(real2, y), H5::PredType::NATIVE_FLOAT);
	for(int Npower=Npower_init; Npower<Npower_final; Npower++){
		int r1up=0,r2up=0;
		N = 1<<Npower;
		Cudamoto2 c2(N,kbar,kbar,f,h);
		c2.make_competitive();
		c2.set_initial_conditions(makeint2(0,0));
		c2.set_lambda(lambda,lambda);
		cout << lambda << " : ";
		cout.flush();
		auto r1r2hist = c2.generate_history_to_time(t_final);
		hsize_t dim[] = {r1r2hist.size()};   /* Dataspace dimensions */
	    H5::DataSpace dataspace( 1, dim );
	    std::stringstream runname;
	    runname << "run_N" <<N << "_lambda"<<lambda;
		auto  dataset = ofile.createDataSet(runname.str(),r1r2type,dataspace);

		dataset.write(&r1r2hist[0], r1r2type);
		ofile.flush(H5F_SCOPE_GLOBAL);
		for(auto r1r2_it : r1r2hist){
			if(r1r2_it.x > r_threshold)
				r1up++;
			if(r1r2_it.y > r_threshold)
				r2up++;
	}
	real score = static_cast<real>(r2up - r1up) / r1r2hist.size();
	cout << r1up << " , " << r2up << " out of " << r1r2hist.size() << " |_| score: " << score << std::endl;
	cout.flush();
	}
	ofile.close();
}
