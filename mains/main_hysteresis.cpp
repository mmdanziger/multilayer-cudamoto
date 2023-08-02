#include "../cudamoto2/src/Cudamoto2.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <hdf5/serial/H5Cpp.h>

using std::cout;

/*
 *
/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl_cpp.a
/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_cpp.a
/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.a
/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.a
 * */
int main(int argc, char** argv){

	srand(time(0));
	uint_t N = argc > 1? 1<<atoi(argv[1]) : 1<<11;
	real kbar = argc >2? atof(argv[2]) : 6;
	real f = argc >3? atof(argv[3]) : 1;
	real2 lambda_0 = argc >4? char2real2(argv[4]) : makereal2(0.05,0.05);
	real2 lambda_1 = argc >5? char2real2(argv[5]) : makereal2(0.2,0.7);
	real2 lambda_2 = argc >6? char2real2(argv[6]) : makereal2(0.5,0.05);
	real steps = argc >7? atof(argv[7]) : 100;
	real t_final = argc >8? atof(argv[8]) : 20000;
	real h=0.01;
	int interaction_type = Cudamoto2::COMPETITIVE;
    std::stringstream ofilename;
    int isdirected = 0;
    int ismixed = 0;
	int r1up=0,r2up=0;
	Cudamoto2 c2(N,kbar,kbar,f,h);
	c2.set_interaction(interaction_type);
	auto pathvector = c2.make_straight_lambda_path_vector(lambda_0,lambda_1,steps);
	pathvector.insert (std::end(pathvector), pathvector.rbegin()+1,pathvector.rend());
	if(isdirected){
		c2.make_directed();
	}
	if(ismixed){
		c2.make_mixed();
	}
    ofilename << "hysteresis_run_N"<<N<<"_k"<<kbar<<"_f"<<f
    		<<"_x"<<interaction_type<<"_directed"<<isdirected
    		<<"_id"<<rand()%9<<rand()%9<<rand()%9<<rand()%9 <<".hdf5";
	auto ofile = H5::H5File(ofilename.str(),H5F_ACC_TRUNC);
	auto r1r2type = H5::CompType(sizeof(real2));
	r1r2type.insertMember("r1", HOFFSET(real2, x), H5::PredType::NATIVE_FLOAT);
	r1r2type.insertMember("r2", HOFFSET(real2, y), H5::PredType::NATIVE_FLOAT);
	vector<real2> r1r2hist;
	c2.set_initial_conditions(makeint2(0,0));
	int counter=0;
	for(auto & lam1lam2 : pathvector){

		c2.set_lambda(lam1lam2.x,lam1lam2.y);
		auto r1r2 = c2.integrate_to_time(t_final);
		r1r2hist.push_back(r1r2);
		std::cout << counter << "/"<<pathvector.size() << " : ( "<< lam1lam2.x << ","<< lam1lam2.y
				<< ") \t--> ("<< r1r2.x << ","<< r1r2.y<<")\n";
		counter++;
	}
	hsize_t dim[] = {r1r2hist.size()};   /* Dataspace dimensions */
	H5::DataSpace dataspace( 1, dim );

	{ //begin repeat scope
		std::stringstream runname;
		runname << "rvalues";
		auto  dataset = ofile.createDataSet(runname.str(),r1r2type,dataspace);
		dataset.write(&r1r2hist[0], r1r2type);
	} //end repeat scope

	{ //begin repeat scope
		std::stringstream runname;
		runname << "lamvalues";
		auto  dataset = ofile.createDataSet(runname.str(),r1r2type,dataspace);
		dataset.write(&pathvector[0], r1r2type);
	} //end repeat scope

	ofile.flush(H5F_SCOPE_GLOBAL);


	//cout << r1up << " , " << r2up << " out of " << r1r2hist.size() << " |_| score: " << score << std::endl;
	//cout.flush();

	ofile.close();
}
