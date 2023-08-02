#include "../cudamoto2/src/Cudamoto2.h"
#include <iostream>
#include <fstream>
using std::cout;


int main(int argc, char** argv){

	srand(time(0));
	uint_t N = argc > 1? 1<<atoi(argv[1]) : 1<<11;
	real kbar = argc >2? atof(argv[2]) : 12;
	real f = argc >3? atof(argv[3]) : 1;
	real lambda_init = argc >4? atof(argv[4]) : 0;
	real lambda_final = argc >5? atof(argv[5]) : 1;
	real lambda_step = argc >6? atof(argv[6]) : 0.02;
	real t_final = argc >7? atof(argv[7]) : 200;
	real h=0.01;

	Cudamoto2 c2(N,kbar,kbar,f,h);

	if (strstr(argv[0],"comp")){
		std::cout << "Competitive\n";
		c2.make_competitive();
	}else if (strstr(argv[0],"inter")){
		std::cout << "Interdependent\n";
		c2.make_interdependent();
	}else if (strstr(argv[0],"hybrid")){
                std::cout << "Hybrid\n";
                c2.make_hybrid();
	}else{
		std::cerr<< "Don't recognize interaction...switching to interdependent\n";
		c2.make_interdependent();
	}
	int2 r1r2_init;
	vector<initialConditionResult> result_vector;
	for (real lambda1 = lambda_init; lambda1<lambda_final; lambda1+=lambda_step){
		for (real lambda2 = lambda_init; lambda2<lambda_final; lambda2+=lambda_step){
			c2.set_lambda(lambda1,lambda2);
			initialConditionResult result;
			result.lambda1 = lambda1; result.lambda2 = lambda2;
			r1r2_init.x=0; r1r2_init.y=0;
			result.r1r2_00 = c2.generate_result(r1r2_init,t_final);
			r1r2_init.x=1; r1r2_init.y=0;
			result.r1r2_10 = c2.generate_result(r1r2_init,t_final);
			r1r2_init.x=1; r1r2_init.y=1;
			result.r1r2_11 = c2.generate_result(r1r2_init,t_final);
			result_vector.push_back(result);
			std::cout << result;
		}
	}
	std::stringstream ofname;
	ofname << "ScanLambda_N"<<N<<"_k"<<kbar<<"_f"<<f<<"_x"<<c2.get_interaction() <<"_t"<<t_final<<"_id"<<rand()%9<<rand()%9<<rand()%9<<rand()%9<<rand()%9<<".json";
	std::ofstream ofile(ofname.str());

	bool first_line=true;
	ofile <<"[";
	for(auto& result : result_vector){
		if(!first_line)
			ofile <<",";
		ofile << result ;
		first_line=false;
	}
	ofile <<"]\n";
	return 0;
}
/**
 * Code for scanning diagonal of lambda_1 lambda_2 phase space
 */
/*
	std::ofstream ofile1("/tmp/scan_r1.txt");
	//std::ofstream ofile2("/tmp/scan_r2.txt");
	c2.make_interdependent();
	auto comp_scan = c2.scan_lambda_values_conserved_sum(lambda_final,lambda_init,lambda_final,lambda_step,t_final);
	bool first_out=true;
	ofile1 << "[\n";
	for(auto res3 : comp_scan){
		if(!first_out)
			ofile1 << "\n,";
		if(first_out);
			first_out=false;
		ofile1 << "[" << res3.x << ", " << res3.y << ", "<< res3.z << "]";
	}
	ofile1 << "\n]";
	ofile1.close();
	*/

	/**
	 * Code for generating 2d matrix of lambda values
	 */
	/*
	std::ofstream ofile1("/tmp/scan_r1.txt");
	std::ofstream ofile2("/tmp/scan_r2.txt");
	c2.make_interdependent();
	auto double_scan = c2.scan_both_lambda_values_symmetric(lambda_init,lambda_final,lambda_step,t_final);

	//for(real lam = lambda_init; lam<lambda_final; lam+=lambda_step){
	for(auto &run : double_scan){
		for(auto& rvec: run){
			ofile1 << rvec.x << " ";
			ofile2 << rvec.y << " ";
		}
		ofile1 << "\n";
		ofile2 << "\n";
	}
	ofile1.close();
	ofile2.close();

}
*/
/**
 * dummy main
 *
 */
/*
int main(int argc, char** argv){

	srand(time(0));
	uint_t N = argc > 1? 1<<atoi(argv[1]) : 1<<14;
	real kbar1 = argc >2? atof(argv[2]) : 12;
	real kbar2 = argc >3? atof(argv[3]) : 6;
	real f = argc >4? atof(argv[4]) : 1;
	real h=0.01;
	Cudamoto2 c2(N,kbar1,kbar2,f,h);
	float2 rvec = c2.get_r_device();
	c2.set_lambda(0,0);
	std::cout << "Lambda : (0,0) \t";
	std::cout << "From : ";
	std::cout << "(" <<rvec.x << "," << rvec.y << ")";
	for(int i =0; i<5000; i++){
		c2.integrate_one_step();
		//std::cout << i<< " ";
	}
	rvec = c2.get_r_device();
	std::cout << " to ";
	std::cout << "(" <<rvec.x << "," << rvec.y << ")" << std::endl;

	c2.set_lambda(0,2);
	std::cout << "Lambda : (0,2) \t";
	std::cout << "From : ";
	std::cout << "(" <<rvec.x << "," << rvec.y << ")";
	for(int i =0; i<5000; i++){
		c2.integrate_one_step();
		//std::cout << i<< " ";
	}
	rvec = c2.get_r_device();
	std::cout << " to ";
	std::cout << "(" <<rvec.x << "," << rvec.y << ")" << std::endl;

	c2.set_lambda(2,0);
	std::cout << "Lambda : (2,0) \t";
	std::cout << "From : ";
	std::cout << "(" <<rvec.x << "," << rvec.y << ")";
	for(int i =0; i<5000; i++){
		c2.integrate_one_step();
		//std::cout << i<< " ";
	}
	rvec = c2.get_r_device();
	std::cout << " to ";
	std::cout << "(" <<rvec.x << "," << rvec.y << ")" << std::endl;

	c2.set_lambda(2,2);
	std::cout << "Lambda : (2,2) \t";
	std::cout << "From : ";
	std::cout << "(" <<rvec.x << "," << rvec.y << ")";
	for(int i =0; i<5000; i++){
		c2.integrate_one_step();
		//std::cout << i<< " ";
	}
	rvec = c2.get_r_device();
	std::cout << " to ";
	std::cout << "(" <<rvec.x << "," << rvec.y << ")" << std::endl;

}
*/
