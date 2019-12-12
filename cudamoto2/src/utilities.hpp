#ifndef UTILITIES_H_
#define UTILITIES_H_

inline
int2 makeint2(int x, int y){
	int2 out;
	out.x = x;
	out.y = y;
	return out;
}

inline
real2 makereal2(real x, real y){
	real2 out;
	out.x = x;
	out.y = y;
	return out;
}


inline
real3 makereal3(real x, real y, real z){
	real3 out;
	out.x = x;
	out.y = y;
	out.z = z;
	return out;
}

inline
real2 char2real2(char* input){
	char* currnum;
    currnum = strtok(input, ",");
    float x =  atof(currnum);
    currnum = strtok(NULL, ",");
    float y = atof(currnum);
    return makereal2(x,y);
}

inline
double millisecond_res_diff(std::chrono::high_resolution_clock::time_point end, std::chrono::high_resolution_clock::time_point start) {
	 std::chrono::milliseconds  time_difference = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return( time_difference.count() / 1000.0);
}



inline
uint_t randint(uint_t max){
    return static_cast<real>(max) * (static_cast<real>(random())/static_cast<real>(RAND_MAX));

}


inline
real randreal(){
    return  static_cast<real>(random())/static_cast<real>(RAND_MAX);

}

inline
real2 vector_average(std::vector<real2>& v, int transient){
	real2 out = makereal2(0,0);
	for(auto xy = v.begin() + transient; xy != v.end(); ++xy){
		out.x += xy->x;
		out.y += xy->y;
	}
	out.x/=(v.size() - transient);
	out.y/=(v.size() - transient);
	return out;
}

inline
real2 vector_std(std::vector<real2>& v, real2 avg, int transient){
	real2 out = makereal2(0,0);
	for(auto xy = v.begin() + transient; xy != v.end(); ++xy){
		out.x += (xy->x - avg.x)*(xy->x - avg.x);
		out.y += (xy->y - avg.y)*(xy->y - avg.y);
	}
	out.x = std::sqrt(out.x);
	out.y = std::sqrt(out.y);
	out.x/=(v.size() - transient);
	out.y/=(v.size() - transient);
	return out;
}



#endif
