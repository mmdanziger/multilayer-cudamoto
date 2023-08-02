.PHONY: all clean cudamoto2_lib cudamoto2_viewer all_mains $(MAIN_TARGETS)

# List of main source files
MAIN_SRCS := $(wildcard mains/main_*.cpp)

# Derive the main targets from the source files
MAIN_TARGETS := $(patsubst mains/%.cpp,%,$(MAIN_SRCS))

all: cudamoto2_lib cudamoto2_viewer all_mains

cudamoto2_lib:
	@mkdir -p cudamoto2/build
	@cd cudamoto2/build && \
	nvcc -O3 -DCOS_FACTOR -Xcompiler="-fPIC" -std=c++11 -x cu -I.. -dc ../src/Cudamoto2.cu -o Cudamoto2.o && \
	nvcc -std=c++11 -O3 -DCOS_FACTOR -dlink -Xcompiler="-fPIC" -o libCudamoto2.o Cudamoto2.o -lcudart && \
	ar rc libCudamoto2.so libCudamoto2.o Cudamoto2.o && \
	ranlib libCudamoto2.so && \
	cp libCudamoto2.so ../../cudamoto2-viewer/src && \
	cp libCudamoto2.so ../../mains/

cudamoto2_viewer: cudamoto2_lib
	@mkdir -p cudamoto2-viewer/build
	@cd cudamoto2-viewer/build && \
	/usr/bin/qmake ../src && \
	make

$(MAIN_TARGETS): % : mains/%.cpp cudamoto2_lib
	@mkdir -p mains/build
	@g++ -std=c++11 -O3 -I/usr/include/hdf5/serial -I/usr/local/cuda/include/ -I/usr/local/cuda/samples/common/inc -c $< -o mains/build/$@.o
	@g++ -std=c++11 -O3 -o $@ mains/build/$@.o -L./mains/ -lCudamoto2 -L/usr/local/cuda/lib64 -lcudart -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5_cpp -lhdf5

all_mains: $(MAIN_TARGETS)

clean:
	@rm -rf cudamoto2/build cudamoto2-viewer/build mains/build *.bin ../../libCudamoto2.so
