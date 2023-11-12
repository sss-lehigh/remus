all:
	@CXX=clang++-15 cmake -B build -DCMAKE_BUILD_TYPE=Release
	@cmake --build build -j8 # -v

clean:
	@rm -rf build
