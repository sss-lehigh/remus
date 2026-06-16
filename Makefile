include MakefileBenchmark
all:
	@CXX=clang++-18 cmake -B build -DCMAKE_BUILD_TYPE=Release
	@cmake --build build -j8 # -v

clean:
	@rm -rf build
	@rm -rf run.screenrc
	@rm -rf dev.screenrc

.PHONY: all clean benchmark
