all: release debug

release:
	@CXX=clang++-18 cmake -B build -DCMAKE_BUILD_TYPE=Release
	@cmake --build build -j

clean:
	@rm -rf build
	@rm -rf build_debug
	@rm -rf run.screenrc
	@rm -rf dev.screenrc

debug:
	@CXX=clang++-18 cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
	@cmake --build build_debug -j

.PHONY: all clean release debug
