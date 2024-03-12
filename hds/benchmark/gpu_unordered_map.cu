#include <random>
#include <thread>
#include <barrier>
#include <iostream>
#include <chrono>
#include <getopt.h>
#include <unordered_set>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <rome/hds/unordered_map/gpu_unordered_map.h>

int test(auto& map, int read_percent, int population, int range, int ops) {

  std::default_random_engine gen;
  std::uniform_int_distribution<long int> key_dist(1, range);

  std::unordered_set<long int> insert_keys_set;

  while (insert_keys_set.size() < population) {
    insert_keys_set.insert(key_dist(gen));
  }

  {
    thrust::host_vector<long int> h_insert_keys(insert_keys_set.begin(), insert_keys_set.end());
    thrust::device_vector<long int> insert_keys = h_insert_keys;
    thrust::device_vector<long int> insert_values(population, 1);
    thrust::device_vector<bool> insert_result(population);

    map.insert(insert_keys.data().get(), insert_values.data().get(), insert_result.data().get(), insert_keys.size()).wait();
  }

  std::uniform_int_distribution<long int> op_dist(0, 99);

  thrust::host_vector<long int> h_get_keys;
  thrust::host_vector<long int> h_insert_keys;
  thrust::host_vector<long int> h_remove_keys;

  bool last_insert = false;
  for(int i = 0; i < ops; ++i) {
    if (op_dist(gen) < read_percent) {
      h_get_keys.push_back(key_dist(gen));
    } else if (last_insert) {
      h_remove_keys.push_back(key_dist(gen));
      last_insert = false;
    } else {
      h_insert_keys.push_back(key_dist(gen));
      last_insert = true;
    }
  }

  thrust::device_vector<long int> get_keys = h_get_keys;
  thrust::device_vector<rome::hds::optional<long int>> get_result(h_get_keys.size());
  thrust::device_vector<long int> insert_keys = h_insert_keys;
  thrust::device_vector<long int> insert_values(h_insert_keys.size());
  thrust::device_vector<bool> insert_result(h_insert_keys.size());
  thrust::device_vector<long int> remove_keys = h_remove_keys;
  thrust::device_vector<bool> remove_result(h_remove_keys.size());

  std::future<void> bar;
  auto start = std::chrono::high_resolution_clock::now();
  
  if(get_keys.size()) {
    bar = map.get(get_keys.data().get(), get_result.data().get(), get_keys.size());
  }
  if (insert_keys.size()) {
   bar = map.insert(insert_keys.data().get(), insert_values.data().get(), insert_result.data().get(), insert_keys.size());
  }
  if (remove_keys.size()) {
    bar = map.remove(remove_keys.data().get(), remove_result.data().get(), remove_keys.size());
  }
  bar.wait();
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << ops / std::chrono::duration<double>(end - start).count() << std::endl;

  return 0;
}

void usage(char** argv) {
  std::cout << "Usage: " << argv[0] 
    << " [-p|--read_percent <percent>]"
    << " [-r|--range <range>]" 
    << " [--population <pop>]"
    << " [-o|--ops <ops per thread>]"
    << " [-s|--size <size>]"
    << " [-v|--verbose]"
    << " [-h|--help]" << std::endl;
}

int main(int argc, char** argv) {

  bool verbose = false;
  int read_percent = 95;
  int range = 1000000;
  int population = range / 2;
  int ops = 1000000;
  int size = population;

  int c;
  int index;

  struct option long_options[] = {
    {"read_percent", required_argument, 0, 'p'},
    {"range", required_argument, 0, 'r'},
    {"population", required_argument, 0, 0},
    {"ops", required_argument, 0, 'o'},
    {"size", required_argument, 0, 's'},
    {"help", no_argument, 0, 'h'},
    {"verbose", no_argument, 0, 'v'},
    {nullptr, 0, 0, 0}
  };

  while(true) {
    c = getopt_long(argc, argv, "p:r:o:s:hv", long_options, &index);

    if(c == -1)
      break;
    
    switch (c) {
      case 0: // long only option
        if(std::string(long_options[index].name) == "population") {
          population = atoi(optarg);
        }
        else {
          usage(argv);
          exit(1);
        }
        break;
      case 'p':
        read_percent = atoi(optarg);
        break;
      case 'r':
        range = atoi(optarg);
        break;
      case 'o':
        ops = atoi(optarg);
        break;
      case 's':
        size = atoi(optarg);
        break;
      case 'v':
        verbose = true;
        break;
      case 'h':
        usage(argv);
        exit(0);
        break;
      default:
        usage(argv);
        exit(1);
    }

  }
  
  if (cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1ull << 30) != cudaSuccess) {
    throw std::runtime_error("Unable to get heap size");
  }


  if (verbose) {

    size_t heap_size;
    if (cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize) != cudaSuccess) {
      throw std::runtime_error("Unable to get heap size");
    }

    std::cout << "Heap size: " << heap_size / (1024.0 * 1024.0 * 1024.0) << "GiB" << std::endl;

    std::cout << "Read percent " << read_percent << std::endl;
    std::cout << "Range " << range << std::endl;
    std::cout << "Population " << population << std::endl;
    std::cout << "Operations per thread " << ops << std::endl;
    std::cout << "Size " << size << std::endl;
  }

  rome::hds::gpu_unordered_map<long int, long int> map(size);

  test(map, read_percent, population, range, ops);

  return 0;
}

