#include <random>
#include <thread>
#include <barrier>
#include <iostream>
#include <chrono>
#include <getopt.h>

#include <rome/hds/allocator/allocator.h>
#include <rome/hds/unordered_map/kv_linked_list/lock_linked_list.h>
#include <rome/hds/unordered_map/kv_linked_list/locked_nodes/reg_cached_nodes.h>
#include <rome/hds/unordered_map/unordered_map.h>
#include <rome/hds/threadgroup/threadgroup.h>

int test(auto map, int read_percent, int population, int range, int nthreads, int ops) {

  std::default_random_engine gen;
  std::uniform_int_distribution<int> key_dist(1, range);

  auto group = rome::hds::threadgroup::single_threadgroup{};
  int count = 0;
  while (count < population) {
    if (map->insert(key_dist(gen), 1, group)) {
      count++;
    }
  }

  std::vector<std::jthread> threads;
  std::barrier bar(nthreads + 1);

  for(int i = 0; i < nthreads; ++i) {
    threads.push_back(std::jthread([&map, range, read_percent, &bar, ops](int tid){
      std::default_random_engine gen;
      std::uniform_int_distribution<int> key_dist(1, range);
      std::uniform_int_distribution<int> op_dist(0, 99);
      bool last_insert = false;
      auto group = rome::hds::threadgroup::single_threadgroup{};

      bar.arrive_and_wait();
      
      for (int i = 0; i < ops; ++i) {
        int key = key_dist(gen);
        if (op_dist(gen) < read_percent) {
          map->get(key, group);
        } else {
          if (last_insert) {
            map->remove(key, group);
            last_insert = false;
          } else {
            map->insert(key, 1, group);
            last_insert = true;
          }
        }
      }

      bar.arrive_and_wait();

    }, i));
  }

  bar.arrive_and_wait();
  auto start = std::chrono::high_resolution_clock::now();
  bar.arrive_and_wait();
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << ops * nthreads / std::chrono::duration<double>(end - start).count() << std::endl;

  return 0;
}

void usage(char** argv) {
  std::cout << "Usage: " << argv[0] 
    << " [-p|--read_percent <percent>]"
    << " [-r|--range <range>]" 
    << " [--population <pop>]"
    << " [-t|--thread <nthreads>]"
    << " [-o|--ops <ops per thread>]"
    << " [-s|--size <size>]"
    << " [-n|--node_size <node size>]"
    << " [-v|--verbose]"
    << " [-h|--help]" << std::endl;
}

int main(int argc, char** argv) {

  bool verbose = false;
  int read_percent = 95;
  int range = 1000000;
  int population = range / 2;
  int nthreads = std::thread::hardware_concurrency();
  int ops = 1000000;
  int size = population;
  int node_size = 1;

  int c;
  int index;

  struct option long_options[] = {
    {"read_percent", required_argument, 0, 'p'},
    {"range", required_argument, 0, 'r'},
    {"population", required_argument, 0, 0},
    {"thread", required_argument, 0, 't'},
    {"ops", required_argument, 0, 'o'},
    {"size", required_argument, 0, 's'},
    {"node_size", required_argument, 0, 'n'},
    {"help", no_argument, 0, 'h'},
    {"verbose", no_argument, 0, 'v'},
    {nullptr, 0, 0, 0}
  };

  while(true) {
    c = getopt_long(argc, argv, "p:r:t:o:s:n:hv", long_options, &index);

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
      case 't':
        nthreads = atoi(optarg);
        break;
      case 'o':
        ops = atoi(optarg);
        break;
      case 's':
        size = atoi(optarg);
        break;
      case 'n':
        node_size = true;
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

  if (verbose) {
    std::cout << "Read percent " << read_percent << std::endl;
    std::cout << "Range " << range << std::endl;
    std::cout << "Population " << population << std::endl;
    std::cout << "Threads " << nthreads << std::endl;
    std::cout << "Operations per thread " << ops << std::endl;
    std::cout << "Size " << size << std::endl;
    std::cout << "Node size " << node_size << std::endl;
  }

  if (node_size == 1) {

    auto map = new rome::hds::unordered_map<int, 
                                            int, 
                                            1, 
                                            rome::hds::kv_linked_list::kv_lock_linked_list,
                                            rome::hds::kv_linked_list::locked_nodes::reg_cached_node_pointer, 
                                            rome::hds::allocator::heap_allocator>(size);

    test(map, read_percent, population, range, nthreads, ops);
  } else if (node_size == 2) {

    auto map = new rome::hds::unordered_map<int, 
                                            int, 
                                            2, 
                                            rome::hds::kv_linked_list::kv_lock_linked_list,
                                            rome::hds::kv_linked_list::locked_nodes::reg_cached_node_pointer, 
                                            rome::hds::allocator::heap_allocator>(size);

    test(map, read_percent, population, range, nthreads, ops);
  } else if (node_size == 4) {

    auto map = new rome::hds::unordered_map<int, 
                                            int, 
                                            4, 
                                            rome::hds::kv_linked_list::kv_lock_linked_list,
                                            rome::hds::kv_linked_list::locked_nodes::reg_cached_node_pointer, 
                                            rome::hds::allocator::heap_allocator>(size);

    test(map, read_percent, population, range, nthreads, ops);
  } else if (node_size == 8) {

    auto map = new rome::hds::unordered_map<int, 
                                            int, 
                                            8, 
                                            rome::hds::kv_linked_list::kv_lock_linked_list,
                                            rome::hds::kv_linked_list::locked_nodes::reg_cached_node_pointer, 
                                            rome::hds::allocator::heap_allocator>(size);

    test(map, read_percent, population, range, nthreads, ops);
  } else {
    std::cerr << "Unsupported node size" << std::endl;
    return 1;
  }

  return 0;
}

