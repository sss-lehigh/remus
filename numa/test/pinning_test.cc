#include <thread>
#include <iostream>

#include <rome/numa/numa.h>

int main() {

  rome::numa::Policy p(rome::numa::NEXT_CORE_IN_NODE, rome::numa::NEXT_CORE_ACROSS_NODE, rome::numa::NEXT_CORE_SIBLING);

  std::cerr << "Found " << p.num_cores() << " cores" << std::endl;

  std::vector<std::thread> threads;

  for(int i = 0; i < static_cast<int>(p.num_cores()); ++i) {
    threads.push_back(std::thread([&p](int id) {
      p.pin(id);
    }, i)); 
  }

  for(auto& t : threads) {
    t.join();
  }

  return 0;

}

