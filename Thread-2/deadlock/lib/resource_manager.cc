#include <mutex>
#include <thread>
#include <chrono>
#include <condition_variable>
#include "resource_manager.h"

namespace proj2 {

int ResourceManager::request(RESOURCE r, int amount) {
	
	
	//Main Idea: Allocate all resources claimed in the budget at the first time of requesting
	
    if (amount <= 0)  return 1;
    
    auto this_id = std::this_thread::get_id();
    
    if (this->alloc[this_id]){
		return 0;
	}
	
	std::map<RESOURCE, int> my_claim = this->claim[this_id];
	std::map<RESOURCE, int>::iterator it,itt;
	
	while (true) {
        if (this->resource_cv[r].wait_for(
            lk, std::chrono::milliseconds(100),
            [this, r, amount] { return this->resource_amount[r] >= amount; }
        )) {
			bool tag = true;
			for (it = my_claim.begin(); it != my_claim.end(); it++){
				RESOURCE res = it->first;
				int res_amount = it->second;
				std::unique_lock<std::mutex> lk(this->resource_mutex[res]);
				if (this->resource_amount[r] < res_amount){
					tag = false;
					for (itt = my_claim.begin(); itt != it; itt++){
						RESOURCE ress = itt->first;
						int ress_amount = itt->second;
						std::unique_lock<std::mutex> lk(this->resource_mutex[ress]);
						this->resource_amount[ress] += ress_amount;
						this->resource_mutex[ress].unlock();
					}
					r = res;
					amount = res_amount;
					break;
				}
				this->resource_amount[res] -= res_amount;
				this->resource_mutex[res].unlock();
			}
			if (tag){
				this->alloc[this_id] = true;
				return 0;
			}
		}
	}
	
	
    return 0;
}

void ResourceManager::release(RESOURCE r, int amount) {
    if (amount <= 0)  return;
    std::unique_lock<std::mutex> lk(this->resource_mutex[r]);
    this->resource_amount[r] += amount;
    this->resource_cv[r].notify_all();
    this->resource_mutex[r].unlock();
}

void ResourceManager::budget_claim(std::map<RESOURCE, int> budget) {
    // This function is called when some workload starts.
    // The workload will eventually consume all resources it claims
    auto this_id = std::this_thread::get_id();
    this->claim[this_id] = budget;
    this->alloc[this_id] = false;
}

} // namespace: proj2
