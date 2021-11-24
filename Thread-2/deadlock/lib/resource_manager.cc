#include <mutex>
#include <thread>
#include <chrono>
#include <iostream>
#include <condition_variable>
#include "resource_manager.h"

namespace proj2 {

int ResourceManager::request(RESOURCE r, int amount) {
    
	if (amount <= 0)  return 1;

    auto this_id = std::this_thread::get_id();
    
    while (true) {
    	
    	std::unique_lock<std::mutex> lk(this->resource_mutex[r]);
		if (this->resource_cv.wait_for(
            lk, std::chrono::milliseconds(100),
            [this, r, amount] { return this->resource_amount[r] >= amount; }
        )){
        	this->resource_mutex[r].lock();
    		bool tag = true;
    		this->resource_amount[r] -= amount;
    		this->alloc[this_id][r] += amount;
    		std::map<RESOURCE, int> work = this->resource_amount;
    		std::map<RESOURCE, int>::iterator it;
    		std::vector <bool> finished;
    		int sz = this->num, tot = 0;
    		finished.resize(sz);
    		for (int i = 0; i < sz; i++){
    			finished[i] = false;
    		}
    		while (tot < sz){
    			int lst = tot;
    			for (int i = 0; i < sz; i++){
    				if (finished[i]) continue;
    				bool tag = true;
    				for (it = this->claim[active[i]].begin(); it != this->claim[active[i]].end(); it++){
    					int need = it->second - alloc[active[i]][it->first];
    					if (need > work[it->first]){
    						tag = false;
    						break;
    					}
    				}
    				if (tag){
    					tot++;
    					finished[i] = true;
    					for (it = this->claim[active[i]].begin(); it != this->claim[active[i]].end(); it++){
    						work[it->first] += alloc[active[i]][it->first];
    					}
    				}
    			}
    			if (lst == tot) break;
    		}
    		if (tot < sz){
    			tag = false;
    			this->resource_amount[r] += amount;
    			this->alloc[this_id][r] -= amount;
    		}
			this->resource_mutex[r].unlock();
			if (tag) return 0;	
    	}
    	//std::cout<<this_id<<std::endl;
    }
    	
    return 0;
}

void ResourceManager::release(RESOURCE r, int amount) {
    if (amount <= 0)  return;
    std::unique_lock<std::mutex> lk(this->resource_mutex[r]);
    this->resource_amount[r] += amount;
    this->resource_cv.notify_all();
}

void ResourceManager::budget_claim(std::map<RESOURCE, int> budget) {
    // This function is called when some workload starts.
    // The workload will eventually consume all resources it claims
    auto this_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(this->mtx);
    if (this->active.size() == 0){
    	this->num = 0;
    }
    this->active.push_back(this_id);
    this->num++;
	this->claim[this_id] = budget;
	std::map<RESOURCE, int>::iterator it;
	std::map<RESOURCE, int> cur_alloc;
	for (it = budget.begin(); it != budget.end(); it++){
		cur_alloc[it->first] = 0;
	}
	this->alloc[this_id] = cur_alloc;
}

} // namespace: proj2
