#ifndef DEADLOCK_LIB_RESOURCE_MANAGER_H_
#define DEADLOCK_LIB_RESOURCE_MANAGER_H_

#include <map>
#include <vector> 
#include <mutex>
#include <thread>
#include <condition_variable>
#include "thread_manager.h"

namespace proj2 {

enum RESOURCE {
    GPU = 0,
    MEMORY,
    DISK,
    NETWORK
};

class ResourceManager {
public:
    ResourceManager(ThreadManager *t, std::map<RESOURCE, int> init_count): \
        resource_amount(init_count), tmgr(t) {}
    void budget_claim(std::map<RESOURCE, int> budget);
    int request(RESOURCE, int amount);
    void release(RESOURCE, int amount);
private:
    std::map<RESOURCE, int> resource_amount;
    std::map<RESOURCE, std::mutex> resource_mutex;
    std::condition_variable resource_cv;
    std::map<std::thread::id, std::map<RESOURCE, int> > claim;
    std::map<std::thread::id, std::map<RESOURCE, int> > alloc;
    std::vector<std::thread::id> active;
    int num;
    ThreadManager *tmgr;
};

}  // namespce: proj2

#endif
