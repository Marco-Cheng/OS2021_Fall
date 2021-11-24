#ifndef BOAT_H_
#define BOAT_H_

#include<stdio.h>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <condition_variable>

#include "boatGrader.h"

namespace proj2{
class Boat{

public:
	Boat();
    ~Boat(){};
	void begin(int, int, BoatGrader*);

private:
	std::mutex mtx;
	std::unique_lock<std::mutex>* mtx_ptr;
	int adult_o, child_o, adult_m, child_m, boat;
	std::condition_variable adult_om, child_om, child_mo, finished;
	void action_child(BoatGrader*);
	void action_adult(BoatGrader*);
};
}

#endif // BOAT_H_
