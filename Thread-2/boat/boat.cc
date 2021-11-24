#include <thread>
#include <vector>
#include <windows.h>

#include "boat.h"

namespace proj2{
	
Boat::Boat(){
}

/* variable boat: 
   0 denotes nobody onboard on Oahu
   1 denotes one child onboard on Oahu
   -1 denotes nobody onboard on Molokai
*/

void Boat:: begin(int a, int b, BoatGrader *bg){
	#define pb push_back
    this->adult_o = this->child_o = this->adult_m = this->child_m = this->boat = 0;
    this->mtx_ptr = new std::unique_lock<std::mutex>(this->mtx);
    std::vector<std::thread> threads;
    for(int i = 0; i < a; i++){
        threads.pb(std::thread(&Boat::action_adult, this, bg));
    }
    for(int i = 0; i < b; i++){
        threads.pb(std::thread(&Boat::action_child, this, bg));
    }
    for (std::thread &each_thread: threads) {
        each_thread.detach();
    }
    this->finished.wait(*(this->mtx_ptr), [this, a, b]{ return this->adult_m == a && this->child_m == b; });
}


void Boat::action_adult(BoatGrader *bg){
    {
		std::unique_lock<std::mutex> lk(this->mtx);
        this->adult_o++;
        bg->initializeAdult();
        lk.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    {
        std::unique_lock<std::mutex> lk(this->mtx);
        while (!this->adult_om.wait_for(lk, std::chrono::milliseconds(100), [this]{ return this->boat == 0 && this->child_o <= 1; })){
        	lk.unlock();
        	lk.lock();
        }
        bg->AdultRowToMolokai();
        this->adult_o--;
		this->adult_m++;
		this->boat = -1;
        this->child_mo.notify_all();
    }
}

void Boat::action_child(BoatGrader *bg){
    {
        std::unique_lock<std::mutex> lk(this->mtx);
        this->child_o++;
        bg->initializeChild();
        lk.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    
    for(int cur=1;;cur^=1){
        if(cur){
            std::unique_lock<std::mutex> lk(this->mtx);
            while (!this->child_om.wait_for(lk, std::chrono::milliseconds(100), [this]{ return (this->boat == 0 && this->child_o >= 2) || this->boat == 1; })){
        		lk.unlock();
        		lk.lock();
        	}
			if(this->boat == 0){
                bg->ChildRowToMolokai();
                this->child_o--;
				this->child_m++;
				this->boat = 1;
                this->child_om.notify_all();
            }
            else{
                bg->ChildRideToMolokai();
                this->child_o--;
				this->child_m++;
				this->boat = -1;
				if (this->child_o == 0 && this->adult_o == 0){
					this->finished.notify_all();
				}
				else{
					this->child_mo.notify_all();
				}
            }
        }
        else{
            std::unique_lock<std::mutex> lk(this->mtx);
            while (!this->child_mo.wait_for(lk, std::chrono::milliseconds(100), [this]{ return this->boat == -1; })){
        		lk.unlock();
        		lk.lock();
        	}
        	if (!(this->child_o == 0 && this->adult_o == 0)){
        		bg->ChildRowToOahu();
            	this->child_o++;
				this->child_m--;
				this->boat = 0;
            	if (this->child_o <= 1){
                	this->adult_om.notify_all();
            	}
            	else{
                	this->child_om.notify_all();
            	}
        	}
        	else{
        		this->finished.notify_all();
        	}
        }
    }
}



}
