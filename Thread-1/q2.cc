#include <vector>
#include <tuple>

#include <string>   // string
#include <chrono>   // timer
#include <iostream> // cout, endl

#include <mutex>
#include <thread>

#include "lib/utils.h"
#include "lib/model.h" 
#include "lib/embedding.h" 
#include "lib/instruction.h"

#define pb push_back
#define mp std::make_pair
#define fi first
#define se second
#define pii std::pair<EmbeddingGradient*, bool>
#define pi std::pair<int, pii > 


namespace proj1 {

void my_cold_start(pii &grad, Embedding* newUser, Embedding* item) {
	grad.fi = cold_start(newUser, item);//store the results
	grad.se = 1;//set the finished flag to be 1
}

void run_one_instruction(Instruction inst, EmbeddingHolder* users, EmbeddingHolder* items, std::mutex& mtx) {
    switch(inst.order) {
        case INIT_EMB: {
        	
            // We need to init the embedding
            int length = users->get_emb_length();
            Embedding* new_user = new Embedding(length);
            mtx.lock();
            int user_idx = users->append(new_user);
            mtx.unlock();
            
            // parallelized processing cold_start
            std::vector <std::thread> item_threads;
            std::vector <pi > records;
            
            mtx.lock();
            for (int item_index: inst.payloads) {
            	records.pb(mp(item_index, mp(nullptr, false)));
			}
			mtx.unlock();
			
			int sz = records.size();
			
            for (int i=0; i<sz; i++) {
            	int item_index = records[i].fi;
                Embedding* item_emb = items->get_embedding(item_index);
                std::thread(my_cold_start, std::ref(records[i].se), new_user, item_emb).join();
            }
            
            // collective update after cold_start
            bool ready = 0;
            
            while(!ready) {
            	
				ready = 1;
				
				for (int i=0; i<sz; i++) {
					if (!records[i].se.se) {
						ready = 0;
						break;
					}
				}
				
				if (ready) {
					users->locks[user_idx]->lock();
					for (int i=0; i<sz; i++) {
						users->update_embedding(user_idx, records[i].se.fi, 0.01);
					}
					users->locks[user_idx]->unlock();
					break;
				}
				
			}
            
            break;
        }
        case UPDATE_EMB: {
            int user_idx = inst.payloads[0];
            int item_idx = inst.payloads[1];
            int label = inst.payloads[2];
            // You might need to add this state in other questions.
            // Here we just show you this as an example
            // int epoch = -1;
            //if (inst.payloads.size() > 3) {
            //    epoch = inst.payloads[3];
            //}
            Embedding* user = users->get_embedding(user_idx);
            Embedding* item = items->get_embedding(item_idx);
            EmbeddingGradient* gradient = calc_gradient(user, item, label);
            users->locks[user_idx]->lock();
            users->update_embedding(user_idx, gradient, 0.01);
            users->locks[user_idx]->unlock();
            delete gradient;
            gradient = calc_gradient(item, user, label);
            items->locks[item_idx]->lock();
            items->update_embedding(item_idx, gradient, 0.001);
            items->locks[item_idx]->unlock();
            delete gradient;
            break;
        }
    }

}
} // namespace proj1

    

int main(int argc, char *argv[]) {
 

    proj1::EmbeddingHolder* users = new proj1::EmbeddingHolder("data/q2.in");
    proj1::EmbeddingHolder* items = new proj1::EmbeddingHolder("data/q2.in");
    proj1::Instructions instructions = proj1::read_instructrions("data/q2_instruction.tsv");
    {
    proj1::AutoTimer timer("q2");  // using this to print out timing of the block
    
    // parallel processing 
    
    std::mutex mtx;
    
    for (proj1::Instruction ins: instructions) {
        std::thread(proj1::run_one_instruction, ins, users, items, std::ref(mtx)).join();
        //each instruction is processed as a thread
    }
    
    }
    

    // Write the result
    users->write_to_stdout();
    items->write_to_stdout();

    // We only need to delete the embedding holders, as the pointers are all
    // pointing at the emb_matx of the holders.
    delete users;
    delete items;

    return 0;
}
