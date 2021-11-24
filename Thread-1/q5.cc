#include "lib/utils.h"
#include "lib/model.h" 
#include "lib/embedding.h" 
#include "lib/instruction.h"
#include "readerwriter.h"

#include <vector>
#include <tuple>
#include <map>

#include <string>   // string
#include <chrono>   // timer
#include <iostream> // cout, endl

#include <mutex>
#include <thread>

#define pb push_back
#define mp std::make_pair
#define fi first
#define se second
#define pii std::pair<EmbeddingGradient*, bool>
#define ppi std::pair<int,bool>
#define pi_ std::pair<proj1::Instruction, ppi >
#define pi std::pair<int, pii > 




namespace proj1 {

std::map<int, read_writer::readerwriter> user_map;
std::map<int, read_writer::readerwriter> item_map;

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
            std::mutex map_lock1;
            map_lock1.lock();
            read_writer::readerwriter &user_lock = user_map[user_idx];
            map_lock1.unlock();

            std::mutex map_lock2;
            map_lock2.lock();
            read_writer::readerwriter &item_lock = item_map[item_idx];
            map_lock2.unlock();

            Embedding* user = users->get_embedding(user_idx);
            Embedding* item = items->get_embedding(item_idx);

            // reader's locks, copy user gradient
            user_lock.readerlock();
            item_lock.readerlock();
            EmbeddingGradient* gradient = calc_gradient(user, item, label);
            int l = gradient->get_length();
            double *copy_grad = new double[l];
            for (int j = 0; j < l; j++){
                copy_grad[j] = gradient->get_data()[j];
            }
            EmbeddingGradient* copied_grad_user = new EmbeddingGradient(l, copy_grad);
            user_lock.readerrelease();
            item_lock.readerrelease();

            // user writer's lock
            user_lock.writerlock();
            users->update_embedding(user_idx, copied_grad_user, 0.01);
            delete gradient;
            delete copied_grad_user;
            user_lock.writerrelease();

            //reader's locks, copy item gradient
            user_lock.readerlock();
            item_lock.readerlock();
            EmbeddingGradient* gradient2 = calc_gradient(item, user, label);
            int l2 = gradient2->get_length();
            double *copy_grad2 = new double[l2];
            for (int j = 0; j < l2; j++){
                copy_grad2[j] = gradient2->get_data()[j];
            }
            EmbeddingGradient* copied_grad_item = new EmbeddingGradient(l2, copy_grad2);
            user_lock.readerrelease();
            item_lock.readerrelease();

            // item writer's lock
            item_lock.writerlock();
            gradient = calc_gradient(item, user, label);
            items->update_embedding(item_idx, copied_grad_item, 0.001);
            delete gradient;
            delete copied_grad_item;
            item_lock.writerrelease();

            break;
        }
        
        case RECOMMEND: {
            int user_idx = inst.payloads[0];
            Embedding* user = users->get_embedding(user_idx);
            
			std::vector<Embedding*> item_pool;
            
            for (unsigned int i = 2; i < inst.payloads.size(); ++i) {
                int item_idx = inst.payloads[i];

                std::mutex map_lock;
                map_lock.lock();
                read_writer::readerwriter &item_lock = item_map[item_idx];
                map_lock.unlock();

                // reader's lock
                item_lock.readerlock();
                Embedding *item_emb = items->get_embedding(item_idx);
                int l = item_emb->get_length();
                double *copy_emb = new double[l];
                for (int j=0; j<l; j++){
                    copy_emb[j] = item_emb->get_data()[j];
                }
                Embedding *copied_emb = new Embedding(l, copy_emb);
                // reader's lock released
                item_lock.readerrelease();
            
                item_pool.push_back(copied_emb);
            }
            
            Embedding* recommendation = recommend(user, item_pool);
            
            mtx.lock(); 
            recommendation->write_to_stdout();
            mtx.unlock();
            
            break;
        }
        
    }

}

void my_instruction(pi_ &sta, EmbeddingHolder* users, EmbeddingHolder* items, std::mutex& mtx) {
	run_one_instruction(sta.fi, users, items, mtx);//store the results
	sta.se.se = true;//set the finished flag to be 1
}

} // namespace proj1

inline bool cmp(pi_ a, pi_ b){
	return a.se.fi<b.se.fi;
}

int main(int argc, char *argv[]) {
 

    proj1::EmbeddingHolder* users = new proj1::EmbeddingHolder("data/q4.in");
    proj1::EmbeddingHolder* items = new proj1::EmbeddingHolder("data/q4.in");
    proj1::Instructions instructions = proj1::read_instructrions("data/q4_instruction.tsv");
    {
    proj1::AutoTimer timer("q4");  // using this to print out timing of the block
    
    // parallel processing 
    
    std::mutex mtx;
    
    std::vector <pi_ > ini;
    std::vector <pi_ > upd;
    std::vector <pi_ > rec;
    
	mtx.lock(); 
	
    for (proj1::Instruction ins: instructions) {
    	int epoch = -2;
        if (ins.order == proj1::UPDATE_EMB){
			if (ins.payloads.size() > 3) {
				epoch = ins.payloads[3]-1;
			}
			upd.pb(mp(ins,mp(epoch,false)));
			continue;
        }
        
        if (ins.order == proj1::RECOMMEND) {
            epoch = ins.payloads[1]-1;
            if (epoch==-2) epoch=-3;
            rec.pb(mp(ins,mp(epoch,false)));
            continue;
        }
        
        ini.pb(mp(ins,mp(epoch,false)));
    }
    
    mtx.unlock();
    
    sort(upd.begin(),upd.end(),cmp);
    sort(rec.begin(),rec.end(),cmp);
	
	int sz = ini.size();
	
	for (int i=0;i<sz;i++){
    	std::thread(proj1::my_instruction, std::ref(ini[i]), users, items, std::ref(mtx)).join();
    }
    
    int sz1 = upd.size(), p1 = 0;
    int sz2 = rec.size(), p2 = 0;
    
    int timestamp = (sz1>0 ? upd[0].se.fi: 2147483647);
    
    while (p2<sz2&&rec[p2].se.fi<timestamp){ 
    	std::thread(proj1::my_instruction, std::ref(rec[p2]), users, items, std::ref(mtx)).join();
    	p2++;
    }
    
    while (p1<sz1){
    	int nxt=p1;
    	while (nxt+1<sz1&&upd[nxt+1].se.fi==upd[p1].se.fi) nxt++;
    	for (int i=p1;i<=nxt;i++){
    		std::thread(proj1::my_instruction, std::ref(upd[i]), users, items, std::ref(mtx)).join();
    	}
    	bool ready = 0;
        while(!ready) {
			ready = 1;
			for (int i=p1;i<=nxt;i++){
				if (!upd[i].se.se) {
					ready = 0;
					break;
				}
			}	
		}
		p1=nxt+1;
		timestamp = (p1<sz1?upd[p1].se.fi: 2147483647);
		
		while (p2<sz2&&rec[p2].se.fi<timestamp){
    		std::thread(proj1::my_instruction, std::ref(rec[p2]), users, items, std::ref(mtx)).join();
    		p2++;
    	}
	}
	
	}
    
    // Write the result
    //users->write_to_stdout();
    //items->write_to_stdout();

    // We only need to delete the embedding holders, as the pointers are all
    // pointing at the emb_matx of the holders.
    delete users;
    delete items;

    return 0;
}