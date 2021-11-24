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
#include "lib/epoch.h"

#define pb push_back
#define mp std::make_pair
#define fi first
#define se second
#define pii std::pair<EmbeddingGradient*, bool>
#define pi_ std::pair<proj1::Instruction, ppi >
#define ppi std::pair<int,bool>
#define pi std::pair<int, pii > 

inline bool cmp(pi_ a, pi_ b){
	return a.se.fi<b.se.fi;
}

int main(int argc, char *argv[]) {
 

    proj1::EmbeddingHolder* users = new proj1::EmbeddingHolder("data/q3.in");
    proj1::EmbeddingHolder* items = new proj1::EmbeddingHolder("data/q3.in");
    proj1::Instructions instructions = proj1::read_instructrions("data/q3_instruction.tsv");
    {
    proj1::AutoTimer timer("q3");  // using this to print out timing of the block
    
    // parallel processing 
    
    std::mutex mtx;
    
    std::vector <pi_ > insts;
    
	mtx.lock(); 
    for (proj1::Instruction ins: instructions) {
    	int epoch = -1;
        if (ins.order == proj1::UPDATE_EMB && ins.payloads.size() > 3) {
            epoch = ins.payloads[3];
        }
        insts.pb(mp(ins,mp(epoch,false)));
    }
    mtx.unlock();
    
    sort(insts.begin(),insts.end(),cmp);
    
    int sz = insts.size(), cur_id = 0;
    
    while (cur_id<sz){
    	int lst_id = cur_id;
    	while (lst_id+1<sz && insts[lst_id+1].se.fi==insts[cur_id].se.fi) lst_id++;
    	for (int i=cur_id;i<=lst_id;i++){
    		std::thread(proj1::my_instruction, std::ref(insts[i]), users, items, std::ref(mtx)).join();
    	}
    	bool ready = 0;
        while(!ready) {
			ready = 1;
			for (int i=cur_id;i<=lst_id;i++){
				if (!insts[i].se.se) {
					ready = 0;
					break;
				}
			}	
		}
		cur_id = lst_id+1;
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
