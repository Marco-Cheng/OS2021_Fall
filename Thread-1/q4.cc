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
#define ppi std::pair<int,bool>
#define pi_ std::pair<proj1::Instruction, ppi >
#define pi std::pair<int, pii > 

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
