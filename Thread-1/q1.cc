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


namespace proj1 {

void run_one_instruction(Instruction inst, EmbeddingHolder* users, EmbeddingHolder* items, std::mutex& mtx) {
    switch(inst.order) {
        case INIT_EMB: {
            // We need to init the embedding
            int length = users->get_emb_length();
            Embedding* new_user = new Embedding(length);
            mtx.lock();
            int user_idx = users->append(new_user);
            mtx.unlock();
            users->locks[user_idx]->lock();
            for (int item_index: inst.payloads) {
                Embedding* item_emb = items->get_embedding(item_index);
                // Call cold start for downstream applications, slow
                EmbeddingGradient* gradient = cold_start(new_user, item_emb);
                users->update_embedding(user_idx, gradient, 0.01);
                delete gradient;
            }
            users->locks[user_idx]->unlock();
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
 

    proj1::EmbeddingHolder* users = new proj1::EmbeddingHolder("data/q1.in");
    proj1::EmbeddingHolder* items = new proj1::EmbeddingHolder("data/q1.in");
    proj1::Instructions instructions = proj1::read_instructrions("data/q1_instruction.tsv");
    {
    proj1::AutoTimer timer("q1");  // using this to print out timing of the block
    
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
