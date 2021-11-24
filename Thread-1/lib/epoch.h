#ifndef THREAD_LIB_EPOCH_H_
#define THREAD_LIB_EPOCH_H_

#include <vector>
#include <tuple>

#include <string>   // string
#include <chrono>   // timer
#include <iostream> // cout, endl

#include <mutex>
#include <thread>

#include "embedding.h"
#include "instruction.h"
#include "utils.h"
#include "model.h"

#define pb push_back
#define mp std::make_pair
#define fi first
#define se second
#define pii std::pair<EmbeddingGradient*, bool>
#define ppi std::pair<int,bool>
#define pi_ std::pair<proj1::Instruction, ppi >
#define pi std::pair<int, pii > 

namespace proj1 {

void my_cold_start(pii &grad, Embedding* newUser, Embedding* item);
void run_one_instruction(Instruction inst, EmbeddingHolder* users, EmbeddingHolder* items, std::mutex& mtx);
void my_instruction(pi_ &sta, EmbeddingHolder* users, EmbeddingHolder* items, std::mutex& mtx);

} // namespace proj1
#endif 
