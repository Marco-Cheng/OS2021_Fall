#include<mutex>
#include<condition_variable>

namespace read_writer{
    class readerwriter{
        private:
            int ar = 0;
            int wr = 0;
            int aw = 0;
            int ww = 0;
            std::condition_variable oktoread;
            std::condition_variable oktowrite;
            std::mutex mtx;
        
        public:
            void readerlock(){
                std::unique_lock<std::mutex> l(mtx);
                while((aw + ww) > 0){
                    wr++;
                    oktoread.wait(l);
                    wr--;
                }
                ar++;
            }
            void readerrelease(){
                std::unique_lock<std::mutex> l(mtx);
                ar--;
                if(ar == 0 && ww > 0){
                    oktowrite.notify_one();
                }
            }
            void writerlock(){
                std::unique_lock<std::mutex> l(mtx);
                while((aw + ar) > 0){
                    ww++;
                    oktowrite.wait(l);
                    ww--;
                }
                aw++;
            }
            void writerrelease(){
                std::unique_lock<std::mutex> l(mtx);
                aw--;
                if(ww > 0){
                    oktowrite.notify_one();
                }
                else if(wr > 0){
                    oktoread.notify_all();
                }
            }
    };
}