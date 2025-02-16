# Project 1 : README FOR TAS

**Author: Zerui Cheng (2019012355) , Shiyu Zhao (2019012358)**



In our implementation of Project 1, the ideas for each question are shown as follows.

For **q1**, each update (i.e. write) operation should have the modified session locked in order that no other threads can read or write in this place, so that all threads will be safety. Moreover, to enable better parallelism, for read operations, they don't have to acquire locks, so that multiple reading operations at the same place can be conducted at the same time.

For **q2**, to optimize the performance of cold_start, we can use multiple threads to accelerate the process. To be precise, we assign a thread for each iteration of cold_start and update the embeddings collectively after running all cold_starts. Similarly as above, when encountering with writing operations such as modification and update, locks should be carefully assigned for thread safety.

For **q3**, we scan the instructions in ascending order where the initialization operation is regarded as of epoch -1, and then run parallel updates of the same epoch at the same time using multiple threads. On top of that, for each epoch, we should guarantee that all threads have already terminated before stepping into the next epoch, and thus we need a flag to show status of threads.

For **q4**, we classify all operations into three kinds, and sort updates and recommendations according to their epoch. First, we carry out all initialization operations in a parallel manner. Later, we run updates according to the ascending order of their corresponding epochs. For updates in the  same epoch, we parallelize their execution using multiple threads. After all such threads terminate for each epoch, we scan the un-executed recommendations operations and run all recommendations that can be executed now using multiple threads in a parallel manner. In this way, with the guarantee of thread safety, we can enjoy plausible efficiency in getting recommendations, and the accuracy of the results can be also guaranteed by carefully added locks.



Afterwards, we make a serious attempt into testing our codes.

For efficiency, we generate large data including $10^5$ instructions (see gen_data.py as the generator and test_data.txt in data folder as the generated instructions, where the embedding remains the same as initial ones distributed by TAs) . Compared with linear version (q0.cc), we can see that the utility of CPU can achieve rather high rate and the efficiency gets significantly improved.

Then, for accuracy, we also generate a couple small data and evaluate all possible outputs to test the accuracy of the final outputs, which can be seen in small_data_x.txt (x is a positive number) as the instructions and the embedding also remains identical to initial ones , and we can see that our output is always correct, which shows the thread safety of our program. Especially, by comparing results of the same input on q0.cc and q4.cc, we find that our results are reasonable and meets all requirements specified in the initial README file.

For instance, with the same embedding (take q1.in as an example), running small_data_1.txt as the instructions for q0.cc and q4.cc results in identical outputs. For small_data_5.txt, although the outputs are different, the set of results are identical and the difference lies in difference sequence of recommendation processing. As for small_data_4.txt, the results are different, however, that's because q0.cc doesn't take epoch into consideration and run all instructions linearly. After rearranging according to the epochs, we can find out that our results generated by q4.cc is correct. And all these tests show the accuracy and robustness of our program. On top of that, for each part of our code, we also carefully implement some unit tests to guarantee the correctness.



In conclusion, through carefully designed lock-adding mechanism and rigorous test, we guarantee the security of the program. Also, as instructed by TAs, to avoid duplicate code, in q3.cc and q4.cc, we remove the identical part of code into epoch.cc and can be called through including epoch.h (epoch.cc and epoch.h ca be found in folder lib) and then q3.cc and q4.cc looks much neater than before, yielding neat coding style and high readability of implementation.

