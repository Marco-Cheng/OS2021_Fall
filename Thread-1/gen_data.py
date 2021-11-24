import numpy as np
import pickle

def gen_init(num_usr, num_item, it_size):
    usr = np.random.randint(num_usr)
    #it_size = np.random.randint(num_item)
    item = np.random.choice(num_item, it_size, replace=False)
    l = np.hstack([usr, item])
    s = '0'
    for i in l:
        s += ' '
        s += str(i)
    return s

def gen_update(num_usr, num_item, max_ep):
    usr = np.random.randint(num_usr)
    item = np.random.randint(num_item)
    epo = np.random.randint(-1, max_ep)
    if epo == -1:
        l = np.hstack([usr, item])
        s = '1'
        for i in l:
            s += ' '
            s += str(i)
        return s
    else:
        l = np.hstack([usr, item, epo])
        s = '1'
        for i in l:
            s += ' '
            s += str(i)
        return s

def gen_recom(num_usr, num_item, max_ep, it_size):
    usr = np.random.randint(num_usr)
    #it_size = np.random.randint(num_item)
    item = np.random.choice(num_item, it_size, replace=False)
    epo = np.random.randint(-1, max_ep)
    l = np.hstack([usr, epo, item])
    s = '2'
    for i in l:
        s += ' '
        s += str(i)
    return s  

print('start!')
instruction_otp = '/Users/thinkpad/Desktop/Thread-1/data/test_data.txt'
infile = open(instruction_otp, 'w')
num_usr = 10
num_item = 10
max_ep = 20
num_lines = 100000
for i in range(num_lines):
    r = np.random.rand()
    if r < 0.5:
        li = gen_init(num_usr, num_item, 10)
    elif 0.33 < r <= 0.66:
        li = gen_update(num_usr, num_item, max_ep)
    else:
        li = gen_recom(num_usr, num_item, max_ep, 10)
    infile.write(li + '\n')
infile.close()
print('finished!')
