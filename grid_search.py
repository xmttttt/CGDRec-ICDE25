from tqdm import tqdm, trange
from model import *
from datasets import *
import logging
import os

from main import *


init_seed(2020)


if __name__ == '__main__':

    opt = generate_opt()
    
    # ['clothing','clothing','clothing','toys','toys','toys','gowalla','gowalla','gowalla']
    # for opt.atten_mode in ['sp','sp','sp','p','p','p']:
    #     main(opt)
    
    opt = generate_opt()
    for opt.dataset in ['beauty', 'yelp', 'toys', 'gowalla', 'clothing']:
        
        # opt.beta = 1
        # for opt.gamma in [0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]:
        #     main(opt)
        
        
        # for opt.lambda_diffcl in [1,1e-1,1e-2,1e-3,1e-4,1e-5,0]:

        main(opt)
    
    
    '''
    for opt.dataset in ['clothing', 'yelp']:

        opt.noise_steps = 1
        for opt.train_steps in [1,2,3,5,7,10,20,50,100]:
            main(opt)
            
        opt.train_steps = 21
        for opt.noise_steps in [0,1,2,3,4,5,7,10,15,20]:
            main(opt)
            
            
    opt = generate_opt()
    opt.dataset = 'toys'
    
    opt.gamma = 1
    for opt.beta in [0,1e-5,1e-4,1e-3,1e-2,0.1,1,10]:
        main(opt)
    '''

    # for opt.r in [0.5,0.6,0.7,0.8,0.9,1]:
    #     main(opt)
        
    # opt.noise_steps = 1
    
    # opt.dataset = 'toys'
    # for opt.middle_size in [1024,2048]:

    #     main(opt)

    # opt = generate_opt()

    # opt.dataset = 'gowalla'
    # for opt.middle_size in [64,128,256,512]:
        
    #     main(opt)

    # for opt.train_ratio in [0.01,0.02,0.05,0.1,0.2,0.5,0.8,1]:
    #     main(opt)
        
    # opt.train_ratio = 0.1
        
    # for opt.sample_epoch in [1,2,3,4,5,8,10]:
    #     main(opt)
    
    # for i in range(10):
    #     main(opt)
        
    # l = [0,1,2,3,4,5,6,7,10]
    
    # for i, opt.denoise_steps in enumerate(l[1:]):  
    #     # print(opt.train_steps, opt.denoise_steps)
    #     for opt.noise_steps in l[:i+2]:
    #         # print(opt.denoise_steps, opt.noise_steps)
    #         main(opt)
    
    # for opt.r in [0.7,0.8,0.9,1]:
    #     for i in range(2):
    #         main(opt)

    # for opt.gamma in [0,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,0.1,0.5,1,5,10]:

    #     main(opt)

        
    # for i, opt.train_steps in enumerate(l):  

    #     main(opt)
    
        
    # 
    
    # opt.noise_steps = 0
    
    # l = [1,2,3,4,5,7,10,15,20,50]
        
    # for i, opt.train_steps in enumerate(l):  
    #     if opt.train_steps == 1:
    #         continue
    #     for opt.denoise_steps in l[:i+1]:  
    #         # print(opt.train_steps, opt.denoise_steps)
    #         for opt.noise_steps in [0,1,2]:
    #             main(opt)
        
    
        


# 针对固定k测试, 大于10后效果都不好, 在1~50左右搜就行