# Things we have implemented 

1. Running of BYOL code 

### Things we have tested and verified 
1. Moving average (at least for tau = 0) 
2. MNIST dataset for 3 epochs - loss goes down but slowly 


### Things to work on 
1. ~Weight decay/regularisation~
    - Already implemented in both BYOL(10-6 in original paper) and SimCLR 
2. Testing of code (Mathilde and Venus)
3. Extend to RGB datasets (e.g. CIFAR 10) (Mun fai) 
  - Color augmentations for data transformations - JONAH: I think this is inplace.  
4. Closer resemblance to paper 
  - ~LARS optimiser~ suffering from installation issues; may not be very important as it is just a tool to help with large batch sizes  
    
    
    
