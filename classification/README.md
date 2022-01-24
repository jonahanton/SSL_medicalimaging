23 Jan 2022
# Comments on fine tuning

To run mnist_classification.py, we need to load in encoder parameters from self-supervised learning. However, this is not uploaded in the repo. 
This is implemented in ds_linear_classifier.py (but not yet fully refactored)

## Details of Current Implementation 
- Basic pipeline: load in model parameters from encoder network of SSL, load in DataLoader, add in transformations to DataLoader, train model 
- During training, freeze all layers except FC layer 
  - Not the same implementation as in SimCLR but allows for faster training currently
- Reach high levels of accuracy very quickly - but may simply be due to MNIST task (especially when SSL was only trained on 1 epoch so far)

## How to test for code
- Check for contrastive loss at start of training and compare that to a purely supervised approach
- Verify code for top1 accuracy and top5 accuracy are correct - understand what they mean 

## Future changes to implementation
Try to resemble SimCLR paper more closely. 
- No more freezing of first few layers - check exact implementation 
- Don't throw away the entire projection head - check SimCLRv2 for implementation or possibly compare between the 2 approaches 
- Possible data transformations for supervised task 

## Refactoring to be done 
- Changes in script names
- Use arg parser to add in argument details like main.py instead of specifying them by hand 
  - Import the same arguments from main.py
- Automatically load in the current checkpoint 






