def accuracy(output, target):
    """ Computed the accuracy for a batch of output data"""
    num_correct = 0
    num_samples = 0

    _, preds = output.max(1)
    num_correct += (preds == target).sum()
    num_samples += preds.size(0)
    
    acc = float(num_correct) / num_samples
    
    return acc



if __name__ == "__main__":
    pass