import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import logging
import os

class SimCLRTrainer:
    """
    
    This class implements the SimCLR pretraining method.
    
    """

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.writer = SummaryWriter()

        # logging.basicConfig(level=logging.DEBUG)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)


    def info_nce_loss(self, features, temperature):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / temperature
        return logits, labels

    
    def train(self, train_loader):

        n_iterations = 0
        logging.info(f"Starting SimCLR training for {self.args.epochs} epochs.")

        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch

            for images, _ in tqdm(train_loader):
                
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                # forward pass
                features = self.model(images)
                logits, labels = self.info_nce_loss(features, self.args.temperature)
                loss = self.criterion(logits, labels)

                print(f"Loss: {loss}")

                # backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                n_iterations += 1
                running_loss += loss.item()

            # Scheduler for optimiser - e.g. cosine annealing
            if epoch >= 10:
                self.scheduler.step() 

            training_loss = running_loss/len(train_loader)
            print("Train Loss:", training_loss)
            logging.info(f"Epoch: {epoch}\tLoss: {training_loss}")

        logging.info("Finished training.")       

        # Save model
        checkpoint_name = f'ssl_{self.args.dataset_name}_trained_model.pth.tar'
        checkpoint_filepath = os.path.join(self.args.outpath, checkpoint_name)
        torch.save( 
                {
                'epoch': self.args.epochs,
                'arch': self.args.arch, 
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
                }, checkpoint_filepath)

        logging.info(f"Model has been saved in directory {self.args.outpath}.")

