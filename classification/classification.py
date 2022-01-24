import torch
from torchvision import datasets, transforms
from models.conv_net import ConvNet
from torch.utils.data import DataLoader
from utils import accuracy
from tqdm import tqdm

# Define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Using device:", device)

### Load in neural net
model = ConvNet()

### Load in data
output_dim = 10
data_transform = transforms.Compose([
    # can add other data augmentation techniques
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # same parameters as self-supervised
])

train_dataset = datasets.MNIST("./datasets", train=True, transform= data_transform, download=False)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataset = datasets.MNIST("./datasets", train=False, transform = data_transform, download=False)
test_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

### Training
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
criterion = torch.nn.CrossEntropyLoss().to(device)

def train_classification(epochs):
    for epoch in range(epochs):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(tqdm(train_loader)):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= (counter + 1)

        top1_accuracy = 0
        top5_accuracy = 0

        # Test data
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

if __name__ == "__main__":
    epochs = 5
    train_classification(epochs)