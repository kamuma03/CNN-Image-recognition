# Get datasets

import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

train_data = datasets.MNIST(root='MNIST-data',                        
                            transform=transforms.ToTensor(),          
                            train=True,                               
                            download=True                             
                           )

test_data = datasets.MNIST(root='MNIST-data',                        
                            transform=transforms.ToTensor(),          
                            train=False,                               
                            download=True                             
                           )

# Split data into train and validation

train_data, valid_data = torch.utils.data.random_split(train_data, [50000, 10000])

# Create dataloaders
batch_size = 200

train_loader = torch.utils.data.DataLoader( 
    train_data, 
    shuffle=True, 
    batch_size=batch_size
)

valid_loader = torch.utils.data.DataLoader(
    valid_data,
    shuffle=True,
    batch_size=batch_size
)

test_loader = torch.utils.data.DataLoader(
    test_data, 
    shuffle=True, 
    batch_size=batch_size
)
# create CNN

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5), # 1 x 28 x 28
            torch.nn.ReLU(),                        # 16 x 24 x 24
            torch.nn.Conv2d(16, 32, kernel_size=5), # 32 x 20 x 20
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2), # 20 x 12 x 12
            torch.nn.Flatten()
        )
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(32 * 20 * 20, 10)
        )

    def forward(self, x):
        x = self.cv_layers(x)
        x = self.fc_layer(x)
        x = torch.nn.functional.softmax(x)
        return x

cnn = CNN()

# Define optimizer function
learning_rate = 0.001

optimizer = torch.optim.Adam(
    params=cnn.parameters(),
    lr=learning_rate    
)
# Define loss function
criterion = torch.nn.CrossEntropyLoss()

writer = SummaryWriter(log_dir="runs")  

# Train the model
def train(model, epochs):
    losses = []
    for epoch in range(epochs):
        for idx, batch in enumerate(train_loader):
            inputs, labels = batch
            pred = model(inputs)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
            print("Epochs: ", epoch, "batch number: ", idx, "loss: ", loss)
            writer.add_scalar('Loss/Train', loss, epoch*len(train_loader) + idx) 
    
    return(losses) 

def accuracy(model, dataloader):
    num_correct = 0
    num_examples = len(test_data)                       # test DATA not test LOADER
    for inputs, labels in dataloader:                  # for all exampls, over all mini-batches in the test dataset
        predictions = model(inputs)
        predictions = torch.max(predictions, axis=1)    # reduce to find max indices along direction which column varies
        predictions = predictions[1]                    # torch.max returns (values, indices)
        num_correct += int(sum(predictions == labels))
    percent_correct = num_correct / num_examples * 100
    print('Accuracy:', percent_correct)


if __name__ == '__main__':
    # Piece our CNN flow here
    train(cnn, 1)
    torch.save(cnn, "trained_model.pt")
    cnn = torch.load("trained_model.pt")
    accuracy(cnn, valid_loader)
    accuracy(cnn, test_loader)