import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

from dataset import get_dataset
from utils import train_on_epoch, test_on_epoch, load_checkpoint, save_checkpoint
from torch.utils.data import DataLoader
import time

#Hyperparameters
NUM_CLASSES = 2
LR = 0.001
EPOCHS = 10
BATCH_SIZE = 32
NUM_WORKERS = 0
TEST_SIZE = 0.2

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

#Load model
model = torchvision.models.densenet161(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Linear(2208, NUM_CLASSES)
model.to(device)

#Load data
dataset = get_dataset(split_size=TEST_SIZE)
train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(dataset['test'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

#loss and optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LR)

#checkpoint

if os.path.exists("model_checkpoint.pth.tar"):
    last_epoch, last_loss = load_checkpoint(torch.load("model_checkpoint.pth.tar"), model, optimizer)
    print(f"Last epoch: {last_epoch}, Train Loss: {last_loss}")
else:
    last_epoch = 0
    last_loss = float("inf")

#Train Network
print("Training Model")
writer = SummaryWriter(
          f"runs/DOG_CAT_CLASSIFIER/MiniBatchSize {BATCH_SIZE} LR {LR}"
      )


for epoch in range(last_epoch + 1,EPOCHS):
    start = time.time()
    tr_acc, tr_loss = train_on_epoch(train_loader, model, device, optimizer, criterion)
    writer.add_scalar("Train loss", tr_loss, global_step=epoch)
    writer.add_scalar(
        "Train Accuracy", tr_acc, global_step=epoch
    )
    te_acc, te_loss = test_on_epoch(test_loader, model, device, criterion)
    writer.add_scalar("Test loss", te_loss, global_step=epoch)
    writer.add_scalar(
        "Test Accuracy", te_acc, global_step=epoch
    )

    end = time.time()
    print(f"Epoch: {epoch}, Train Loss: {tr_loss}, Train Acc: {tr_acc}, Test Loss: {te_loss}, Test Acc: {te_acc}, Time: {end-start} seconds")

    #save_checkpoint
    if te_loss < last_loss:
        last_loss = te_loss
        checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': tr_loss}
        save_checkpoint(checkpoint)
