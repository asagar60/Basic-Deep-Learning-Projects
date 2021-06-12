import torch

def train_on_epoch(train_loader, model, device, optimizer, criterion):
    losses = []
    num_correct = 0
    num_samples = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device = device)
        targets = targets.to(device = device)

        #forward
        scores = model(data)   
        loss = criterion(scores, targets)
        losses.append(loss.item())

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predictions = scores.max(1)
        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)

    acc = float(num_correct)/float(num_samples)
    loss = sum(losses)/len(losses)
    
    return acc, loss

def test_on_epoch(test_loader, model, device, criterion):
    losses = []
    num_correct = 0
    num_samples = 0

    model.eval()
    with torch.no_grad(): 
        for x,y in test_loader:
            x = x.to(device = device)
            y = y.to(device = device)

            scores = model(x)
            loss = criterion(scores, y)
            losses.append(loss.item())

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    acc = float(num_correct)/float(num_samples)
    loss = sum(losses)/len(losses)
    model.train()

    return acc, loss

    
#Check acc on training and test

def check_accuracy(loader, model, device):
    if loader.dataset.train:
        print("Checking acc of train data")
    else:
        print("Checking acc on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad(): 
        for x,y in loader:
            x = x.to(device = device)
            y = y.to(device = device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        acc = float(num_correct)/float(num_samples)
        print(f"Got:  {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) * 100:.2f}")
    model.train()
    return acc

#Checkpoint
def save_checkpoint(state, filename = "model_checkpoint.pth.tar"):
    torch.save(state, filename)
    print("Checkpoint Saved at ",filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    last_epoch = checkpoint['epoch']
    last_loss = checkpoint['loss']
    return last_epoch, last_loss