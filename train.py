#%%
from dataset import ImagesDataset
from classifier import MinimalResNet, FeatureExtractor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import random_split
from torchvision import transforms
from torch.optim import lr_scheduler
torch.manual_seed(20)

def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    lr=0.0001,
    epochs=12,
    optimiser=torch.optim.SGD
):
    """
    Trains a neural network on a dataset and returns the trained model

    Parameters:
    - model: a pytorch model
    - train_loader, val_loader, test_loader - pytorch dataloaders for training, validation, and test sets
    - lr - master learning rate (default 0,0001)
    - eochs - number of epochs (default 50)
    - optimiser - optimiser used for updating weights, (default: Stochastic Gradient Descent, SGD)
  

    Returns:
    - model: a trained pytorch model
    """

    writer = SummaryWriter()

    # initialise optimiser, learning rate scheduler, iteration variables
    optimiser = optimiser(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = lr_scheduler.MultiStepLR(optimiser, milestones=[4,9], gamma=0.1,verbose=True)
    
    # weights_fn='model_evaluation/TransferLearning_34_unfreeze2023-02-23-06:27:02/saved_weights/_4_latest_weights.pt'
    # state_dict=torch.load(weights_fn)
    # model.load_state_dict(state_dict)
    batch_idx = 0
    epoch_idx= 0

    for epoch in range(epochs):  # for each epoch
        print('Epoch:', epoch_idx,'LR:', scheduler.get_lr())
        weights_filename=model.weights_folder_name + '_' + str(epoch) + '_latest_weights.pt'
        epoch_idx +=1
        torch.save(model.state_dict(), weights_filename)
        for batch in train_loader:  
            model.train()
            features, labels = batch
            prediction = model(features)  
            loss = F.cross_entropy(prediction, labels)
            loss.backward()
            optimiser.step()
            print("Epoch:", epoch, "Batch:", batch_idx,
                  "Loss:", loss.item())  # log the loss
            optimiser.zero_grad()  # zero grad
            writer.add_scalar("Loss/Train", loss.item(), batch_idx)
            batch_idx += 1

            
        print('Evaluating on valiudation set')
        # evaluate the validation set performance
        val_loss, val_acc = evaluate(model, val_loader)
        writer.add_scalar("Loss/Val", val_loss, batch_idx)
        writer.add_scalar("Accuracy/Val", val_acc, batch_idx)
        scheduler.step()
    
    
    print('Evaluating on test set')
    test_loss = evaluate(model, test_loader)
    # writer.add_scalar("Loss/Test", test_loss, batch_idx)
    model.test_loss = test_loss
    final_model_fn='test_final_model.pt'
    torch.save(model.state_dict(), final_model_fn)
    return model   # return trained model
    

def evaluate(model, dataloader):
    model.eval()
    losses = []
    correct = 0
    n_examples = 0
    for batch in dataloader:
        features, labels = batch
        with torch.no_grad():
            prediction = model(features)
            
        loss = F.cross_entropy(prediction, labels)
        losses.append(loss.detach())
        correct += torch.sum(torch.argmax(prediction, dim=1) == labels)
        n_examples += len(labels)
    avg_loss = np.mean(losses)
    accuracy = correct / n_examples
    print("Loss:", avg_loss, "Accuracy:", accuracy.detach().numpy())
    return avg_loss, accuracy

def test_final_model(model,test_loader,path_to_final_state_dict):
    optimiser = optimiser(model.parameters(), lr=lr, weight_decay=0.001)
    state_dict=torch.load( path_to_final_state_dict )
    model.load_state_dict(state_dict)
    print('Evaluating on test set')
    test_loss = evaluate(model, test_loader)
    # writer.add_scalar("Loss/Test", test_loss, batch_idx)
    model.test_loss = test_loss
    return test_loss

def split_dataset(dataset):
    train_set_len = round(0.7*len(dataset))
    val_set_len = round(0.15*len(dataset))
    test_set_len = len(dataset) - val_set_len - train_set_len
    split_lengths = [train_set_len, val_set_len, test_set_len]
    train_set, val_set, test_set = random_split(dataset, split_lengths)
    return train_set,val_set,test_set

if __name__ == "__main__":

    size = 224
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop((size,size), pad_if_needed=True),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomHorizontalFlip(p=0.25),
        # transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    dataset = ImagesDataset(transform=transform)
    
    train_set,val_set,test_set=split_dataset(dataset)
    batch_size = 64
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    # nn = NeuralNetworkClassifier()
    # cnn = CNN()
    model = FeatureExtractor()
    
    train(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=20,
        lr=0.0001,
        optimiser=torch.optim.AdamW
        
    )
 




# %%
