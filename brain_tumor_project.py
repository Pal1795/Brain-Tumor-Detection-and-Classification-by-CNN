#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install split-folders
#!pip install torch-summary


# # 1 Import Libraries

# In[2]:


# Import essential libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='darkgrid')
import copy # A module that provides functions for creating copies of objects, useful for avoiding unintended modifications to variables.
import os # A module that provides a way to interact with the operating system, allowing for tasks such as file and directory manipulation.
import torch
from PIL import Image # A module from the Python Imaging Library (PIL) that provides functionality for opening, manipulating, and saving various image file formats.
from torch.utils.data import Dataset # A PyTorch class that represents a dataset and provides an interface for accessing and processing the data during training.
import torchvision
import torchvision.transforms as transforms # A module from the torchvision library that provides common image transformations, such as resizing, cropping, and normalization.
from torch.utils.data import random_split # A function from PyTorch that allows for randomly splitting a dataset into training and validation subsets.
from torch.optim.lr_scheduler import ReduceLROnPlateau # A PyTorch scheduler that adjusts the learning rate during training based on a specified metric, reducing it when the metric plateaus.
import torch.nn as nn # A module in PyTorch that provides classes for defining and building neural networks.
from torchvision import utils # A module from torchvision that contains utility functions for working with images, such as saving and visualizing them.
from torchvision.datasets import ImageFolder
import splitfolders
from torchsummary import summary
import torch.nn.functional as F
import pathlib
from sklearn.metrics import confusion_matrix, classification_report
import itertools # This import statement imports the itertools module, which provides functions for efficient looping and combining of iterables. It can be used for tasks such as generating combinations or permutations of elements.
from tqdm.notebook import trange, tqdm # These functions allow for the creation of progress bars to track the progress of loops or tasks.
from torch import optim
import warnings
warnings.filterwarnings('ignore')


# # 2 Load Dataset

# In[10]:


labels_df = pd.read_csv('metadata.csv')
print(labels_df.head().to_markdown())


# In[11]:


labels_df.shape


# In[12]:


os.listdir('/Users/pallavign/Downloads/Ban 676 Deep Learning/Team Project/Final Project/Brain Tumor Data Set')


# # 3 Data Preparation

# In[14]:


# Dataset Path
data_dir = '/Users/pallavign/Downloads/Ban 676 Deep Learning/Team Project/Final Project/Brain Tumor Data Set'
data_dir = pathlib.Path(data_dir)


# In[15]:


# Splitting dataset to train_set, val_set / test_set
splitfolders.ratio(data_dir, output='brain', seed=20, ratio=(0.8, 0.2))


# In[16]:


# New dataset path
data_dir = '/Users/pallavign/Downloads/Ban 676 Deep Learning/Team Project/Final Project/brain'
data_dir = pathlib.Path(data_dir)


# # 4 Image Augmentation Definitions

# In[23]:


# define transformation
transform = transforms.Compose( # transforms.Compose 
    [
        transforms.Resize((256,256)), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30), 
        transforms.ToTensor(), 
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
   ]
)


# In[24]:


# Define an object of the custom dataset for the train and validation.
train_set = torchvision.datasets.ImageFolder(data_dir.joinpath("train"), transform=transform) 
train_set.transform
val_set = torchvision.datasets.ImageFolder(data_dir.joinpath("val"), transform=transform)
val_set.transform


# In[26]:


# Visualiztion some images from Train Set 
CLA_label = {
    0 : 'Brain Tumor',
    1 : 'Healthy'
} 
figure = plt.figure(figsize=(10, 10))
cols, rows = 4, 4
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_set), size=(1,)).item() 
    img, label = train_set[sample_idx] 
    figure.add_subplot(rows, cols, i) 
    plt.title(CLA_label[label]) 
    plt.axis("off") 
    img_np = img.numpy().transpose((1, 2, 0)) 
    
    # Clip pixel values to [0, 1]
    img_valid_range = np.clip(img_np, 0, 1)
    plt.imshow(img_valid_range) 
    plt.suptitle('Brain Images', y=0.95) 
plt.show()


# # 5 Creating Dataloaders

# In[27]:


# import and load train, validation
batch_size = 64 # batch_size 

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 2)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True, num_workers = 2)


# In[28]:


# print shape for Training data and Validation data
for key, value in {'Training data': train_loader, "Validation data": val_loader}.items():
    for X, y in value:
        print(f"{key}:")
        print(f"Shape of X : {X.shape}") 
        print(f"Shape of y: {y.shape} {y.dtype}\n") 
        break


# # 6 Define Brain Tumor Classifier

# In[29]:


'''This function can be useful in determining the output size of a convolutional layer in a neural network,
given the input dimensions and the convolutional layer's parameters.'''

def findConv2dOutShape(hin,win,conv,pool=2): 
    
    # get conv arguments
    kernel_size = conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)


# In[30]:


# Define Architecture For CNN_TUMOR Model
class CNN_TUMOR(nn.Module):
    
    # Network Initialisation
    def __init__(self, params):
        
        super(CNN_TUMOR, self).__init__()
    
        Cin,Hin,Win = params["shape_in"]
        init_f = params["initial_filters"] 
        num_fc1 = params["num_fc1"]  
        num_classes = params["num_classes"] 
        self.dropout_rate = params["dropout_rate"] 
        
        # Convolution Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h,w=findConv2dOutShape(Hin,Win,self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)
        
        # compute the flatten size
        self.num_flatten=h*w*8*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self,X):
        
        # Convolution & Pool Layers
        X = F.relu(self.conv1(X)); 
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)


# In[31]:


params_model={
        "shape_in": (3,256,256), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2}

# Create instantiation of Network class
cnn_model = CNN_TUMOR(params_model)

# define computation hardware approach (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = cnn_model.to(device)


# In[32]:


# Model Summary for CNN Model
summary(cnn_model, input_size=(3, 256, 256),device=device.type)


# # 7 Loss Function Definition

# In[33]:


loss_func = nn.NLLLoss(reduction="sum") 


# # 8 Optimiser Definition

# In[34]:


opt = optim.Adam(cnn_model.parameters(), lr=3e-4) 
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)


# # 9 Training Model

# In[35]:


# Function to get the learning rate 
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# Function to compute the loss value per batch of data 
def loss_batch(loss_func, output, target, opt=None):
    
    loss = loss_func(output, target) # get loss
    pred = output.argmax(dim=1, keepdim=True) # Get Output Class
    metric_b=pred.eq(target.view_as(pred)).sum().item() # get performance metric
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


# In[37]:


# Compute the loss value & performance metric for the entire dataset (epoch) 
def loss_epoch(model,loss_func,dataset_dl,opt=None):
    
    run_loss=0.0 
    t_metric=0.0
    len_data=len(dataset_dl.dataset)

    # internal loop over dataset
    for xb, yb in dataset_dl:
        # move batch to device
        xb=xb.to(device)
        yb=yb.to(device)
        output=model(xb) # get model output
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt) # get loss per batch
        run_loss+=loss_b        # update running loss

        if metric_b is not None: # update running metric
            t_metric+=metric_b    
    
    loss=run_loss/float(len_data)  # average loss value
    metric=t_metric/float(len_data) # average metric value
    
    return loss, metric


# ## 9.2 Training Function

# In[38]:


def Train_Val(model, params,verbose=False): 
    
    # Get the parameters
    epochs=params["epochs"]
    loss_func=params["f_loss"]
    opt=params["optimiser"]
    train_dl=params["train"]
    val_dl=params["val"]
    lr_scheduler=params["lr_change"]
    weight_path=params["weight_path"]
    

    # history of loss values in each epoch 
    loss_history={"train": [],"val": []} 
    # histroy of metric values in each epoch 
    metric_history={"train": [],"val": []} 
    # a deep copy of weights for the best performing model 
    best_model_wts = copy.deepcopy(model.state_dict()) 
    # initialize best loss to a large value 
    best_loss=float('inf') 
    
# Train Model n_epochs 
# (the progress of training by printing the epoch number and the associated learning rate. 
# It can be helpful for debugging, monitoring the learning rate schedule, or gaining insights into the training process.) 
    
    for epoch in tqdm(range(epochs)):
        
        # Get the Learning Rate
        current_lr=get_lr(opt)
        if(verbose):
            print('Epoch {}/{}, current lr={}'.format(epoch, epochs - 1, current_lr))

        
# Train Model Process 

        model.train() 
        train_loss, train_metric = loss_epoch(model,loss_func,train_dl,opt) 

        # collect losses 
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        # Validation Step
        model.eval()  # set the model to evaluation mode  # ** newly added **
        with torch.no_grad():  # disable gradient calculation for validation  # ** newly added **
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)  # ** newly added **

        # collect validation losses
        loss_history["val"].append(val_loss)  # ** newly added **
        metric_history["val"].append(val_metric)  # ** newly added **
        
        # learning rate schedule
        lr_scheduler.step(val_loss)  # use validation loss for scheduler 
        if val_loss < best_loss:  # check if the current model is the best  # ** newly added **
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict()) 
            if verbose:
                print("Best model weights updated!")  # ** newly added **
        
        if verbose:
            print(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, "
                  f"train accuracy: {100*train_metric:.2f}, val accuracy: {100*val_metric:.2f}")  # ** modified **
            print("-" * 10)

    # load best model weights
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history


# ## 9.3 Training Process

# In[39]:


# Define various parameters used for training and evaluation of a cnn_model
params_train={
 "train": train_loader,"val": val_loader,
 "epochs": 60,
 "optimiser": optim.Adam(cnn_model.parameters(),lr=3e-4),
 "lr_change": ReduceLROnPlateau(opt,
                                mode='min',
                                factor=0.5,
                                patience=20,
                                verbose=0),
 "f_loss": nn.NLLLoss(reduction="sum"),
 "weight_path": "weights.pt",
}


# In[40]:


# train and validate the model
cnn_model,loss_hist,metric_hist = Train_Val(cnn_model,params_train)


# # 10 Evaluation Metric Visualization 

# In[41]:


# Convergence History Plot
epochs=params_train["epochs"]
fig,ax = plt.subplots(1,2,figsize=(12,5))

sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["train"],ax=ax[0],label='loss_hist["train"]')
sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["val"],ax=ax[0],label='loss_hist["val"]')
sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist["train"],ax=ax[1],label='Acc_hist["train"]')
sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist["val"],ax=ax[1],label='Acc_hist["val"]')


# ## 10.2 Confusion_Matrix

# In[42]:


# define function For Classification Report
def Ture_and_Pred(val_loader, model):
    i = 0
    y_true = []
    y_pred = []
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.numpy()
        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)
        pred = pred.detach().cpu().numpy()
        
        y_true = np.append(y_true, labels)
        y_pred = np.append(y_pred, pred)
    
    return y_true, y_pred


# check confusion matrix for error analysis
y_true, y_pred = Ture_and_Pred(val_loader, cnn_model)

print(classification_report(y_true, y_pred), '\n\n')
cm = confusion_matrix(y_true, y_pred)


# In[43]:


# Confusion Matrix Plotting Function
def show_confusion_matrix(cm, CLA_label, title='Confusion matrix', cmap=plt.cm.YlGnBu):
    
    plt.figure(figsize=(10,7))
    plt.grid(False)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(CLA_label))

    plt.xticks(tick_marks, [f"{value}={key}" for key , value in CLA_label.items()], rotation=45)
    plt.yticks(tick_marks, [f"{value}={key}" for key , value in CLA_label.items()])

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i,j]}\n{cm[i,j]/np.sum(cm)*100:.2f}%", horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

show_confusion_matrix(cm, CLA_label)


# # 11 Save Model 

# In[44]:


torch.save(cnn_model, "Brain_Tumor_model.pt")

