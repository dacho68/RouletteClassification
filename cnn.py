import numpy as np
import pandas as pd
import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader as DL
from PIL import Image
import os
from torchvision.transforms import ToTensor

#https://www.youtube.com/watch?v=mozBidd58VQ&list=PLBmhpLzC4hvgm8ix2FLyaVgXtDl7uAtYd&index=11
#from skimage import io
print(torch.__version__)


train_dir_path = "C:\\dev\\src\\RouletteClassification\\data\\roulette_numbers\\train"
test_dir_path = "C:\\dev\\src\\RouletteClassification\\data\\roulette_numbers\\test1"

class RouletteNumberDataset(Dataset):
  def __init__(self, root_dir, transforms = None):
    self.root_dir = root_dir
    self.transforms = transforms
    dirs = os.listdir(root_dir)
    self.items = []
    self.count = 0
    for dir_num in dirs:
      folder_num_path = f"{root_dir}//{dir_num}"
      files = os.listdir(folder_num_path)
      for file in files:
        item = (dir_num,file)
        self.items.append(item)
        self.count += 1
    
  def __len__(self):
    return len(self.items)
  
  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir,self.items[index][0],self.items[index][1])
    image = Image.open(img_path)
    #image = io.imread(img_path)
    
    #print(img_path)
    if self.transforms:
      for t in self.transforms:
        image = t(image)
    
    label = self.items[index][0]
    return (image,torch.tensor(int(label)))


def showOneImage(data):
  img = data[0]
  label = data[1].item()
  plt.imshow(img,cmap="gray")


#transform_lst = [T.Grayscale()]
#ds_train = RouletteNumberDataset(train_dir_path,transform_lst)


class ImageClassifier(nn.Module):

  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.Conv2d(1,32,(3,3)),
      nn.ReLU(),
      nn.Conv2d(32,64,(3,3)),
      nn.ReLU(),
      nn.Conv2d(64,64,(3,3)),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(64*(20-6)*(30-6),37)     
    )
  
  def forward(self,x):
    return self.model(x)

# Instance of the neural network, loss, optimizer

clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# training flow

if __name__ == "__main__":
    transform_lst = [T.Grayscale(),T.Resize((20,30)),T.ToTensor()]
    ds_train = RouletteNumberDataset(train_dir_path,transform_lst)
    print(len(ds_train))
    bs = 32
    train_load = DL(dataset = ds_train, batch_size=bs, shuffle=True )
    for i, (inputs, labels) in enumerate(train_load):
        print(inputs.shape, labels.shape, labels)
        break

    for epoch in range(10) : # train for 10 epochs
        for batch in train_load:
            X,y = batch
            X,y = X.to('cuda'), y.to('cuda')
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            #Apply back prop
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            print(f" Epoch : {epoch} loss is {loss.item()} ")

    with open('model_state.pt','wb') as f:
        save(clf.state_dict(),f)

      