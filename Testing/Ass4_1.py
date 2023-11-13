import pandas as pd
import numpy as np
import torch.nn as nn
from PIL import Image 
import cv2
import torch
import os
import sys
from torchvision import transforms
import sys
from torch.utils.data import Dataset,DataLoader 
import torchtext as text

class LatexDataset(Dataset):
    def __init__(self,df):
        self.df = df
        self.transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0])])
    
    def __getitem__(self,index):
        row = self.df.iloc[index]
        img_path = row['image']
        img = Image.open(img_path)
        img = self.transform(img)
        return img,row['formula']
    
    def __len__(self):
        return len(self.df)

# TODO: Could update this function to be a bit more better
def get_path(dir_path,is_synthetic,data_type):
    if is_synthetic: 
        data_path = "/SyntheticData/"
        image_path = "images/"
    else:
        data_path = "/HandwrittenData/"
        image_path = "images/" + data_type + "/"
        data_type += "_hw"
    csv_path = dir_path + data_path + data_type + ".csv"
    img_path = dir_path + data_path + image_path
    return csv_path,img_path

def import_data(dir_path,is_synthetic:bool=True, data_type:str = "train",batch_size = 64):
    '''
    Imports data from the given directory and uses the LatexDataset class and the torch.DataLoader to create a dataloader iterator
    Args:
        dir_path: The parent directory path to dataset (specifically ./Dataset)
        is_synthetic: Variable for determining whether to use synthetic or Handwritten Data
        data_type: Variable for determining train, test or validation dataset to return
    Returns:
        A torch.DataLoader object containing the required Data ]in batches with shuffling(if training data)]
    '''
    is_shuffle = True if data_type == "train" else False

    # The below code block is only because of the given dataset directory structure
    csv_path,img_path = get_path(dir_path,is_synthetic,data_type)

    df = pd.read_csv(csv_path) # read csv
    df["image"] = df["image"].apply(lambda x: img_path + x) # add path to image name
    dataset = LatexDataset(df)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=is_shuffle,num_workers=4)
    return dataloader, df


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Conv2d(3,32,5)
        self.layer2 = nn.Conv2d(32,64,5)
        self.layer3 = nn.Conv2d(64,128,5)
        self.layer4 = nn.Conv2d(128,256,5)
        self.layer5 = nn.Conv2d(256,512,5)

        self.pooling = nn.AvgPool2d(2)
        self.last_pooling = nn.AvgPool2d(3)
        self.relu = nn.ReLU()
        
    def single_example_cnn(self,x):
        x = self.layer1(x)
        x = self.pooling(self.relu(x))
        x = self.layer2(x)
        x = self.pooling(self.relu(x))
        x = self.layer3(x)
        x = self.pooling(self.relu(x))
        x = self.layer4(x)
        x = self.pooling(self.relu(x))
        x = self.layer5(x)
        x = self.pooling(self.relu(x))
        return self.last_pooling(self.relu(x))
    
    def fit_cnn(self,x):
        res = self.single_example_cnn(x)
        return res.reshape(128,1,1,512)
        

class LSTM (nn.Module):
    def __init__(self,vocab_size,emb_dim,hid_dim):
        super(LSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size,emb_dim,padding_idx=1)
        self.lstm = nn.LSTM(emb_dim,hid_dim,num_layers=1,batch_first=True)
        self.fc = nn.Linear(hid_dim,vocab_size)

    # ?? error ??
    def fit_lstm(self,x):
        data = x.reshape(128,512)
        emb = self.emb(data.to(torch.long))
        out,hidden = self.lstm(emb)
        prediction = self.fc(out)
        return prediction

    
class Actual_net (nn.Module):
    def __init__(self,vocab):
        super(Actual_net, self).__init__()
        self.cnn = CNN()
        self.lstm = LSTM(len(vocab),512,512)
    
    def fit(self,x):
        cnn_out = self.cnn.fit_cnn(x)
        lstm_out = self.lstm.fit_lstm(cnn_out)
        loss = nn.CrossEntropyLoss(lstm_out)
        return lstm_out
    
    def predict(self,lstm_out):
        # cnn_out = self.cnn.fit_cnn(x)
        # lstm_out = self.lstm.fit_lstm(cnn_out)
        _, top_indices = lstm_out.topk(1, dim=-1)
    # Decode top_indices to get formula
        formula = decode_text(vocab, top_indices.squeeze().tolist())
        return formula

def decode_text(vocab, indices):
    # Convert indices to tokens
    tokens = [vocab[i] for i in indices]
    # Join tokens into a string with LaTeX formatting
    formula = ' '.join(tokens)
    return formula
        


def vocabulary(latex_data):
    latex_data = latex_data.flatten()
    tokenizer = text.data.utils.get_tokenizer('basic_english')
    vocab = text.vocab.build_vocab_from_iterator(map(tokenizer, latex_data))
    return vocab

def encode_text(vocab,formula):
    tokenizer = text.data.utils.get_tokenizer('basic_english')
    text_transform = lambda x: [vocab[token] for token in tokenizer(x)]
    trans = text_transform(formula)
    arr = np.zeros(512)
    arr[trans] = 1
    return arr

def embed_text(vocab,formulas,embedding):
    res = []
    i = 0
    for formula in formulas:
        idx = torch.Tensor(encode_text(vocab,formula)).to(torch.long)
        emb1 = embedding(idx)
        res.append(emb1)
        i = i+1
    return torch.stack(res)






'''Testing Area'''
if __name__ == '__main__':
    #NOTE: Use the following small codeblock for device usage (particularly when using HPC)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    device = "cpu"
    print(f"Using {device} device")
    dir_path = sys.argv[1]
    # batch_s = 128
    # dirname = os.path.dirname(__file__)
    # dir_path = os.path.join(dirname,"../Dataset")
    tr_syn_dl,tr_syn_df = import_data(dir_path,True,"train",128)
    t_syn,t_syn_df = import_data(dir_path,True,"test",128)
    v_syn,v_syn_df= import_data(dir_path,True,"val",128)
    tr_hw,tr_hw_df = import_data(dir_path,False,"train",128)
    val_hw,v_hw_df = import_data(dir_path,False,"val",128)
    # print(df)
    # print(images)
    formula = np.array(tr_syn_df["formula"])
    vocab = vocabulary(formula)
    model = Actual_net(vocab)
    loss = nn.CrossEntropyLoss()
    emb = nn.Embedding(512,469,padding_idx=1)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    model = model.to(device)
    loss = loss.to(device)
    emb = emb.to(device)
    num_epochs = 1
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(tr_syn_dl): # -> Convert the DataLoader object to iterator and then call next for getting the batches one by one which would contain tensor of both images and formulas
            inputs = inputs.to(device)
            # labels = torch.Tensor(labels).to(device)
            encoding = embed_text(vocab,labels,emb)
            # print("encoding", encoding.shape)
            print(i)
            if i == 200:
                break
            out = model.fit(inputs)
            # print(model.predict(out))
            loss_out = loss(out,encoding)
            optimizer.zero_grad()
            loss_out.backward()
            optimizer.step()
            # Print loss for every 128 steps
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(tr_syn_dl)}], Loss: {loss_out.item()}')
            



    # print(next(iter(tr_syn))) # -> Convert the DataLoader object to iterator and then call next for getting the batches one by one which would contain tensor of both images and formulas


    '''
    # NOTE: Each batch would be a 4D Tensor of size [batch_size, num_channels, height, width], whereas each image within it is a 3D tensor of size [num_channels, height, width]
    
    A Sample code for using the dataloader assuming some model
    model = ...
    criterion = ...
    optimizer = ...
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            # Move data to GPU if available
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss for every 128 steps
            if i % 128 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')
    '''