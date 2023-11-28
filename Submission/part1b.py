import os
import sys
import time
import pandas as pd
import numpy as np
import torch.nn as nn
from PIL import Image 
import torch
from torchvision import transforms
import sys
from torch.utils.data import Dataset,DataLoader 
from tqdm import tqdm
import torchtext as text
import random
import csv
import nltk

def sbleu(GT,PRED):
    score = 0
    for i in range(len(GT)):
        Lgt = len(GT[i].split(' '))
        if Lgt > 4 :
            cscore = nltk.translate.bleu_score.sentence_bleu([GT[i].split(' ')],PRED[i].split(' '),weights=(0.25,0.25,0.25,0.25),smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
        else:
            weight_lst = tuple([1.0/Lgt]*Lgt)
            cscore = nltk.translate.bleu_score.sentence_bleu([GT[i].split(' ')],PRED[i].split(' '),weights=weight_lst,smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
        score += cscore
    return score/(len(GT))

 
'''Helper Functions for GPU'''
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

'''Data Handling'''
class LatexDataset(Dataset):
    def __init__(self,df):
        self.df = df
        self.transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[127.5,127.5,127.5],std=[127.5,127.5,127.5])])
    
    def __getitem__(self,index):
        row = self.df.iloc[index]
        img_path = row['image']
        img = Image.open(img_path).convert('RGB')
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
        if data_type == "test":
            image_path = "images/" + "test" + "/"
        else:
            image_path = "images/" + "train" + "/"
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

def vocabulary(latex_data):
    '''
    Creates a vocabulary from the given latex data
    Args:
        latex_data: A numpy array containing the latex data
    Returns:
        A torchtext.vocab object containing the vocabulary
    '''
    latex_data = latex_data.flatten() # Flatten array -> Latex_data is mx1 array
    tokenizer = text.data.utils.get_tokenizer(None)
    vocab = text.vocab.build_vocab_from_iterator(map(tokenizer, latex_data), specials=["<SOS>","<EOS>","<PAD>","<UNK>"],special_first=True)
    return vocab

# TODO: Could use this as the collate fn in Data_loader which would then directly return the labels as tensors instead of doing it in the main code
def labels_to_tensor(labels,vocab):
    '''
    Converts the given labels to a tensor
    Args:
        labels: A numpy array containing the latex labels
        vocab: A torchtext.vocab object containing the vocabulary
    Returns:
        A list of tensors containing the labels in the form of indices from the vocab
    '''
    tokenizer = text.data.utils.get_tokenizer(None)
    
    output_labels = [torch.tensor([vocab["<SOS>"]]+[vocab[token] if token in vocab else vocab["<UNK>"] for token in tokenizer(label)]+[vocab["<EOS>"]], dtype=torch.long) for label in labels]
    
    # # NOTE: To avoid variable time LSTM wrt batches, just fix max_size as some large value and remove the updating step inside previous loop
    max_size = max(tensor.size(0) for tensor in output_labels)

    # The above lines is equivalent to the below for loop
    # output_labels = []
    # max_size = 0
    # for label in labels:
    #     tokens = torch.tensor([vocab[token] for token in tokenizer(label)],dtype=torch.long)
    #     if tokens.shape[0] > max_size: 
    #         max_size = tokens.shape[0]
    #     output_labels.append(tokens)

    # Pad the tensors with 0s aka the <PAD> token at end -> We are gonna do variable length batching (TODO: Check if this is correct)
    output_labels = nn.utils.rnn.pad_sequence(output_labels,batch_first=True,padding_value=2)
    # print(max_size)
    return output_labels

'''Model Architecture'''

# CNN Architecture
class CNN(nn.Module):
    def __init__(self,num_layers=5):
        super(CNN, self).__init__()
        self.num_layers = 5
        self.layer1 = nn.Conv2d(3,32,5)
        self.layer2 = nn.Conv2d(32,64,5)
        self.layer3 = nn.Conv2d(64,128,5)
        self.layer4 = nn.Conv2d(128,256,5)
        self.layer5 = nn.Conv2d(256,512,5)

        self.pooling = nn.MaxPool2d(2)
        self.last_pooling = nn.AvgPool2d(3)
        self.relu = nn.ReLU()
            
    def fit_cnn(self,x):
        for i in range(1, self.num_layers+1):
            x = getattr(self, f'layer{i}')(x)
            x = self.pooling(self.relu(x))
        res = self.last_pooling(self.relu(x))
        return res.reshape(res.shape[0],1,1,512) # -> Reshape to [batch_size, 1, 1, 512] for feeding into LSTM

# LSTM Architecture
class LSTM(nn.Module):
    def __init__(self,vocab,input_dim=1024,emb_dim=512,hidden_dim=512):
        super(LSTM, self).__init__()
        self.vocab = vocab
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(len(vocab),emb_dim,padding_idx=2)
        self.lstm_cell = nn.LSTMCell(input_size = 1024,hidden_size = hidden_dim)
        self.linear = nn.Linear(hidden_dim,len(vocab))

    def lstm_forward(self,context_vec,labels=None):
        '''
        Args:
            x: A tensor of shape [batch_size, input_dim]
        Returns:
            A tensor of shape [batch_size, hidden_dim]
        '''
        batch_size = context_vec.shape[0]
        start_emb = self.embedding(torch.tensor([self.vocab.__getitem__("<SOS>")]*batch_size,dtype=torch.long).to(device)) #TODO: Check if this is even correct + Why is it that I have to do to(device) here?
        if labels is None: #ie the case of pure inference
            total_time_steps = 650 #Max expected size of a formula?
            tf = 0 #Avoid doing the label embedding step + pure prediction
        else: #ie the case of training
            total_time_steps = labels.shape[1] 
            label_emb = self.embedding(labels) 
            tf = 0.5 #Teacher forcing probability
        x_t = torch.concat((context_vec,start_emb),dim=1) 
        h_t = c_t = context_vec
        outputs = torch.zeros(batch_size,total_time_steps,len(self.vocab))
        for t in range(1,total_time_steps):
            h_t, c_t = self.lstm_cell(x_t,(h_t,c_t))
            output = self.linear(h_t)
            outputs[:,t,:] = output #TODO: Check if this is even correct
            if random.random() < tf: #Teacher forcing
                next_inp = label_emb[:,t,:]
            else:
                next_inp = self.embedding(torch.argmax(outputs[:,t,:],dim=1).to(device)) #TODO: Check if this is even correct + Why is it that I have to do to(device) here?
            x_t = torch.concat((context_vec,next_inp),dim=1)
        return outputs

        
# Combined Architecture
class Latex_arch(nn.Module):
    def __init__(self,vocab):
        super(Latex_arch, self).__init__()
        self.encoder = CNN()
        self.decoder = LSTM(vocab)
        self.vocab = vocab

    def latex_forward(self,x,labels=None):
        '''
        Args:
            x: A tensor of shape [batch_size, 3, 224, 224]
        Returns:
            A tensor of shape [batch_size, hidden_dim]
        '''
        context_vec = self.encoder.fit_cnn(x).reshape(x.shape[0],512)
        out = self.decoder.lstm_forward(context_vec,labels)
        return out

''' Converting prediction to latex '''
def prediction_to_latex(prediction,vocab):
    '''
    Converts the given prediction to latex
    Args:
        prediction: A tensor of shape [batch_size, max_len, vocab_size]
        vocab: A torchtext.vocab object containing the vocabulary
    Returns:
        A list of strings containing the latex labels
    '''
    output_labels = []
    for i in range(prediction.shape[0]):
        output_labels.append(" ".join([vocab.lookup_token(torch.argmax(index)) for index in prediction[i]]))
    return output_labels

def labels_to_latex(labels,vocab):
    '''
    Converts the given labels to latex
    Args:
        labels: A tensor of shape [batch_size, max_len]
        vocab: A torchtext.vocab object containing the vocabulary
    Returns:
        A list of strings containing the latex labels
    '''
    output_labels = []
    for i in range(labels.shape[0]):
        output_labels.append(" ".join([vocab.lookup_token(index) for index in labels[i]]))
    return output_labels

"Testing Area"
if __name__ == '__main__':
    # Converting to GPU if available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"Using {device} device")


    print("Loading Datasets...")
    dir_path = sys.argv[1]
    # dir_path = "/kaggle/input/converting-handwritten-equations-to-latex-code/col_774_A4_2023"
    model_save_path = "./model.tar"
    hw_model_save_path = "./model_hw.tar"
    batch_size = 64
    tr_syn,tr_syn_df = import_data(dir_path,True,"train",batch_size)
    t_syn,t_syn_df = import_data(dir_path,True,"test",1)
    v_syn,v_syn_df= import_data(dir_path,True,"val",1)
    tr_hw,tr_hw_df = import_data(dir_path,False,"train",batch_size)
    t_hw,t_hw_df = import_data(dir_path,False,"test",1)
    val_hw,v_hw_df = import_data(dir_path,False,"val",1)
    
    print("Creating Vocabulary...")
    formula = np.array(tr_syn_df["formula"])
    vocab = vocabulary(formula)
    
    print("Initializing Model...")
    model = Latex_arch(vocab)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3) 

    # Load model if it exists
    if os.path.exists(model_save_path):
        print("Loading Model...")
        checkpoint = torch.load(model_save_path,map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    # Shift things to device
    model.to(device)
    optimizer_to(optimizer,device)
    # criterion.to(device)

    num_epochs = 100
    print("Fine Tuning for HandWritten Data...")
    for epoch in range(num_epochs):
        train_loss = 0
        for i,(images,labels) in enumerate(tqdm(tr_hw)): #Could do as tqdm(tr_syn) to get a progress bar 
            model.train() # Set model to training mode
            images = images.to(device)
            tensor_labels = labels_to_tensor(labels,vocab).to(device)

            if (i+1)%20 == 0: # Save model every 20 steps
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, hw_model_save_path)

            output = model.latex_forward(images,tensor_labels).to(device)

            loss = criterion(output.reshape(output.shape[0],output.shape[2],output.shape[1]),tensor_labels)  #NOTE: The reshaping is due to definition of cross-entropy
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
 
            train_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Batch Loss: {train_loss/len(tr_syn)}')

    print('''Testing Model...''')
    test_datas = [v_syn,t_syn,val_hw]
    test_data_names = {v_syn:"Synthetic Validation",t_syn:"Synthetic Test",val_hw:"Handwritten Validation"}
    for test_data in test_datas:
        ground_truths = []
        predictions = []
        for i,(images,labels) in enumerate(tqdm(test_data)):
            model.eval() # Set model to evaluation mode
            images = images.to(device)
            tensor_labels = labels_to_tensor(labels,vocab).to(device)
            output = model.latex_forward(images).to(device)

            present_ground_truths = labels_to_latex(tensor_labels,vocab)
            present_predictions = prediction_to_latex(output,vocab)
            #print(f'Ground Truth: {present_ground_truths}')
            #print(f'Prediction: {present_predictions}')
            ground_truths.extend(present_ground_truths)
            predictions.extend(present_predictions)

            bleu_score_iter = sbleu(predictions, ground_truths)
            
            print(f'Current Macro Bleu for batch {i}: {bleu_score_iter}')
            print("\n")
        total_bleu_score = sbleu(predictions, ground_truths)
        print(f'Total Macro Bleu for {test_data_names[test_data]}: {total_bleu_score}')
