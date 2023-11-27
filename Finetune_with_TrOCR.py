# %% [markdown]
# <a href="https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_native_PyTorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# ## Fine-tune TrOCR on the IAM Handwriting Database
# 
# In this notebook, we are going to fine-tune a pre-trained TrOCR model on the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database), a collection of annotated images of handwritten text.
# 
# We will do this using the new `VisionEncoderDecoderModel` class, which can be used to combine any image Transformer encoder (such as ViT, BEiT) with any text Transformer as decoder (such as BERT, RoBERTa, GPT-2). TrOCR is an instance of this, as it has an encoder-decoder architecture, with the weights of the encoder initialized from a pre-trained BEiT, and the weights of the decoder initialized from a pre-trained RoBERTa. The weights of the cross-attention layer were randomly initialized, before the authors pre-trained the model further on millions of (partially synthetic) annotated images of handwritten text. 
# 
# This figure gives a good overview of the model (from the original paper):
# 

# * TrOCR paper: https://arxiv.org/abs/2109.10282
# * TrOCR documentation: https://huggingface.co/transformers/master/model_doc/trocr.html
# 
# 
# Note that Patrick also wrote a very good [blog post](https://huggingface.co/blog/warm-starting-encoder-decoder) on warm-starting encoder-decoder models (which is what the TrOCR authors did). This blog post was very helpful for me to create this notebook. 
# 
# We will fine-tune the model using native PyTorch.
# 
# 
# 
# ## Set-up environment
# 
# First, let's install the required libraries:
# * Transformers (for the TrOCR model)
# * Datasets & Jiwer (for the evaluation metric)
# 
# We will not be using HuggingFace Datasets in this notebook for data preprocessing, we will just create a good old basic PyTorch Dataset.

# %%
# !pip install -q transformers

# %%
# !pip install -q datasets jiwer

# %% [markdown]
# ## Prepare data
# 
# We first download the data. Here, I'm just using the IAM test set, as this was released by the TrOCR authors in the unilm repository. It can be downloaded from [this page](https://github.com/microsoft/unilm/tree/master/trocr). 
# 
# Let's make a [regular PyTorch dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). We first create a Pandas dataframe with 2 columns. Each row consists of the file name of an image, and the corresponding text.

# %%
import pandas as pd

# df = pd.read_fwf('/content/drive/MyDrive/TrOCR/Tutorial notebooks/IAM/gt_test.txt', header=None)
train_syn_csv_path = 'Dataset/SyntheticData/train.csv'
train_df = pd.read_csv(train_syn_csv_path)
train_df.rename(columns={"image": "file_name", "formula": "text"}, inplace=True)
# print(train_df)
# del df[2]
# some file names end with jp instead of jpg, let's fix this
train_df['file_name'] = train_df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
train_df.head()

# %% [markdown]
# We split up the data into training + testing, using sklearn's `train_test_split` function.

# %%
# from sklearn.model_selection import train_test_split

# train_df, test_df = train_test_split(df, test_size=0.2)
# # we reset the indices to start from zero
# train_df.reset_index(drop=True, inplace=True)
# test_df.reset_index(drop=True, inplace=True)
test_syn_csv_path = 'Dataset/SyntheticData/test.csv'
test_df = pd.read_csv(test_syn_csv_path)
test_df.rename(columns={"image": "file_name", "formula": "text"}, inplace=True)
# print(test_df)

# %% [markdown]
# Each element of the dataset should return 2 things:
# * `pixel_values`, which serve as input to the model.
# * `labels`, which are the `input_ids` of the corresponding text in the image.
# 
# We use `TrOCRProcessor` to prepare the data for the model. `TrOCRProcessor` is actually just a wrapper around a `ViTFeatureExtractor` (which can be used to resize + normalize images) and a `RobertaTokenizer` (which can be used to encode and decode text into/from `input_ids`). 

# %%
import torch
from torch.utils.data import Dataset
from PIL import Image

class LatexDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

# %% [markdown]
# Let's initialize the training and evaluation datasets:

# %%
# NOTE: IMAGE PROCESSOR CLASS NOT FOUND APPARANTLY -> POSSIBLE FUTURE ERROR SOURCE
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
train_dataset = LatexDataset(root_dir='Dataset/SyntheticData/images/',
                           df=train_df,
                           processor=processor)
eval_dataset = LatexDataset(root_dir='Dataset/SyntheticData/images/',
                           df=test_df,
                           processor=processor)

# %%
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

# %% [markdown]
# Let's verify an example from the training dataset:

# %%
encoding = train_dataset[0]
for k,v in encoding.items():
  print(k, v.shape)

# %% [markdown]
# We can also check the original image and decode the labels:

# %%
image = Image.open(train_dataset.root_dir + train_df['file_name'][0]).convert("RGB")
image

# %%
labels = encoding['labels']
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.decode(labels, skip_special_tokens=True)
print(label_str)

# %% [markdown]
# Let's create corresponding dataloaders:

# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=4)

# %% [markdown]
# ## Train a model
# 
# Here, we initialize the TrOCR model from its pretrained weights. Note that the weights of the language modeling head are already initialized from pre-training, as the model was already trained to generate text during its pre-training stage. Refer to the paper for details.

# %%
from transformers import VisionEncoderDecoderModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
model.to(device)

# %% [markdown]
# Importantly, we need to set a couple of attributes, namely:
# * the attributes required for creating the `decoder_input_ids` from the `labels` (the model will automatically create the `decoder_input_ids` by shifting the `labels` one position to the right and prepending the `decoder_start_token_id`, as well as replacing ids which are -100 by the pad_token_id)
# * the vocabulary size of the model (for the language modeling head on top of the decoder)
# * beam-search related parameters which are used when generating text.

# %%
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 200
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

# %% [markdown]
# We will evaluate the model on the Character Error Rate (CER), which is available in HuggingFace Datasets (see [here](https://huggingface.co/metrics/cer)).

# %%
from datasets import load_metric

cer_metric = load_metric("cer")

# %%
def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer

# %%
# !pip install ipywidgets
# !jupyter nbextension enable --py widgetsnbextension
from transformers import AdamW
from tqdm.notebook import tqdm

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(20):  # loop over the dataset multiple times
   # train
   model.train()
   train_loss = 0.0
   for batch in tqdm(train_dataloader):
      # get the inputs
      for k,v in batch.items():
        batch[k] = v.to(device)

      # forward + backward + optimize
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      train_loss += loss.item()

   print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    
   # evaluate
   model.eval()
   valid_cer = 0.0
   with torch.no_grad():
     for batch in tqdm(eval_dataloader):
       # run batch generation
       outputs = model.generate(batch["pixel_values"].to(device))
       # compute metrics
       cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
       valid_cer += cer 

   print("Validation CER:", valid_cer / len(eval_dataloader))

model.save_pretrained(".")

# %% [markdown]
# ## Inference
# 
# Note that after training, you can easily load the model using the .`from_pretrained(output_dir)` method.
# 
# For inference on new images, I refer to my inference notebook, that can also be found in my [Transformers Tutorials repository](https://github.com/NielsRogge/Transformers-Tutorials) on Github.


