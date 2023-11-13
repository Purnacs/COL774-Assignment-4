## Individual Tasks: <br />
   -- Add any individual tasks --

## Group Tasks:
- [ ] Week 1:
  - [x] Install Torch Library - 5/11/23
  - [x] Make a Kaggle account and Join the Competition - 5/11/23
  - [x] Read Problem Statement - 5/11/23
  - [x] Initial Discussions and Implementation of CNN Architecture - 6/11/23 to 7/11/23
  - [ ] Integrate the Data_processing.py and Ass4_1.py codes for input handling -> Would completely fix the Input Handling part of assigment
  - [ ] Optional: Change name of Ass4_1.py to something more representative (lol)
  - [ ] Read about Word Embedding and vocabulary [here](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)
  - [ ] Read about LSTM Implementation [here](https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/)
  - [ ] Read about PyTorch NN tutorial [here] (https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
  - [x] Define loss and backprop over CNN and RNN
  - [ ] What we have is a classification model. Learn more about translation models and implement them
  - [ ] Try out naive implementations to solve the problem statement
  - [ ] Look out for better models to increase accuracy

- [ ] Week 2:
- [ ] Week 3:

- [ ] Submit Assignment - 27/11/23

## Future Tasks:


- Logging/Saving files:
  - since we can't keep on waiting for stuff like reading file and converting to tensor every single time we run the file we would have to start some kind of logging
  - Some of the things we would have to log:
      1) Initial Tensor of train/test/val data
      2) If convolution completed properly then save convoluted tensor of train/tensor/val data 
      3) The weights of the RNN's (anyway need to do this since for final submission we basically have to submit the completed RNN with final learned weights ig)
  - Also we prbly need to save all the weights we get as we train everytime  and choose the weights that give max accuracy at the end of submission time ig
  - Idea for logging of 1) Check if the "tensor" file exists , if it does then directly read that instead of trying to convert images to tensor, if it does not then run the get_data function
- NOTE: Apparently torch directly has a method for saving models - [Saving and Loading Models](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html), TL;DR - torch.save(model.state_dict(), "model.pth") [here model is the instance of the neural net class that we would implement]


- Reading about more efficient libraries:
  - Read about torch vanilla [here](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09)
  - Torch vanilla [documentation](https://lightning.ai/docs/pytorch/stable/)

## Tasks just for fun (For code cleanliness lol):

- In `./Testing/Data_processing.py` update the `get_path()` function to make it more generalized