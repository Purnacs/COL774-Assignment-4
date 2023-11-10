## Individual Tasks: <br />
   -- Add any individual tasks --

## Group Tasks:
- [ ] Week 1:
  - [x] Install Torch Library - 5/11/23
  - [x] Make a Kaggle account and Join the Competition - 5/11/23
  - [x] Read Problem Statement - 5/11/23
  - [x] Initial Discussions and Implementation of CNN Architecture - 6/11/23 to 7/11/23
  - [ ] Read about Word Embedding and vocabulary [here](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)
  - [ ] Read about LSTM Implementation [here](https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/)
  - [ ] Define loss and backprop over CNN and RNN
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

- Reading about more efficient libraries:
  - Read about torch vanilla [here](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09)
  - Torch vanilla [documentation](https://lightning.ai/docs/pytorch/stable/)

