# Deep_Learning_Assignment_3
Deep Learning Assignment 3: Sequence models.

In this assignment we implemented next word prediction model using RNNs. 
Then, we used this model for text generation in several ways.

---
## Load The Data
we used the Penn Treebank Dataset from [here](https://deepai.org/dataset/penn-treebank).

 ## Networks Visualization   
```
LSTM_Predictor(
  (embedding): Embedding(9701, 128)
  (lstm): LSTM(128, 256, num_layers=2, dropout=0.3)
  (out_fc): Linear(in_features=256, out_features=9701, bias=True)
  (log_softmax): LogSoftmax(dim=2)
  (loss_func): CrossEntropyLoss()
)

number of parameters = 4,656,485
```

## Training Results
```
Train: Accuracy = 37.65%, Avg Loss = 3.63
Validation: Accuracy = 34.41%, Avg Loss = 4.26
Test: Accuracy = 35.57%, Avg Loss = 4.19

Early stopping after 25 / 30 epohes

	Time taken: 00:09:20.33
```