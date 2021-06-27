
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm

# GLOBALS:
from Penn_Treebank_Dataset import load_datasets

device = None





class WordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, h_dim, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = h_dim

        # nn.Embedding converts from token index to dense tensor
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim)  # this is the embedding for the "translated" language..

        # PyTorch multilayer LSTM
        self.lstm = nn.LSTM(embedding_dim, h_dim, num_layers=num_layers, dropout=dropout)

        # Output layer, note the output dimension!
        self.out_fc = nn.Linear(h_dim,
                                vocab_size)  # output-dim = vocab_size here... since we'll apply softmax over the whole available words..

        # over vocab dim...
        self.log_softmax = nn.LogSoftmax(dim=2)

        # ATTEMPTS:
        # self.bn = nn.BatchNorm1d(embedding_dim) #after 1st embedding and relu
        # self.relu = nn.ReLU() #after 1st embedding, before bn layer

    def forward(self, x, forward_mode="train", initial_vec_h0=None, initial_vec_c0=None):

        if forward_mode == "train" or forward_mode == "gen_by_input":  # this will be used upon training/generating by input (b. case)
            # x shape: (S, B) Note batch dim is not first!
            S, B = x.shape

            # Embed the input:
            embedded = self.embedding(x)  # embedded shape: (S, B, E) - remains the same...

            # ATEMPT TO IMRPOV:
            # embedded = torch.swapaxes(embedded,0,1)
            # embedded = self.relu(embedded)
            # embedded = self.bn(embedded)
            # embedded = torch.swapaxes(embedded,0,1)

            # Run through the LSTMs
            output, (h_t, c_t) = self.lstm(
                embedded)  # notice, we don't use the "out", meaning the y's here. (since it's the encoder..)

            # Project H back to the vocab size V, to get a score per word
            output = nn.functional.relu(output)
            out = self.out_fc(output)  # (seq,batch,vocsize)

            return out

        if forward_mode == "teacher_force":  # used upon generation by teacher_forcing
            # x = a vector of size=voc, unbatched (though in batch size of 1). (S,B) s=1,b=1. (this is the "first word")
            # initial_vec_c0/h0 = of shape (layers, batchsize, hiddensize)
            # x shape: (S, B) Note batch dim is not first!
            S, B = x.shape
            # h_0 = torch.zeros(self.num_layers, B, self.hidden_size)  # (num_layers, batch_size, h_dim)
            # c_0 = torch.zeros(self.num_layers, B, self.hidden_size)  # (num_layers, batch_size, h_dim)

            embedded, y_t, h_t, c_t = None, None, None, None
            generated_sentence = []  # will hold the generated sentence as a list of tokenized words

            for i in range(50):  # max 50 words generated`
                if i == 0:  # for the 1st block:
                    embedded = self.embedding(x)
                    y_t, (h_t, c_t) = self.lstm(embedded, (initial_vec_h0, initial_vec_c0))
                    y_t = self.out_fc(y_t)  # y_t = (seq,batch,vocsize) - holds scores per word in voc.
                    y_t = torch.argmax(y_t, dim=2)  # of size (S,B)???? CHECK

                else:  # for the 2nd+ blocks:
                    embedded = self.embedding(y_t)
                    y_t, (h_t, c_t) = self.lstm(embedded, (h_t, c_t))
                    y_t = self.out_fc(y_t)  # y_t = (seq,batch,vocsize) - holds scores per word in voc.
                    y_t = torch.argmax(y_t, dim=2)  # of size (S,B)???? CHECK

                generated_sentence += y_t.tolist()

            return generated_sentence


def train_model(model, dl_train, optimizer, loss_fn, clip_grad=1.):
    losses = []
    with tqdm.tqdm(total=len(dl_train), file=sys.stdout) as pbar:

        model.train()

        for idx_batch, batch in enumerate(dl_train, start=1):

            x, x_len = batch.d  # x.shape = Seq,Batch

            # Forward pass:
            y_hat = model(x)
            # S,B,V = y_hat.shape

            # Now we want to estimate: X2,...,Xk to Y1,...,Yk-1.
            y_gt = x[1:, :]  # drop <sos> in every sequence..
            y_hat = y_hat[:(y_hat.shape[0] - 1), :, :]  # grab only y1..yk-1 from every output sequence
            S, B, V = y_hat.shape

            # set it in the right dimensions for CrossEntropyLoss:
            y_gt = y_gt.reshape(S * B)
            y_hat = y_hat.reshape(S * B, V)

            # calculate loss now:
            optimizer.zero_grad()
            loss = loss_fn(y_hat, y_gt)
            loss.backward()

            # Prevent large gradients
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            # Update parameters
            optimizer.step()
            losses.append(loss.item())
            pbar.update();
            pbar.set_description(f'train loss={losses[-1]:.8f}')

        mean_loss = torch.mean(torch.FloatTensor(losses))
        pbar.set_description(f'train loss={mean_loss:.8f}')

    return [mean_loss]


def evaluate_model(model, dl_data, loss_func):
    accuracies = []
    losses = []
    with tqdm.tqdm(total=len(dl_data), file=sys.stdout) as pbar:
        model.eval()

        for idx_batch, batch in enumerate(dl_data, start=1):
            x, x_len = batch.d

            y_hat = model(x)

            # Now we want to estimate: X2,...,Xk to Y1,...,Yk-1. (both in acc and loss calculations)
            y_gt = x[1:, :]  # drop <sos> in every sequence..
            y_hat = y_hat[:(y_hat.shape[0] - 1), :, :]  # grab only y1..yk-1 from every output sequence

            # set it in the right dimensions for CrossEntropyLoss:
            S, B, V = y_hat.shape
            y_gt_reshaped = y_gt.reshape(S * B)
            y_hat_reshaped = y_hat.reshape(S * B, V)
            loss = loss_func(y_hat_reshaped, y_gt_reshaped)
            losses.append(loss.item())

            # for acc calculations..:
            y_hat = torch.argmax(y_hat, dim=2)  # greedy-sample(meaning take best of each Yi..) (S, B, V) -> (S,B)

            # JUST FOR CHECKINGS -- TO BE REMOVED...
            # if idx_batch == 1:
            #     torch.set_printoptions(profile="full")
            #     print(y_gt)
            #     print(y_hat)

            # Don't calculate <pad> words (in the accuracy calculations):
            y_gt_indexs = y_gt != 1
            # Compare prediction to ground truth
            accuracies.append(torch.sum(y_gt[y_gt_indexs] == y_hat[y_gt_indexs]) / (torch.sum(y_gt_indexs == True)))
            pbar.update();
            pbar.set_description("eval acc={}, loss={}".format(accuracies[-1], losses[-1]))

        # take mean of losses & acc's
        mean_acc = torch.mean(torch.FloatTensor(accuracies))
        mean_loss = torch.mean(torch.FloatTensor(losses))
        pbar.set_description("mean acc={}, mean loss={}".format(mean_acc, mean_loss))

    return [mean_acc], [mean_loss]


def generate_words(model_path, empty_model, vocab):
    # load model parameters:
    model = empty_model
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    hidden_size = model.hidden_size
    layers = model.num_layers
    batch_size = 10

    # Create initial h_0,c_0 and words:
    init_words = torch.randint(5, 9700, (1, batch_size))  # (S=1,B). s=1 since it's 1st word only
    init_h0 = torch.randn(layers, batch_size, hidden_size)  # (layers, B, H)
    init_c0 = torch.randn(layers, batch_size, hidden_size)

    tokenized_gen_words = model(init_words, "teacher_force", init_h0, init_c0)  # (Generated_S=50, B)

    # untokenize words:
    for i in range(50):  # 50 is the maximum generated sequence size
        for j in range(batch_size):
            tokenized_gen_words[i][j] = vocab.itos[tokenized_gen_words[i][j]]

    # print generated sentences:
    tokenized_gen_words = np.array(tokenized_gen_words)
    tokenized_gen_words = np.transpose(tokenized_gen_words)
    for i in range(50):
        print("sentence{i} = ".format(i))
        print(np.array2string(tokenized_gen_words[i]))

    i = 0


# ------------------MAINNNNNN--------------
if __name__ == "__main__":

    # Some variables:
    BATCH_SIZE = 16  # keep
    EMB_DIM = 256  # improve
    HID_DIM = 1024  # imrpv
    NUM_LAYERS = 2
    GRAD_CLIP = 1.
    EPOCHS = 2
    # BATCHES_PER_EPOCH=200 #remove
    MODEL_PATH = 'model_2.pt'

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print(device)

    # Grab datasets, create model,optimizer and loss function:
    dl_train, dl_valid, dl_test, field = load_datasets(BATCH_SIZE)
    vocab_length = len(field.vocab)
    model = WordPredictor(vocab_length, EMB_DIM, NUM_LAYERS, HID_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    PAD_TOKEN = field.vocab.stoi['<pad>']
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)


    ########## GENERATION ##########3
    # empty_model = WordPredictor(vocab_length,EMB_DIM,NUM_LAYERS,HID_DIM)
    # generate_words("model_1.pt", empty_model, field.vocab)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print("#parameters in model  = ", count_parameters(model))

    # Train the model:
    losses_train = []
    losses_val = []
    accuracies_train = []
    accuracies_val = []
    accuracy_test = None

    # Early stop parameters:
    min_val_loss = np.inf
    patience = 10
    best_epoch = 0  # will hold the epoch num where early-stop occured..
    counter_no_improve = 0  # counts  no-improvement in val loss epochs untill reaching "patience"
    # es_flag = False  # True if early-stop occured already during training (so we'll know the 1st occurence), False otherwise.

    for idx_epoch in range(EPOCHS):
        # Linearly decay amount of teacher forcing for the first 20 epochs (example)
        p_tf = 1 - min((idx_epoch / 20), 1)
        print(f'=== EPOCH {idx_epoch + 1}/{EPOCHS}, p_tf={p_tf:.2f} ===')

        _ = train_model(model, dl_train, optimizer, loss_fn, GRAD_CLIP)

        # Evaluate model over train & validation sets:
        epoch_train_acc, epoch_train_loss = evaluate_model(model, dl_train, loss_fn)
        epoch_val_acc, epoch_val_loss = evaluate_model(model, dl_valid, loss_fn)

        accuracies_train += epoch_train_acc
        losses_train += epoch_train_loss
        accuracies_val += epoch_val_acc
        losses_val += epoch_val_loss

        # Early Stop:
        # Update early stopping:
        if epoch_val_loss[0] < min_val_loss:  # and es_flag == False:  # if new val-min found..
            print("Current best model found in epoch = ", idx_epoch + 1)
            min_val_loss = epoch_val_loss[0]
            best_epoch = idx_epoch + 1
            counter_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            counter_no_improve += 1

        if counter_no_improve >= patience:
            # es_flag = True
            break

    accuracy_test = evaluate_model(model, dl_test, loss_fn)[0]

    # SAVE MODEL:
    torch.save(model.state_dict(), MODEL_PATH)

    # Plot accuracy & losses for train,val and test sets:
    plt.plot(accuracies_train, label='Train accuracy')
    plt.plot(accuracies_val, label='Validation accuracy')
    plt.plot(EPOCHS - 1, accuracy_test, marker='o', markersize=3, color='red', label='Final Test accuracy')
    plt.axvline(best_epoch, linestyle='--', color='r',
                label='Early Stopping Checkpoint')  # plot the line indicating the early-stop point
    plt.title("Model Accuracies")
    plt.grid(True)
    plt.plot()
    plt.legend()
    plt.show()
    # Plot losses:
    plt.plot(losses_train, label='Train loss')
    plt.plot(losses_val, label='Val loss')
    plt.axvline(best_epoch, linestyle='--', color='r',
                label='Early Stopping Checkpoint')  # plot the line indicating the early-stop point
    plt.title("Model losses")
    plt.grid(True)
    plt.plot()
    plt.legend()
    plt.show()
