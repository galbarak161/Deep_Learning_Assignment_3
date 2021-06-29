import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_FILE_NAME = 'LTSM_model.pt'


class LSTM_Predictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, h_dim, ignore_index, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = h_dim

        # Embedding layer to converts from token index to dense tensor
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # PyTorch multilayer LSTM
        self.lstm = nn.LSTM(embedding_dim, h_dim, num_layers=num_layers, dropout=dropout)

        # Output layer
        self.out_fc = nn.Linear(h_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self.optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.999), eps=1e-08)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.to(DEVICE)

    def forward(self, x, forward_mode="train", initial_vec_h0=None, initial_vec_c0=None):
        # this will be used upon training/generating by input
        if forward_mode == "train" or forward_mode == "gen_by_input":
            # Embed the input:
            embedded = self.embedding(x)

            # Run through the LSTMs
            output, _ = self.lstm(embedded)

            # Project H back to the vocab size V, to get a score per word
            output = functional.relu(output)
            return self.out_fc(output)

        # used upon generation by teacher_forcing
        if forward_mode == "teacher_force":
            embedded, y_t, h_t, c_t = None, None, None, None
            generated_sentence = []
            max_words_generated = 50

            for i in range(max_words_generated):
                if i == 0:  # for the 1st block:
                    embedded = self.embedding(x)
                    y_t, (h_t, c_t) = self.lstm(embedded, (initial_vec_h0, initial_vec_c0))
                    y_t = self.out_fc(y_t)
                    y_t = torch.argmax(y_t, dim=2)

                else:  # for the 2nd+ blocks:
                    embedded = self.embedding(y_t)
                    y_t, (h_t, c_t) = self.lstm(embedded, (h_t, c_t))
                    y_t = self.out_fc(y_t)
                    y_t = torch.argmax(y_t, dim=2)

                generated_sentence += y_t.tolist()

            return generated_sentence

    def train_model(self, epochs: int, train_data_loader, valid_data_loader, test_data_loader):
        losses_train = []
        losses_val = []
        accuracies_train = []
        accuracies_val = []

        # Early stop parameters:
        min_val_loss = np.inf
        patience = 10
        best_epoch = 0
        early_stop_counter = 0

        for idx_epoch in range(epochs):
            print(f'epoch number {idx_epoch + 1}')
            self.train_model_step(train_data_loader)

            # evaluate model over train and validation sets:
            epoch_train_acc, epoch_train_loss = self.evaluate_model(train_data_loader)
            epoch_val_acc, epoch_val_loss = self.evaluate_model(valid_data_loader)

            accuracies_train += epoch_train_acc
            losses_train += epoch_train_loss
            accuracies_val += epoch_val_acc
            losses_val += epoch_val_loss

            # update early stopping:
            if epoch_val_loss[0] < min_val_loss:
                print(f'Current best model found in epoch = {idx_epoch + 1}')
                min_val_loss = epoch_val_loss[0]
                best_epoch = idx_epoch + 1
                early_stop_counter = 0
                torch.save(self.state_dict(), MODEL_FILE_NAME)
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                break

        print('Evaluate model...')
        test_acc, test_loss = self.evaluate_model(test_data_loader)
        self.load_state_dict(torch.load(MODEL_FILE_NAME))

        # Plot accuracy & losses for train,val and test sets:
        fig = plt.figure()
        plot_title = "Model Accuracies"
        plt.plot(accuracies_train, label='Train accuracy')
        plt.plot(accuracies_val, label='Validation accuracy')
        plt.plot(epochs - 1, test_acc, marker='o', markersize=3, color='red', label='Final Test accuracy')
        plt.axvline(best_epoch, linestyle='--', color='r', label='Early Stopping Checkpoint')
        plt.title(plot_title)
        plt.plot()
        plt.legend()
        fig.savefig(plot_title)
        plt.close(fig)
        plt.clf()

        # Plot losses:
        fig = plt.figure()
        plot_title = "Model losses"
        plt.plot(losses_train, label='Train loss')
        plt.plot(losses_val, label='Validation loss')
        plt.axvline(best_epoch, linestyle='--', color='r', label='Early Stopping Checkpoint')
        plt.title(plot_title)
        plt.plot()
        plt.legend()
        fig.savefig(plot_title)
        plt.close(fig)
        plt.clf()

        # final results
        print(f'Train: Accuracy = {(accuracies_train[best_epoch] * 100):.2f}%, '
              f'Avg Loss = {losses_train[best_epoch]:.2f}')
        print(f'Validation: Accuracy = {(accuracies_val[best_epoch] * 100):.2f}%, '
              f'Avg Loss = {losses_val[best_epoch]:.2f}')
        print(f'Test: Accuracy = {(test_acc * 100):.2f}%, Avg Loss = {test_loss:.2f}')
        print()

    def train_model_step(self, train_data_loader, grad_clip=1.):
        losses = []
        self.train()

        for idx_batch, batch in enumerate(train_data_loader, start=1):
            x, x_len = batch.d
            #x = x.to(DEVICE)

            y_hat = self(x)#.to(DEVICE)

            # estimate: X2,...,Xk to Y1,...,Yk-1.
            y_gt = x[1:, :]
            y_hat = y_hat[:(y_hat.shape[0] - 1), :, :]
            S, B, V = y_hat.shape

            y_gt = y_gt.reshape(S * B)
            y_hat = y_hat.reshape(S * B, V)

            # calculate loss
            self.optimizer.zero_grad()
            loss = self.loss_func(y_hat, y_gt)#.to(DEVICE)
            loss.backward()

            # prevent large gradients
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

            self.optimizer.step()
            losses.append(loss.item())

        return [np.mean(np.asarray(losses))]

    def evaluate_model(self, data_loader):
        accuracies = []
        losses = []

        self.eval()
        with torch.no_grad():
            for idx_batch, batch in enumerate(data_loader, start=1):
                x, x_len = batch.d
                #x = x.to(DEVICE)

                y_hat = self(x)#.to(DEVICE)

                y_gt = x[1:, :]
                y_hat = y_hat[:(y_hat.shape[0] - 1), :, :]
                S, B, V = y_hat.shape
                y_gt_reshaped = y_gt.reshape(S * B)
                y_hat_reshaped = y_hat.reshape(S * B, V)

                loss = self.loss_func(y_hat_reshaped, y_gt_reshaped)#.to(DEVICE)
                losses.append(loss.item())

                # we don't calculate <pad> words
                y_hat = torch.argmax(y_hat, dim=2)
                y_gt_indexes = y_gt != 1
                accuracies.append(
                    torch.sum(y_gt[y_gt_indexes] == y_hat[y_gt_indexes]) / (torch.sum(y_gt_indexes == True))
                )

        mean_acc = torch.mean(torch.FloatTensor(accuracies))
        mean_loss = torch.mean(torch.FloatTensor(losses))

        return [mean_acc], [mean_loss]
