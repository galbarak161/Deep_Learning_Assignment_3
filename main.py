import numpy as np
import torch
from Penn_Treebank_Dataset import load_datasets
from Predictor_Module import LSTM_Predictor, DEVICE


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


def main():
    # Grab datasets, create model,optimizer and loss function:
    dl_train, dl_valid, dl_test, field = load_datasets(batch_size=32)
    vocab_length = len(field.vocab)
    pad_tokens = field.vocab.stoi['<pad>']

    epochs = 2
    model = LSTM_Predictor(vocab_length, embedding_dim=256, num_layers=2, h_dim=1024, ignore_index=pad_tokens)
    model.train_model(epochs, dl_train, dl_valid, dl_test)


if __name__ == "__main__":
    print(DEVICE)
    main()
