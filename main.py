from Penn_Treebank_Dataset import load_datasets
from Predictor_Module import LSTM_Predictor, DEVICE
from words_generator import generate_words


def main():
    batch_size = 64

    # create data loaders from dataset
    dl_train, dl_valid, dl_test, field = load_datasets(batch_size=batch_size)
    vocab_length = len(field.vocab)
    pad_tokens = field.vocab.stoi['<pad>']

    epochs = 1
    model = LSTM_Predictor(vocab_length, embedding_dim=256, num_layers=2, h_dim=1024, ignore_index=pad_tokens)
    model.train_model(epochs, dl_train, dl_valid, dl_test)

    empty_model = LSTM_Predictor(vocab_length, embedding_dim=256, num_layers=2, h_dim=1024, ignore_index=pad_tokens)
    generate_words(empty_model, field.vocab, batch_size)


if __name__ == "__main__":
    print(DEVICE)
    main()
