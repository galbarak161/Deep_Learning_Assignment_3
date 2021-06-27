from pennTreebank_dataset import load_datasets
from predictor_module import LSTM_Predictor, DEVICE
from words_generator import generate_words


def main():
    batch_size = 16

    # create data loaders from dataset
    print('Create data loaders from dataset...')
    dl_train, dl_valid, dl_test, field = load_datasets(batch_size=batch_size)
    vocab_length = len(field.vocab)
    pad_tokens = field.vocab.stoi['<pad>']

    epochs = 100
    print(f'Train model with {epochs}...')
    model = LSTM_Predictor(vocab_length, embedding_dim=256, num_layers=2, h_dim=1024, ignore_index=pad_tokens)
    model.train_model(epochs, dl_train, dl_valid, dl_test)

    print('Generate words...')
    empty_model = LSTM_Predictor(vocab_length, embedding_dim=256, num_layers=2, h_dim=1024, ignore_index=pad_tokens)
    generate_words(empty_model, field.vocab, batch_size)


if __name__ == "__main__":
    main()
