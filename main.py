from datetime import time

from torchsummary import summary

from pennTreebank_dataset import load_datasets
from predictor_module import LSTM_Predictor, DEVICE
from words_generator import generate_words

def print_time(time_taken: float) -> None:
    """
    Utility function for time printing
    :param time_taken: the time we need to print
    """
    hours, rem = divmod(time_taken, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\tTime taken: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))

def main():
    batch_size = 16

    # create data loaders from dataset
    print('Create data loaders from dataset...')
    dl_train, dl_valid, dl_test, field = load_datasets(batch_size=batch_size)
    vocab_length = len(field.vocab)
    pad_tokens = field.vocab.stoi['<pad>']

    epochs = 50
    start_time = time.time()
    model = LSTM_Predictor(
        vocab_length, embedding_dim=128, num_layers=2, h_dim=256, ignore_index=pad_tokens
    ).to(DEVICE)
    print(model)
    print(f'\nTrain model using {epochs} epochs...')
    model.train_model(epochs, dl_train, dl_valid, dl_test)
    end_time = time.time()
    print_time(end_time - start_time)

    print('Generate words...')
    empty_model = LSTM_Predictor(
        vocab_length, embedding_dim=128, num_layers=2, h_dim=256, ignore_index=pad_tokens
    ).to(DEVICE)

    generate_words(empty_model, field.vocab, batch_size)


if __name__ == "__main__":
    main()
