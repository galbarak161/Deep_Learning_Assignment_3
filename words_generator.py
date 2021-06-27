import numpy as np
import torch

from predictor_module import MODEL_FILE_NAME
from predictor_module import DEVICE


def generate_words(empty_model, vocab, batch_size, max_generated_sequence=50):
    # load model parameters
    model = empty_model
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_FILE_NAME))
    else:
        model.load_state_dict(torch.load(MODEL_FILE_NAME, map_location='cpu'))

    hidden_size = model.hidden_size
    layers = model.num_layers

    # create initial h_0,c_0 and words:
    init_words = torch.randint(5, 9700, (1, batch_size)).to(DEVICE)
    init_h0 = torch.randn(layers, batch_size, hidden_size).to(DEVICE)
    init_c0 = torch.randn(layers, batch_size, hidden_size).to(DEVICE)

    tokenized_gen_words = model(init_words, "teacher_force", init_h0, init_c0).to(DEVICE)

    # untokenize words:
    for i in range(max_generated_sequence):
        for j in range(batch_size):
            tokenized_gen_words[i][j] = vocab.itos[tokenized_gen_words[i][j]]

    tokenized_gen_words = np.array(tokenized_gen_words)
    tokenized_gen_words = np.transpose(tokenized_gen_words)
    for i in range(max_generated_sequence):
        print(f'sentence{i+1} = ')
        print(np.array2string(tokenized_gen_words[i]))
