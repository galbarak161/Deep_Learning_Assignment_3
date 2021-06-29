import numpy as np
import torch

from predictor_module import MODEL_FILE_NAME, DEVICE


def generates_word_based_on_previous(empty_model, vocab, batch_size):
    # load model parameters
    model = empty_model
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_FILE_NAME))
    else:
        model.load_state_dict(torch.load(MODEL_FILE_NAME, map_location='cpu'))

    hidden_size = model.hidden_size
    layers = model.num_layers

    # create initial h_0,c_0 and words:
    init_words = torch.randint(4, len(vocab) - 1, (1, batch_size)).to(DEVICE)
    init_h0 = torch.randn(layers, batch_size, hidden_size).to(DEVICE)
    init_c0 = torch.randn(layers, batch_size, hidden_size).to(DEVICE)

    tokenize_gen_sequences = model(init_words, "teacher_force", init_h0, init_c0)

    sequence_counter = 1
    with open('Generated Sequences.txt', 'w', encoding="utf8") as f:
        f.write('\nGenerates words based on previous one:\n')
        for sequence in tokenize_gen_sequences:
            untokenize_sequence = []
            f.write(f'\nSentence {sequence_counter}:\n')
            for token in sequence:
                word = vocab.itos[token]
                if word == '<eos>':
                    break
                untokenize_sequence.append(word)

            untokenize_sequence = np.array(untokenize_sequence).T
            f.write(f'{untokenize_sequence}\n')
            sequence_counter += 1

