import numpy as np
import torch

from beam_Search import beam_search
from predictor_module import MODEL_FILE_NAME, DEVICE


def generate_words(empty_model, batch_size, vocab, dl_test):
    MAX_GEN_SEQ_LENGTH = 50

    model = empty_model
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_FILE_NAME))
    else:
        model.load_state_dict(torch.load(MODEL_FILE_NAME, map_location='cpu'))

    model.eval()

    hidden_size = model.hidden_size
    layers = model.num_layers

    # create initial h_0,c_0 and words:
    init_words = torch.randint(4, len(vocab) - 1, (1, batch_size)).to(DEVICE)
    init_h0 = torch.randn(layers, batch_size, hidden_size).to(DEVICE)
    init_c0 = torch.randn(layers, batch_size, hidden_size).to(DEVICE)

    # grab a single batch from Test Set
    gt_batch = next(iter(dl_test))
    gt_batch, batch_len = gt_batch.d
    gt_batch = gt_batch[:, :MAX_GEN_SEQ_LENGTH]

    # Case A: Generates by previous
    generates_by_previous(MAX_GEN_SEQ_LENGTH, batch_size, init_words, init_h0, init_c0, model, vocab)

    # Case B: Generates by previous
    generating_by_Ground_Truth(batch_size, gt_batch, init_c0, init_h0, model, vocab)

    # Case C: Generates by Beam-Search

    # initialize sequence that starts with <SOS>
    init_words = torch.randint(4, len(vocab) - 1, (1, 1)).to(DEVICE)
    init_words[:, :] = 2
    init_h0 = torch.randn(layers, 1, hidden_size).to(DEVICE)
    init_c0 = torch.randn(layers, 1, hidden_size).to(DEVICE)

    generating_by_beam_search(MAX_GEN_SEQ_LENGTH, batch_size, model, vocab, init_words, init_h0, init_c0)


def generating_by_beam_search(max_gen_seq_length, batch_size, model, vocab, init_words, init_h0, init_c0):
    print("\nGenerates words based on Beam search: \n")
    BEAM_WIDTH = 5
    # Write generated sequences to file
    with open('generated_sequences.txt', 'a+', encoding="utf8") as f:
        f.write('\n####################################\n')
        f.write('Generates words based on Beam search:\n')
        f.write('#####################################\n')
        for i in range(batch_size):
            tokenized_gen_words = beam_search(max_gen_seq_length, model, BEAM_WIDTH, init_words, init_h0, init_c0)

            # Untokenize the generated sequence:
            for j in range(len(tokenized_gen_words)):
                tokenized_gen_words[j] = vocab.itos[tokenized_gen_words[j]]

            final_seq = " ".join(tokenized_gen_words)
            f.write(f'\nsentence {i + 1}:\n{final_seq}\n')


def generating_by_Ground_Truth(batch_size, gt_batch, init_c0, init_h0, model, vocab):
    print("\n Generates words based on Ground Truth: \n")

    tokenized_gen_words = gt_batch.tolist()

    # untokenize words:
    for i in range(len(tokenized_gen_words)):
        for j in range(batch_size):
            tokenized_gen_words[i][j] = vocab.itos[tokenized_gen_words[i][j]]

    # Print Ground Truth
    with open('generated_sequences.txt', 'a+', encoding="utf8") as f:
        f.write('\n#####################################\n')
        f.write('Generates words based on Ground Truth:\n')
        f.write('######################################\n')

        f.write('\n----- Ground Truth sequences: ------\n')
        tokenized_gen_words = np.array(tokenized_gen_words)
        tokenized_gen_words = tokenized_gen_words.T
        for i in range(batch_size):
            final_seq = np.array2string(tokenized_gen_words[i])
            final_seq = final_seq.split("<eos>", 1)[0]
            f.write(f'\nsentence {i + 1}:\n{final_seq}\n')

        # Generate sentences:
        tokenized_gen_words_scores = model(gt_batch, "generating_by_Ground_Truth", init_c0, init_h0)
        tokenized_gen_words_scores = tokenized_gen_words_scores[:(tokenized_gen_words_scores.shape[0] - 1), :, :]
        tokenized_gen_words = torch.argmax(tokenized_gen_words_scores, dim=2)
        tokenized_gen_words = tokenized_gen_words.tolist()

        # Untokenize words:
        for i in range(len(tokenized_gen_words)):
            for j in range(batch_size):
                tokenized_gen_words[i][j] = vocab.itos[tokenized_gen_words[i][j]]

        # Print Generated sentences:
        tokenized_gen_words = np.array(tokenized_gen_words)
        tokenized_gen_words = tokenized_gen_words.T
        f.write('\n----- Generated sequences: ------\n')
        for i in range(batch_size):
            final_seq = np.array2string(tokenized_gen_words[i])
            f.write(f'\nsentence {i + 1}:\n{final_seq}\n')


def generates_by_previous(max_gen_seq_length, batch_size, init_words, init_h0, init_c0, model, vocab):
    print("\n Generates words based on previous one: \n")

    tokenized_gen_words = model(init_words, "generates_by_previous", init_h0, init_c0)  # (Generated_S=50, B)

    # Untokenize words:
    for i in range(max_gen_seq_length):
        for j in range(batch_size):
            tokenized_gen_words[i][j] = vocab.itos[tokenized_gen_words[i][j]]

    # Write generated sequences to file
    with open('generated_sequences.txt', 'a+', encoding="utf8") as f:
        f.write('\n#####################################\n')
        f.write('Generates words based on previous one:\n')
        f.write('######################################\n')
        tokenized_gen_words = np.array(tokenized_gen_words)
        tokenized_gen_words = tokenized_gen_words.T
        for i in range(batch_size):
            final_seq = np.array2string(tokenized_gen_words[i])
            f.write(f'\nsentence {i + 1}:\n{final_seq}\n')
