import heapq
from queue import Queue
import numpy as np
import torch


class BeamSearchNode(object):
    def __init__(self, log_prob, tokenized_words):
        self.log_prob = log_prob
        self.minus_log_prob = -log_prob
        self.tokenized_words = tokenized_words

    def __lt__(self, other):
        return self.minus_log_prob < other.minus_log_prob


def beam_search(max_gen_seq_length, model, beam_width, init_words, init_h0, init_c0):
    # queue to hold the next beam-front
    queue = Queue()

    # create Min-heap
    heap = []
    heapq.heapify(heap)

    softmax = torch.nn.LogSoftmax(dim=2)
    eos_flag = False

    # Generate 1st words:
    first_words = model(init_words, "beam_search", init_h0, init_c0)
    first_words_probabilities = softmax(first_words)
    vocab_length = first_words_probabilities.shape[2]

    # Create beam search_objects according to scores, and insert to the heap
    for i in range(vocab_length):
        tokenized_word = [i]
        word_prob = first_words_probabilities[0, 0, i]
        new_node = BeamSearchNode(word_prob, tokenized_word)
        heapq.heappush(heap, new_node)

    # Grab top BEAM_WIDTH elements to proceed with
    for i in range(beam_width):
        node = heapq.heappop(heap)
        queue.put(node)

    # Perform the rest of the beam-search:
    for i in range(max_gen_seq_length):
        # Clean heap:
        heap = []
        heapq.heapify(heap)

        # Insert new elements to the heap,
        # every queue-node which represents current best sentences, holds ~10k elements
        while not queue.empty():
            q_node = queue.get()

            # add a Batch-dimension so we can run it on model
            x = [2] + q_node.tokenized_words
            x = torch.IntTensor(x)
            x = torch.unsqueeze(x, 1)

            # Generate the i-th-stage words:
            ith_words = model(x, "beam_search", init_h0, init_c0)
            ith_words = ith_words[-1, :, :]  # only the last Y is relevant...
            ith_words = torch.unsqueeze(ith_words, 0)  # retrieve the 1st dimension lost from previous line ^
            ith_words_probabilities = softmax(ith_words)

            # Create beam search_objects according to scores, and insert to heap
            for j in range(vocab_length):
                # adjust current word with previous words
                tokenized_word = q_node.tokenized_words + [j]

                # adjust current score with previous scores
                word_prob = q_node.log_prob + ith_words_probabilities[0, 0, j]

                new_node = BeamSearchNode(word_prob, tokenized_word)
                heapq.heappush(heap, new_node)

        # Check stop-case (if most-likely sentence holds <eos> as his last generated word)
        check_node = heap[0]
        if check_node.tokenized_words[-1] == 3:
            eos_flag = True
            break

        # Grab top BEAM_WIDTH elements to proceed with
        for j in range(beam_width):
            node = heapq.heappop(heap)
            queue.put(node)

    if eos_flag:
        node = heapq.heappop(heap)
        final_sentence = node.tokenized_words
        return final_sentence

    # increase reached a generated sentence of size 50
    else:
        best_node = None
        best_prob = -np.inf

        while not queue.empty():
            node = queue.get()
            if node.log_prob > best_prob:
                best_node = node

        if best_node is None:
            return node.tokenized_words
        return best_node.tokenized_words

