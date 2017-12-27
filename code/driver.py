from data import *
from utils import benchmark
from algo_components import *

cutoff = 100
threshold = 100
num_topics = 20

with benchmark("End-to-end") as complete:
    # with benchmark('Reading vocab file') as vf:
    #     v = Vocab()
    #     with open('../data/nips/vocab.nips.txt', 'r') as v_file:
    #         v.read_vocab_file(v_file)

    # with benchmark('Storing document-word matrix') as save_dwm:
    #     process_docword('../data/nips/docword.nips.txt', save_format='npz')

    with benchmark('Loading document-word matrix') as load_dwm:
        M = get_word_doc_matrix('../data/nips/docword.nips.npz')
        M, vocab = truncate_vocab(M, '../data/nips/vocab.nips.txt', cutoff)

    with benchmark('Building the Q matrix') as q_mat:
        Q = get_word_word_cormat(M)

    with benchmark('Finding anchor words') as anchor_find:
        candidates = get_candidate_anchors(M, threshold)
        anchor_indices = fast_anchor_words(Q, num_topics, 0.1, candidates)

    with benchmark('Recover KL') as rec_kl:
        A, topic_likelihoods = recover_kl(Q, anchor_indices, 1e-7)
        # A = recover_orig(Q, anchor_indices)
        num_topics = 20
        top_words=10
        for k in range(num_topics):
            topwords = np.argsort(A[:, k])[-top_words:][::-1]
            print(vocab[anchor_indices[k]], ": ", end=' ')
            for w in topwords:
                print(vocab[w], end=' ')
            print()
