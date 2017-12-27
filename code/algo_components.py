import numpy as np
from sklearn.random_projection import GaussianRandomProjection
import scipy.io
import scipy.sparse
from tqdm import tqdm
import math


def truncate_vocab(doc_word_matrix, vocab_file, threshold):
    """
    Truncate the original vocab to remove stopwords and infrequent words

    :param doc_word_matrix: document word matrix in sparse csr format
    :type doc_word_matrix: scipy.sparse.csr_matrix
    :param vocab_file: path of the vocabulary file
    :type: string
    :param threshold: minimum number of documents that a word must occur in
    :type: int
    :return: truncated document-word matrix
    :rtype: scipy.sparse.csr_matrix
    """

    table = dict()
    num = 0
    with open(vocab_file, 'r') as file:
        for l in file:
            table[l.rstrip()] = num
            num += 1

    remove = [False] * num

    with open('../data/nips/stopwords.txt', 'r') as file:
        for l in file:
            if l.rstrip() in table:
                remove[table[l.rstrip()]] = True

    M = doc_word_matrix.T

    if M.shape[0] != num:
        print("Error in sizes")

    M = M.tocsr()

    new_indptr = np.zeros(M.indptr.shape[0], dtype=np.int32)
    new_indices = np.zeros(M.indices.shape[0], dtype=np.int32)
    new_data = np.zeros(M.data.shape[0], dtype=np.float64)

    indptr_counter = 1

    for i in range(M.indptr.size - 1):

        # if this is not a stopword
        if not remove[i]:

            # start and end indices for row i
            start = M.indptr[i]
            end = M.indptr[i + 1]

            # if number of distinct documents that this word appears in is >= cutoff
            if (end - start) >= threshold:
                new_indptr[indptr_counter] = new_indptr[indptr_counter - 1] + end - start
                new_data[new_indptr[indptr_counter - 1]:new_indptr[indptr_counter]] = M.data[start:end]
                new_indices[new_indptr[indptr_counter - 1]:new_indptr[indptr_counter]] = M.indices[start:end]
                indptr_counter += 1
            else:
                remove[i] = True

    new_indptr = new_indptr[0:indptr_counter]
    new_indices = new_indices[0:new_indptr[indptr_counter - 1]]
    new_data = new_data[0:new_indptr[indptr_counter - 1]]

    M = scipy.sparse.csr_matrix((new_data, new_indices, new_indptr))

    # Generate new vocab
    vocab = {}
    counter = 0
    with open(vocab_file, 'r') as v:
        for idx, line in enumerate(v):
            if not remove[idx]:
                vocab[counter] = line.rstrip()
                counter += 1

    return M.T.tocsr(), vocab


def random_projection(matrix, dim):
    return GaussianRandomProjection(n_components=dim).fit_transform(matrix)


# def random_projection(M, new_dim):
#     prng = np.random.RandomState()
#     old_dim = M[:, 0].size
#     p = np.array([1. / 6, 2. / 3, 1. / 6])
#     c = np.cumsum(p)
#     randdoubles = prng.random_sample(new_dim * old_dim)
#     R = np.searchsorted(c, randdoubles)
#     R = math.sqrt(3) * (R - 1)
#     R = np.reshape(R, (new_dim, old_dim))
#
#     M_red = np.dot(R, M)
#     return M_red.T


def get_word_word_cormat(doc_word_matrix):
    """
    Get the word-word correlation matrix from document word matrix

    :param doc_word_matrix: document word matrix generated from the corpus
    :type doc_word_matrix: scipy.sparse.csr_matrix
    :return: word-word correlation matrix
    :rtype: scipy.sparse.csr_matrix
    """

    # Finding the actual word-word correlation is infeasible.
    # Instead, we compute H.T * H - H^ s.t
    #               E[H.T * H - H^] = A * W * W.T * A.T
    # ---------------------------------------------------------
    #
    # Let H_d be the i-th row of the doc-word matrix, we
    # compute the normalized rows
    #               H_d = H_d / srqt(n_d * (n_d - 1))
    #
    # And diagonal matrix H_d^
    #               H_d^ = diag(H_d) / (n_d * (n_d - 1))
    # where n_d is the number of words in this document
    # ---------------------------------------------------------
    #
    # H is then collection of all H_d and H^ is sum of all
    # diagonal matrices

    n_docs = doc_word_matrix.shape[0]
    num_words_in_doc = np.array(np.sum(doc_word_matrix, axis=1))

    H_diag = np.squeeze(np.asarray(np.sum(doc_word_matrix / (num_words_in_doc * (num_words_in_doc - 1)), axis=0)))
    H = doc_word_matrix / np.sqrt(num_words_in_doc * (num_words_in_doc - 1))

    return np.asarray((H.T * H - np.diag(H_diag))) / n_docs


def get_candidate_anchors(doc_word_matrix, threshold):
    """
    Get candidate anchor words. Candidate anchor words are those that occur in more
    than `threshold` number of documents

    :param doc_word_matrix: document word matrix in sparse csr format
    :type doc_word_matrix: scipy.sparse.csr_matrix
    :param threshold: number of documents that a candidate has to be in
    :type threshold: int
    :return: indices (in vocabulary) of candidate anchor words
    :rtype: numpy.ndarray
    """

    candidate = []

    for idx, i in enumerate(np.count_nonzero(doc_word_matrix.toarray(), axis=0)):
        if i > threshold:
            candidate.append(idx)
    return np.array(candidate)


def fast_anchor_words(word_word_cormat, num_topics, epsilon, candidates):
    """
    Implements the FastAnchorWords (Algorithm 4) from the paper.

    :param word_word_cormat: word-word correlation matrix
    :type word_word_cormat: scipy.sparse.csr_matrix
    :param num_topics: Number of topics in the model
    :type num_topics: int
    :param epsilon: tolerance
    :type epsilon: float
    :param candidates: indices (in the vocabulary) of the candidate anchor words
    :type candidates: list of int or numpy.ndarray
    :return: indices of the anchor words
    :rtype: numpy.ndarray
    """

    # Normalize the rows of word-word correlation matrix
    Q = word_word_cormat.copy()                     # copy() so that we don't modify the original
    Q = Q / np.sum(Q, axis=1)[:, np.newaxis]
    # Q = word_word_cormat    # just take the original matrix for now

    # TODO:
    # The dimension of random subspace onto which we project the correlation
    # matrix. This doesn't work for some reason. Hmmmm....

    # rproj_dim = int((4 * np.log(v_size)) / (epsilon ** 2))

    # Fix this sometime soon
    random_projection_dim = 30

    Q_proj = random_projection(Q, random_projection_dim)

    # Make a copy of the projected matrix
    # Only consider the row corresponding to candidate anchors
    Q_proj = Q_proj[candidates]
    Q_aux = Q_proj.copy()

    anchor_words = np.zeros((num_topics, random_projection_dim))
    anchor_indices = np.zeros(num_topics, dtype=int)
    basis = np.zeros((num_topics - 1, random_projection_dim))

    # Find the point (out of the vectors corresponding
    # to candidate anchors) which is farthest from the origin
    farthest_idx = np.argmax([np.dot(x, x) for x in Q_aux])
    anchor_words[0] = Q_proj[farthest_idx]
    anchor_indices[0] = candidates[farthest_idx]

    # Center the vectors of the random projection with
    # vector corresponding farthest_idx as origin
    Q_aux = Q_aux - anchor_words[0]

    # Now find the next farthest point
    second_farthest_idx = np.argmax([np.dot(x, x) for x in Q_aux])
    anchor_words[1] = Q_proj[second_farthest_idx]
    anchor_indices[1] = candidates[second_farthest_idx]
    vector = Q_aux[second_farthest_idx]
    basis[0] = vector / np.sqrt(np.dot(vector, vector))

    # Iteratively find remaining K - 2 anchor words,
    for i in range(1, num_topics - 1):
        max_dist = 0
        # Apply gram-schmidt to find the spans
        for row_idx in range(len(candidates)):
            Q_aux[row_idx] -= np.dot(Q_aux[row_idx], basis[i - 1]) * basis[i - 1]
            dist = np.dot(Q_aux[row_idx], Q_aux[row_idx])

            if dist > max_dist:
                max_dist = dist
                anchor_words[i + 1] = Q_proj[row_idx]
                anchor_indices[i + 1] = candidates[row_idx]
                basis[i] = Q_aux[row_idx] / np.sqrt(dist)

    return anchor_indices


def recover_orig(word_word_cormat, anchors):
    K = len(anchors)
    Q = word_word_cormat

    permutation = list(anchors) + [x for x in range(Q.shape[1]) if x not in anchors]
    Q_prime = Q[permutation, :]
    Q_prime = Q_prime[:, permutation]

    DRD = Q_prime[0:K, 0:K]
    DR_AT = Q_prime[0:K, :]
    DR1 = np.dot(DR_AT, np.ones(DR_AT.shape[1]))

    z = np.linalg.solve(DRD, DR1)
    A = np.dot(np.linalg.inv(np.dot(DRD, np.diag(z))), DR_AT).transpose()

    reverse_permutation = [permutation.index(p) for p in permutation]
    A = A[reverse_permutation]

    return A


def kl_divergence(p, q):
    """
    Find the KL divergence between vector p and q
    KL(p, q) = p * log(p/q)
    """
    divergence = np.dot(p, np.log(p) - np.log(q))

    if divergence < 0 or np.isnan(divergence):
        raise ValueError("KL divergence was found to be less than 0.")

    return divergence


def logsum_exp(y):
    m = y.max()
    return m + np.log((np.exp(y - m)).sum())


def exponentiated_gradient_solver(vec, mat, epsilon):
    """
    Find a vector x that minimizes d(b, T*x) where b = mat. Epsilon is
    the tolerance parameter

    :param vec: vector w.r.t to minimize
    :type vec: numpy.ndarray
    :param mat: matrix w.r.t. to minimize
    :type mat: numpy.ndarray
    :param epsilon: tolerance
    :type epsilon: float
    :return: vector x that minimizes d(vec, mat*x)
    :rtype: numpy.ndarray
    """

    #  Number of topics
    K = mat.shape[0]

    # Initialize each component of x = 1/K
    alpha = np.ones(K) / K

    # Clip input vectors to values between 0 to 1
    y = np.clip(vec, 0, 1)
    x = np.clip(mat, 0, 1)

    # keep only non-zero columns
    mask = list(np.nonzero(y)[0])
    y = y[mask]
    x = x[:, mask]

    # Increase all elements of x by very small amount
    # Possibly to avoid divide by zero errors
    x += 1e-9
    x = x / x.sum(axis=1)[:, np.newaxis]

    # Start the exponentiated gradient descent
    c1 = 1e-4
    c2 = 0.9
    log_alpha = np.log(alpha)

    # Project alpha onto x
    proj = np.dot(alpha, x)

    new_obj = kl_divergence(y, proj)
    y_over_proj = y / proj

    grad = -np.dot(x, y_over_proj.transpose())

    step_size = 1
    decreasing = False

    iteration_count = 1

    while 1:
        eta = step_size
        old_obj = new_obj
        old_alpha = np.copy(alpha)
        old_log_alpha = np.copy(log_alpha)

        old_proj = np.copy(proj)

        iteration_count += 1

        # Take a step in the gradient direction
        log_alpha -= eta * grad

        # Normalize to project onto simplex
        log_alpha -= logsum_exp(log_alpha)

        # compute new objective
        alpha = np.exp(log_alpha)
        proj = np.dot(alpha, x)
        new_obj = kl_divergence(y, proj)

        # If the new object is less than epsilon, then we are done
        if new_obj < epsilon:
            break

        # Check gradients
        grad_dot_deltaAlpha = np.dot(grad, alpha - old_alpha)
        # assert (grad_dot_deltaAlpha <= 1-9)

        if not new_obj <= old_obj + c1 * step_size * grad_dot_deltaAlpha:  # sufficient decrease
            # reduce stepsize
            step_size /= 2.0

            # If the step size becomes too small
            if step_size < 10 ** (-6):
                break

            alpha = old_alpha
            log_alpha = old_log_alpha
            proj = old_proj
            new_obj = old_obj
            decreasing = True
            continue

        # compute the new gradient
        old_grad = np.copy(grad)
        y_over_proj = y / proj
        grad = -np.dot(x, y_over_proj)

        if not np.dot(grad, alpha - old_alpha) >= c2 * grad_dot_deltaAlpha and not decreasing:  # curvature
            step_size *= 2.0  # increase stepsize
            alpha = old_alpha
            log_alpha = old_log_alpha
            grad = old_grad
            proj = old_proj
            new_obj = old_obj
            continue

        decreasing = False
        lam = np.copy(grad)
        lam -= lam.min()

        gap = np.dot(alpha, lam)
        convergence = gap
        if convergence < epsilon:
            break

    return alpha


def minimize_divergence(word_idx, Q_i, X, anchors, epsilon):
    """
    Return a vector C_i that minimizes D_kl(Q_i || \sum_{k \in anchors} C_i * Q_sk)
    subject to \sum_k C_i = 1 and C_k >= 0

    :param word_idx: index of the word
    :type word_idx: int
    :param Q_i: Row vector corresponding to i-th word in vocabulary
    :type Q_i: numpy.ndarray
    :param X: Matrix of anchor word vectors
    :type X: numpy.ndarray
    :param anchors: list of anchor indices
    :type anchors: numpy.ndarray
    :return: a vector C_i that minimizes that distance measure D_kl
    :rtype: numpy.ndarray
    """

    # Number of topics is equal to number of anchor words
    K = len(anchors)

    # If the word is an anchor-word, then the minimizer is
    # the word's vector itself
    if word_idx in anchors:
        C_i = np.zeros(K)
        C_i[list(anchors).index(word_idx)] = 1
        return C_i

    # Else find the vector that minimizes the distance measure
    # For now only minimize KL distance
    C_i = exponentiated_gradient_solver(Q_i, X, epsilon)

    # If EG returns a vector with NaNs in it, we return a 1/K
    if np.isnan(C_i).any():
        C_i = np.ones(K) / K

    return C_i


def recover_kl(word_word_cormat, anchors, epsilon):

    Q = word_word_cormat
    vocab_size = Q.shape[0]

    # Number of topics
    K = len(anchors)

    # Word-topic matrix
    A = np.zeros((vocab_size, K))

    # Row normalization constants for matrix Q
    p_w = np.diag(np.dot(Q, np.ones(vocab_size)))

    # Normalize the rows of Q
    Q = Q / np.sum(Q, axis=1)[:, np.newaxis]

    # Get rows corresponding to the anchor indices, Q_s in the paper
    X = Q[anchors, :]
    # Multiply X with it's transpose
    X_XT = np.dot(X, X.T)

    # For each word in vocabulary, solve the optimization
    # mentioned in Algorithm 3 of paper
    for i in tqdm(range(vocab_size), 'Word num'):
        Q_i = Q[i]
        A[i] = minimize_divergence(i, Q_i, X, anchors, epsilon)

    A = np.dot(p_w, A)

    # normalize columns of A. This is the normalization constant P(z)
    topic_likelihoods = A.sum(0)

    for k in range(K):
        A[:, k] = A[:, k] / A[:, k].sum()

    A = np.array(A)

    return A, topic_likelihoods
