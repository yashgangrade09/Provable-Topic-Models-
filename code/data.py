from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.io import savemat, loadmat
import os


class Vocab(dict):
    """Thin wrapper over python dict"""

    def read_vocab_file(self, vocab_file):
        """
        Read vocabulary from file

        :param vocab_file: A file object that contains the vocabulary
        :type vocab_file: _io.TextIO
        """

        for idx, line in enumerate(vocab_file):
            word = line.strip()
            # Create a bi-dict
            self[word] = idx
            self[idx] = word

    def __getitem__(self, key):
        # Return None is key is not found in vocabulary
        return dict.get(key, None)


def process_docword(filepath, save_format='npz'):
    """
    Read the docword file specified by filepath and save a sparse csr matrix
    :param filepath: path to the bag of words file in the format specified
        here: https://archive.ics.uci.edu/ml/datasets/bag+of+words
    :type filepath: str
    :param save_format: Either of {'npz', 'mat'}. npz stores numpy and mat stores MATLAB style matrices
    :type save_format: str
    """

    savepath = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    docword_file = open(filepath, 'r')

    # Read the first three lines containing number of documents, words and non-zero entries
    n_docs = int(docword_file.readline())
    n_words = int(docword_file.readline())
    _ = int(docword_file.readline())

    data, row_ind, col_ind = [], [], []

    for line in docword_file:
        doc_id, word_id, count = [int(x) for x in line.rstrip().split()]
        data.append(count)
        row_ind.append(doc_id - 1)                      # Since doc_ids and word_ids start from
        col_ind.append(word_id - 1)                     # 1 instead of 0, shift left by one

    # Create a compressed sparse row matrix
    docword_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n_docs, n_words))

    # Save the matrix to a file at the same path as filepath
    if save_format == 'mat':
        save_filepath = os.path.join(savepath, filename.replace('txt', 'mat'))
        savemat(save_filepath, {'docword': docword_matrix}, oned_as='column')
    elif save_format == 'npz':
        save_filepath = os.path.join(savepath, filename.replace('txt', 'npz'))
        save_npz(save_filepath, docword_matrix)

    docword_file.close()


def get_word_doc_matrix(path):
    mat_format = path[-3:]
    if mat_format == 'npz':
        return load_npz(path)
    elif mat_format == 'mat':
        return loadmat(path)['docword']
