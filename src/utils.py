import gzip
import numpy as np


def load_data(data_file):
    """Loads data from gzip pickled files -> convert to numpy arrays """

    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_tensor(train_set)
    valid_set_x, valid_set_y = make_tensor(valid_set)
    test_set_x, test_set_y = make_tensor(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def make_tensor(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    # data_x = torch.tensor(data_x)
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def load_pickle(f):
    """
    loads and returns the content of a pickle file
    it handles the inconsistencies betwwen the pickle packages available in Python 2 and Python 3"""
    # try:
    #     import cPickle as thepickle
    # except ImportError:
    import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret


def permutation_data(X, Y, label_X, label_Y, method='preserved'):
    """
    permutes the data and labels
    """

    if method == 'random':
        perm = np.random.permutation(X.shape[0])
    elif method == 'preserved':
        lenY = Y.shape[0]
        init = np.arange(lenY)
        perm = np.concatenate([np.random.permutation(
            init[i: min(i + 10, lenY)]) for i in range(0, lenY-1, 10)])

    Y = Y[perm]
    label_Y = label_Y[perm]
    Xs = [X, Y]
    alignT = np.stack([perm, np.arange(Y.shape[0])], axis=0)
    return Xs, label_X, label_Y, alignT


def initialze(alignT, method='random'):
    if method == 'partial':
        first_part = np.random.permutation(alignT.shape[0]//2)
        second_part = np.arange(alignT.shape[0]//2, alignT.shape[0])
        part = np.concatenate([first_part, second_part], axis=0)

        permutation_wrong_first = alignT[0, part]
        align0 = np.stack(
            [permutation_wrong_first, np.range(alignT.shape[0])], axis=0)

    elif method == "true":
        align0 = alignT.copy()
    else:
        align0 = np.stack([np.arange(alignT.shape[0]),
                          np.arange(alignT.shape[0])], axis=0)

    return align0
