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
    elif method == 'partial-preserved':
        N = Y.shape[0]
        group_len = 10
        num_permute = 4
        permute_array = np.stack([np.random.choice(
            group_len, num_permute, replace=False) for _ in range(N//group_len)])
        permute_array_true = np.mgrid[0:N//group_len,
                                      0:num_permute][0] * group_len + permute_array
        permute_sq = np.stack([np.random.permutation(num_permute)
                              for _ in range(N//group_len)])
        row = np.mgrid[0:N//group_len, 0:num_permute][0]
        location_array = np.stack([row, permute_sq], axis=2)
        permute_array_after = permute_array_true[location_array[:,
                                                                :, 0], location_array[:, :, 1]]
        # Y[permute_array_true] = Y[permute_array_after]
        perm = np.arange(N)
        perm[permute_array_true] = perm[permute_array_after]

    Y = Y[perm]
    label_Y = label_Y[perm]
    Xs = [X, Y]
    alignT = np.stack([perm, np.arange(Y.shape[0])], axis=0)
    return Xs, label_X, label_Y, alignT


def initialze(alignT, method='random'):
    if method == 'partial':
        first_part = np.random.permutation(alignT.shape[1]//2)
        second_part = np.arange(alignT.shape[1]//2, alignT.shape[1])
        part = np.concatenate([first_part, second_part], axis=0)

        permutation_wrong_first = alignT[0, part]
        align0 = np.stack(
            [permutation_wrong_first, np.arange(alignT.shape[1])], axis=0)

    elif method == "true":
        align0 = alignT.copy()

    else:
        align0 = np.stack([np.arange(alignT.shape[1]),
                          np.arange(alignT.shape[1])], axis=0)

    return align0


def load_data_adu(num_data_point=1000, normalize=False, permutation=False):
    data1 = load_data("data/noisymnist_view1.gz")
    data2 = load_data("data/noisymnist_view2.gz")

    # pca = PCA(n_components=123)

    train1, train2 = data1[0][0], data2[0][0]
    val1, val2 = data1[1][0], data2[1][0]
    test1, test2 = data1[2][0], data2[2][0]

    label_train1, label_train2 = data1[0][1], data2[0][1]
    label_val1, label_val2 = data1[1][1], data2[1][1]
    label_test1, label_test2 = data1[2][1], data2[2][1]

    if permutation:
        perm_train_init = np.random.permutation(train1.shape[0])
        perm_val_init = np.random.permutation(val1.shape[0])
        train1, label_train1 = train1[perm_train_init], label_train1[perm_train_init]
        train2, label_train2 = train2[perm_train_init], label_train2[perm_train_init]
        val1, label_val1 = val1[perm_val_init], label_val1[perm_val_init]
        val2, label_val2 = val2[perm_val_init], label_val2[perm_val_init]

    # train1 = pca.fit_transform(train1)
    # val1 = pca.transform(val1)
    # train2 = pca.fit_transform(train2)
    # val2 = pca.transform(val2)
    if normalize:
        # scaler = preprocessing.StandardScaler()
        # train1 = scaler.fit_transform(train1)
        # val1 = scaler.transform(val1)
        # train2 = scaler.fit_transform(train2)
        # val2 = scaler.transform(val2)
        pass

    train_choice = np.random.choice(50000, num_data_point, replace=False)
    test_choice = np.random.choice(10000, num_data_point//5, replace=False)

    # train1, train2, val1, val2 = train1[:num_data_point], train2[:
    #  num_data_point], val1[:num_data_point], val2[:num_data_point]
    # label_train1, label_train2, label_val1, label_val2 = label_train1[:num_data_point], label_train2[
    # :num_data_point], label_val1[:num_data_point], label_val2[:num_data_point]

    train1, train2, val1, val2 = train1[train_choice], train2[train_choice], val1[test_choice], val2[test_choice]
    label_train1, label_train2, label_val1, label_val2 = label_train1[
        train_choice], label_train2[train_choice], label_val1[test_choice], label_val2[test_choice]

    print(
        f"Finished loading data, Shape: Train1 {train1.shape}, Train2 {train2.shape}, Val1 {val1.shape}, Val2 {val2.shape}")

    return train1, train2, val1, val2, label_train1, label_train2, label_val1, label_val2
