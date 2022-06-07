# from cgi import test
# from tkinter import N
from matplotlib import pyplot as plt
import ot
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from linear_cca import linear_cca
from src.utils import initialze, permutation_data
import wandb
from utils import load_data
import seaborn as sns
from opw import opw


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce


class CcaOT:
    def __init__(self, outdim_size, num_iter=100, tol=1e-4, num_iter_ot=500, reg=1e-3, metric='sqeuclidean'):
        self.num_iter = num_iter
        self.tol = tol
        self.num_iter_ot = num_iter_ot
        self.reg = reg
        self.metric = metric
        self.linear_cca = linear_cca()
        self.outdim_size = outdim_size

    def fit(self, Xs, align0=None, alignT=None):

        NUM_DATA_POINT = Xs[0].shape[0]
        if alignT is not None:
            tp_planT = np.zeros(NUM_DATA_POINT, NUM_DATA_POINT)
            for i in range(alignT.shape[1]):
                tp_planT[alignT[0, i], alignT[1, i]] = 1

        if align0 is None:
            align0 = np.stack([np.arange(NUM_DATA_POINT),
                               np.arange(NUM_DATA_POINT)], axis=0)

        for iter in range(self.num_iter):

            Ys = self.seqInp(Xs, align0)
            self.linear_cca.fit(Ys[0], Ys[1])

            Xs_project = self.linear_cca.transform(Xs[0], Xs[1])

            tp_plan = self.apply_ot(
                Xs_project, metric=self.metric, reg=self.reg, numItermax=self.num_iter_ot)

            align = self.get_align_path(tp_plan)

            Ys_project = self.seqInp(Xs_project, align)

            align0 = align.copy()

            f_norm = np.linalg.norm(Ys_project[0] - Ys_project[1])
            wandb.log({'f_norm': f_norm}, step=iter)

            if tp_planT:
                wandb.log({self.check_diff(tp_plan, tp_planT)}, step=iter)

    def run_1iter(self, Xs, align0, iter_num=0):
        Ys = self.seqInp(Xs, align0)
        self.linear_cca.fit(Ys[0], Ys[1], outdim_size=self.outdim_size)

        Xs_project = self.linear_cca.transform(Xs[0], Xs[1])

        self.change_iter_ot((iter_num + 1) * 10)

        tp_plan = self.apply_ot(
            Xs_project, metric=self.metric, reg=self.reg, numItermax=self.num_iter_ot)

        align = self.get_align_path(tp_plan)

        # Ys_project = self.seqInp(Xs_project, align)

        return (align, tp_plan)

    def transform(self, Xs):
        """
        Transform Xs to Ys using CCA
        """
        Ys_project = self.linear_cca.transform(Xs[0], Xs[1])
        return Ys_project

    def apply_ot(self, Xs, **kwargs):
        """apply OT to Xs
        Args:
            Xs (list): num_view * ( num_samples, num_features)
            **kwargs: keyword arguments for OT

        Returns:
            tp_plan (ndarray): shape (num_samples, num_samples)
        """
        metric = kwargs.get('metric', 'euclidean')
        reg = kwargs.get('reg', 1e-1)
        numItermax = kwargs.get('numItermax', 1000)
        algo = kwargs.get('algo', ' preserved')

        if algo == 'preserved':
            distance, tp_plan = opw(
                Xs[0], Xs[1], lambda1=50, lambda2=reg, delta=1)

        else:  # algo == 'regularized':
            cost_matrix = ot.dist(Xs[0], Xs[1], metric=metric)
            tp_plan = ot.sinkhorn(np.ones(Xs[0].shape[0]) / Xs[0].shape[0], np.ones(
                Xs[1].shape[0]) / Xs[1].shape[0], cost_matrix, reg=reg, numItermax=numItermax)
        return tp_plan

    def get_align_path(self, tp_plan):
        """get alignment path from transport plan

        Args:
            tp_plan (ndarray): shape (num_samples, num_samples)

        Returns:
            align: alignment path (num_views, alignment_length)
        """
        # get argmax by row (i.e. view original image)

        num_samples = tp_plan.shape[0]
        argmax_row = np.stack(
            [np.arange(num_samples), tp_plan.argmax(axis=1)], axis=0)
        # get argmax by column (i.e. view noisy image)
        argmax_col = np.stack(
            [tp_plan.argmax(axis=0), np.arange(num_samples)], axis=0)
        align = np.concatenate([argmax_row, argmax_col], axis=1)
        alignment_length = align.shape[1]
        return align

    def change_iter_ot(self, new_iter_num):
        self.num_iter_ot = new_iter_num

    def seqInp(self, X0s, alis):
        """
        Input sequence for CCA

        Args:
            Xs (list): (m, ni, di) feature vectors of each view in m
            ali (ndarray): (m, L) initial alignment

        Returns:
            Ys : (m, L, di) feature vectors of each view in m
        """
        m = 2  # number of views
        Xs = [np.zeros_like(X0s[i]) for i in range(len(X0s))]
        for i in range(m):
            X0 = X0s[i]
            ali = alis[i]

            Xs[i] = X0[ali, :]

        return Xs

    def check_diff(self, plan, true_plan):
        """
        Check difference between true plan and plan
        """

        return cross_entropy(plan.flat, true_plan.flat)


def load_data_adu(num_data_point=1000):
    data1 = load_data("data/noisymnist_view1.gz")
    data2 = load_data("data/noisymnist_view2.gz")

    pca = PCA(n_components=123)

    train1, train2 = data1[0][0], data2[0][0]
    val1, val2 = data1[1][0], data2[1][0]
    test1, test2 = data1[2][0], data2[2][0]

    train1 = pca.fit_transform(train1)
    val1 = pca.transform(val1)
    train2 = pca.fit_transform(train2)
    val2 = pca.transform(val2)

    label_train1, label_train2 = data1[0][1], data2[0][1]
    label_val1, label_val2 = data1[1][1], data2[1][1]
    label_test1, label_test2 = data1[2][1], data2[2][1]

    train_choice = np.random.choice(50000, num_data_point, replace=False)
    test_choice = np.random.choice(10000, num_data_point//10, replace=False)

    # train1, train2, val1, val2 = train1[:num_data_point], train2[:
    #  num_data_point], val1[:num_data_point], val2[:num_data_point]
    # label_train1, label_train2, label_val1, label_val2 = label_train1[:num_data_point], label_train2[
    # :num_data_point], label_val1[:num_data_point], label_val2[:num_data_point]

    train1, train2, val1, val2 = train1[train_choice], train2[train_choice], val1[test_choice], val2[test_choice]
    label_train1, label_train2, label_val1, label_val2 = label_train1[
        train_choice], label_train2[train_choice], label_val1[test_choice], label_val2[test_choice]

    return train1, train2, val1, val2, label_train1, label_train2, label_val1, label_val2


def main():
    wandb.init(project="ccaOT")
    # wandb.run.name = "Partial int - 5000 - ones ot"

    init = 'true'

    NUM_DATA_POINT = 5000
    NUM_ITER_MAX = 100
    DIM_AFTER_REDUCE = 10
    draw_step = 10

    train1, train2, val1, val2, label_train1, label_train2, label_val1, label_val2 = load_data_adu(
        num_data_point=NUM_DATA_POINT)


    Xs, label_train1, label_train2, alignT = permutation_data(train1, train2, label_train1, label_train2, method='preserved')

    align0 = initialze(alignT, method='random')




    ccaot = CcaOT(outdim_size=DIM_AFTER_REDUCE,
                  num_iter=NUM_ITER_MAX, reg=1e-1, num_iter_ot=10000)
    NUM_DATA_POINT = Xs[0].shape[0]
    if alignT is not None:
        tp_planT = np.zeros((NUM_DATA_POINT, NUM_DATA_POINT))
        for i in range(alignT.shape[1]):
            tp_planT[alignT[0, i], alignT[1, i]] = 1

    if align0 is None:
        align0 = np.stack([np.arange(NUM_DATA_POINT),
                           np.arange(NUM_DATA_POINT)], axis=0)

    for iter in range(NUM_ITER_MAX):
        align, tp_plan = ccaot.run_1iter(Xs, align0, iter_num=iter)
        val1_project, val2_project = ccaot.transform([val1, val2])
        f_norm = np.linalg.norm(val1_project - val2_project)
        print(align[0][10] == align[1][10])
        # 'align': wandb.Table(columns=['view1', 'view2'], data=align.T)
        wandb.log({'f_norm': f_norm}, step=iter)
        wandb.log({'transport_loss': ccaot.check_diff(
            tp_plan, tp_planT)}, step=iter)
        sns.heatmap(tp_plan)
        wandb.log({'tp_plan': wandb.Image(plt)}, step=iter)
        plt.clf()
        if iter % draw_step == 0:
            # wandb.log(
            #     {"project_matrix_view1":  ccaot.linear_cca.w[0], "mean_view1": ccaot.linear_cca.m[0]}, step=iter)
            tsne_plot(val1_project, label_val1, 2)
            wandb.log({'tsne_val1': wandb.Image(plt)}, step=iter)
            plt.clf()
        align0 = align

    wandb.finish()


def tsne_plot(data, label, dim_size=2):
    tsne = TSNE(n_components=dim_size)
    after_tsne = tsne.fit_transform(data)
    # plt.scatter(after_tsne[:, 0], after_tsne[:, 1], c=label)
    sns.scatterplot(after_tsne[:, 0], after_tsne[:, 1],
                    palette=sns.color_palette("hls", 10), hue=label)
    del tsne  # clear memory


if __name__ == "__main__":
    main()
    plt.close()
