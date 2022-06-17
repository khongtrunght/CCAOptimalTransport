# from cgi import test
# from tkinter import N
from matplotlib import pyplot as plt
import ot
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from linear_cca import linear_cca
from utils import initialze, permutation_data, load_data_adu
import wandb
import seaborn as sns
from opw import opw
import argparse


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
        self.cca_sklearn = CCA(n_components=outdim_size)
        self.outdim_size = outdim_size

    def run_1iter(self, Xs, align0, iter_num=0, algo='preserved', **kwargs):
        Ys = self.seqInp(Xs, align0)

        # --------- Cua tac gia ---------------
        self.linear_cca.fit(Ys[0], Ys[1], outdim_size=self.outdim_size)
        Xs_project = self.linear_cca.transform(Xs[0], Xs[1])

        # ----------- Cua sklearn -----------------
        # print("Test , Ys shape", Ys[0].shape, Ys[1].shape)
        # self.cca_sklearn.fit(Ys[0], Ys[1])
        # Xs0_c = self.cca_sklearn.transform(Xs[0])
        # print("Xs0_c shape", Xs0_c.shape)
        # exit()

        self.change_iter_ot((iter_num + 1) * 10)

        tp_plan = self.apply_ot(
            Xs_project, metric=self.metric, reg=self.reg, numItermax=self.num_iter_ot, algo=algo, **kwargs)

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
        metric = kwargs.pop('metric', 'euclidean')
        reg = kwargs.pop('reg', 1e-1)
        numItermax = kwargs.pop('numItermax', 1000)
        algo = kwargs.pop('algo', 'preserved')
        lambda1 = kwargs.pop('lambda1', 100)
        lambda2 = kwargs.pop('lambda2', 2)
        delta = kwargs.pop('delta', 1)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if algo == 'preserved':
            distance, tp_plan = opw(
                Xs[0], Xs[1], lambda1=lambda1, lambda2=lambda2, delta=delta)

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


def main(args):

    init = 'true'

    train1, train2, val1, val2, label_train1, label_train2, label_val1, label_val2 = load_data_adu(
        num_data_point=args.num_data_point, normalize=args.normalize, permutation=args.permutation)

    Xs, label_train1, label_train2, alignT = permutation_data(
        train1, train2, label_train1, label_train2, method=args.permutation_method)

    align0 = initialze(alignT, method=args.init)

    ccaot = CcaOT(outdim_size=args.dim_data,
                  num_iter=args.num_iter_max, reg=1e-1, num_iter_ot=10000)

    if alignT is not None:
        tp_planT = np.zeros((args.num_data_point, args.num_data_point))
        for i in range(alignT.shape[1]):
            tp_planT[alignT[0, i], alignT[1, i]] = 1

    if align0 is None:
        align0 = np.stack([np.arange(args.num_data_point),
                           np.arange(args.num_data_point)], axis=0)

    f_norm_previous = np.inf
    for iter in range(args.num_iter_max):
        align, tp_plan = ccaot.run_1iter(
            Xs, align0, iter_num=iter, lambda1=args.lambda1, lambda2=args.lambda2, delta=args.delta)
        val1_project, val2_project = ccaot.transform([val1, val2])
        f_norm = np.linalg.norm(val1_project - val2_project)
        print(align[0][10] == align[1][10])
        # 'align': wandb.Table(columns=['view1', 'view2'], data=align.T)
        wandb.log({'f_norm': f_norm}, step=iter)
        wandb.log({'transport_loss': ccaot.check_diff(
            tp_plan, tp_planT)}, step=iter)

        # label loss (cross entropy)
        label_view1 = label_train1[align]
        label_view2 = label_train2[align]
        label_loss = cross_entropy(label_view2, label_view1)
        wandb.log({'label_loss': label_loss}, step=iter)

        if iter == 0:
            sns.heatmap(tp_planT[:40, :40])
            wandb.log({"tp_planT": wandb.Image(plt)}, step=iter)
            plt.clf()
        sns.heatmap(tp_plan[:40, :40])  # , cmap='viridis'
        wandb.log({'tp_plan': wandb.Image(plt)}, step=iter)
        plt.clf()
        if iter % args.draw_tsne_step == 0:
            # wandb.log(
            #     {"project_matrix_view1":  ccaot.linear_cca.w[0], "mean_view1": ccaot.linear_cca.m[0]}, step=iter)
            tsne_plot(val1_project, label_val1, 2)
            wandb.log({'tsne_val1': wandb.Image(plt)}, step=iter)
            plt.clf()
            if abs(f_norm_previous - f_norm) < 1e-2:
                break

        align0 = align
        f_norm_previous = f_norm

    wandb.finish()


def tsne_plot(data, label, dim_size=2):
    tsne = TSNE(n_components=dim_size)
    after_tsne = tsne.fit_transform(data)
    # plt.scatter(after_tsne[:, 0], after_tsne[:, 1], c=label)
    sns.scatterplot(x=after_tsne[:, 0], y=after_tsne[:, 1],
                    palette=sns.color_palette("hls", 10), hue=label)
    del tsne  # clear memory


# wandb.run.name = "Partial int - 5000 - ones ot"
parser = argparse.ArgumentParser()
main_run_args = parser.add_argument_group('Main run')
ot_args = parser.add_argument_group('OT')
exp_args = parser.add_argument_group('Experiment')

main_run_args.add_argument(
    '-nd', '--num_data_point', type=int, required=True, help='Number of data point')
main_run_args.add_argument('-nim', '--num-iter-max', type=int, default=100)
main_run_args.add_argument('-dim', '--dim-data', type=int, default=10)

ot_args.add_argument('-algo', '--algo', type=str, default='preserved')
ot_args.add_argument('-l1', '--lambda1', type=float,
                     default=50, help="the weight of the IDM regularization")
ot_args.add_argument('-l2', '--lambda2', type=float, default=0.1,
                     help="the weight of the KL-divergence regularization")
ot_args.add_argument('-d', '--delta', type=float, default=1,
                     help="the parameter of the prior Gaussian distribution")


exp_args.add_argument('--init', type=str, default='random')
exp_args.add_argument('--draw-tsne-step', type=int, default=10)
exp_args.add_argument('--normalize', type=bool,
                      default=True, help="Normalize input data")
exp_args.add_argument('--permutation', type=bool,
                      default=True, help="Permute input data")
exp_args.add_argument('--permutation-method', type=str,
                      default='partial-preserved')

args = parser.parse_args()
print(args)
wandb.init(project="ccaOT")
wandb.config.update(args)


if __name__ == "__main__":
    main(args)
    plt.close()
