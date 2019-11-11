import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_curve, auc, precision_recall_curve  # average_precision_score


def plot_history_results(history):
    l_metrics = list(history.keys())
    for metric in sorted(l_metrics):
        # print(metric)
        if "val" in metric:
            print("removing", metric)
            l_metrics.remove(metric)

    nplots = len(l_metrics)
    ncols = 4
    nrows = nplots // 3 + np.sum((nplots % ncols) != 0)

    plt.figure(figsize=(4 * ncols, 4 * nrows))

    for iplot, metric in enumerate(l_metrics, 1):
        plt.subplot(nrows, ncols, iplot)
        plt.plot(history[metric], "--", linewidth=3, label="training", alpha=0.7, color='red')
        try:
            plt.plot(history["val_{}".format(metric)], "-o", linewidth=3, label="validation", alpha=0.1, color='green')
            plt.plot(savgol_filter(history["val_{}".format(metric)], 5, 1), linewidth=3, color='green')
        except:
            None
        plt.title(metric, fontsize=16)
        plt.legend()


def pos_neg_histograms(neg_preds, pos_preds, dbin=None, histtype="step", linewidth=3, xlabel=None, title=None,
                       fontsize=16, legend=True, bins=None):

    if bins is None:
        if dbin:
            bins = np.arange(0., 1 + dbin, dbin)

        else:
            preds = pd.Series(np.append(neg_preds, pos_preds))
            dbin = preds.std() / 3.
            bins = np.arange(preds.min(), preds.max() + dbin, dbin)


    label = f'negatives ({len(neg_preds):,})'
    plt.hist(neg_preds, bins=bins, density=True,
             color="red", alpha=0.7, histtype=histtype, linewidth=linewidth, label=label)

    label = f'positives ({len(pos_preds):,})'
    plt.hist(pos_preds, bins=bins, density=True, color="green", alpha=0.7, histtype=histtype, linewidth=linewidth,
             label=label)

    if not xlabel:
        xlabel = 'p(pos)'
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)

    if legend:
        plt.legend()


def plot_calibration_curve(true, predictions, n_bins=5, fontsize=16):
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(true, predictions, n_bins=n_bins)

    clf_score = brier_score_loss(true, predictions, pos_label=true.max())
    plt.plot(mean_predicted_value, fraction_of_positives, 's-',
             label=f'(Brier score: {np.round(clf_score, 3)}')
    plt.plot([0., 1.], [0., 1.], '--')
    plt.legend()
    plt.xlabel('Predicted probabilities (of being positive)', fontsize=fontsize)
    plt.ylabel('Actual probabilities\n(of being positive)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize);
    plt.yticks(fontsize=fontsize)


def plot_precision_recall_roc(true, predictions, figsize=(16, 5), fontsize=16, lw=2):
    df = pd.DataFrame({'actual': true, 'predictions': predictions})

    benchmark = df['actual'].value_counts(normalize=True)[1]

    fpr, tpr, thresholds_roc = roc_curve(df['actual'], df['predictions'])
    roc_auc = auc(fpr, tpr)

    # Precision Recall metrics
    precision, recall, thresholds_pr = precision_recall_curve(df['actual'], df['predictions'])

    # idx = np.arange(0, len(tpr))[tpr < sr_best['recall']][-1]

    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    label = 'AUC %0.2f' % roc_auc
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=label)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.scatter(fpr[idx], tpr[idx], label=None, marker='x', s=150, linewidth=5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1-Specificity)', fontsize=fontsize)
    plt.ylabel('True Positive Rate (Recall)', fontsize=fontsize)
    plt.title('Receiver Operating Characteristic', fontsize=fontsize)
    plt.legend(loc='lower right', fontsize=fontsize)

    plt.subplot(1, 2, 2)
    # label = 'Average Precision = %0.2f' % average_precision
    plt.plot(recall, precision, color='darkorange', lw=lw)  # , label=label)
    yval = benchmark
    plt.plot([0, 1], [yval, yval], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall/Sensitivity/(True Positive Rate)', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.title('Precision Recall Curve', fontsize=fontsize)

    # --- f1 score contours ---
    f_scores = np.linspace(0.2, 0.8, num=4)
    f_scores = np.append(f_scores, 0.9)

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
