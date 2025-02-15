
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from scipy.io import loadmat
import pandas as pd
import os
from nutrient_utils.my_utils import cal_performance, plot_scatter

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from autogluon.tabular import TabularDataset, TabularPredictor
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def load_train_data():

    # train data
    file_dir = "F:\subset_S2_S3_transfer_model"
    fns = os.listdir(file_dir)
    input = []
    output = []
    for fn in fns:
        if '2019_10_21' in fn:  # individual test date
            continue
        if 'input' in fn:
            input.append(np.load(os.path.join(file_dir, fn)))
        elif 'output' in fn:
            output.append(np.load(os.path.join(file_dir, fn)))
    input = np.concatenate(input, axis=0)
    output = np.concatenate(output, axis=0)
    np.save('input_train.npy', input)
    np.save('output_train.npy', output)

    # input = np.load('input_train.npy')
    # output = np.load('output_train.npy')

    s3_bands = [443, 490, 560, 620, 709, 754, 779, 865]  # s3ab bands
    s2_bands = [443, 492, 560, 665, 704, 740, 783, 865]  # s2a bands [使用s2a的波段名命名autogluon的label]

    features = pd.DataFrame(input,
                            columns=["s2rhorc_" + str(band) for band in s2_bands])  # sentinel-2a 8 bands

    labels = pd.DataFrame(output, columns=["s3rhorc_" + str(band) for band in s3_bands])  # s3 8 bands

    # 划分训练集和验证集
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                                random_state=42)
    # return features_train, labels_train, features_test, labels_test
    return features_train, features_test, labels_train, labels_test


def load_individual_test_data():
    # 加载Matlab文件中的数据
    file_dir = '/Volumes/WindyT7/subset'
    fns = os.listdir(file_dir)
    input = np.load(os.path.join(file_dir, 'input_2019_10_21.npy'))
    output = np.load(os.path.join(file_dir, 'output_2019_10_21.npy'))
    np.save('input_individual_test.npy', input)
    np.save('output_individual_test.npy', output)


    s3_bands = [443, 490, 560, 620, 709, 754, 779, 865]  # s3ab bands
    s2_bands = [443, 492, 560, 665, 704, 740, 783, 865]  # s2a bands [使用s2a的波段名命名autogluon的label]

    features = pd.DataFrame(input,
                            columns=["s2rhorc_" + str(band) for band in s2_bands])  # sentinel-2a 8 bands
    labels = pd.DataFrame(output, columns=["s3rhorc_" + str(band) for band in s3_bands])  # s3 8 bands
    return features, labels


def draw_scatter(ax, y_test, y_pred, y_train, y_pred_train, label, flag_x_label=True, flag_y_label=True,
                 flag_legend=True):
    font_dic = {"size": 16,
                "family": "arial",
                "weight": "normal"}
    bbox_dic = {"facecolor": "white",
                "ec": "grey",
                "lw": 0,
                "alpha": 0.7}
    R2, rmse, mape, n = cal_performance(y_test, y_pred)

    metrics = [f"$R^{2}$ = {R2:.2f}",
               f"RMSE = {rmse:.2f}",
               f"MAPE = {mape:.1f} %",
               f"n = {n}"]

    if y_train is not None:
        ax.scatter(y_train, y_pred_train, label='train',
                   facecolors='#B8E0E3', edgecolors='#2D9CDB', linewidths=1,
                   s=30, marker='s')

    # ax.scatter(y_test, y_pred, label='test',
    #            facecolors='#F9C095', edgecolors='#E26C25', linewidths=1,
    #            s=30, marker='o')
    ax.hist2d(y_test, y_pred, bins=40, cmap='Blues', cmin=1, norm=mcolors.LogNorm())

    ax.text(0.15, 0.9, "\n".join(metrics), transform=ax.transAxes,
            ha='left', va='top',
            fontdict={'size': 9, 'family': 'arial'}, color='#E26C25', bbox=bbox_dic)
    max = 0.2
    ax.plot([0, max], [0, max], zorder=0, color='grey', linewidth=1, linestyle='--')
    ax.set_ylim(0, max)
    ax.set_xlim(0, max)
    ax.set_xticks(np.arange(0, max, 0.1).round(1))
    ax.set_yticks(np.arange(0, max, 0.1).round(1))
    ax.tick_params(axis='both', labelsize=14)
    ax.xaxis.set_tick_params(width=1.5, length=5, direction='out')
    ax.yaxis.set_tick_params(width=1.5, length=5, direction='out')
    for axis in ['left', 'right', 'bottom', 'top']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('k')

    if flag_x_label:
        ax.set_xlabel('In-situ', fontdict=font_dic)
    if flag_y_label:
        ax.set_ylabel('Predicted', fontdict=font_dic)
    ax.set_title(label, fontdict=font_dic)

    if flag_legend:
        ax.legend(prop={'size': 12, 'family': 'arial'}, frameon=True, loc='upper right')

def autogl_train():
    features_train, features_test, labels_train, labels_test = load_train_data()
    df_train = pd.concat([features_train, labels_train], axis=1)
    df_test = pd.concat([features_test, labels_test], axis=1)

    train_data = TabularDataset(df_train)
    test_data = TabularDataset(df_test)

    s3_bands = [443, 490, 560, 620, 709, 754, 779, 865]  # s3ab bands
    s2_bands = [443, 492, 560, 665, 704, 740, 783, 865]  # s2a bands

    s3_bands_labels = ['s3rhorc_' + str(band) for band in s3_bands]

    # train
    # for idx, s3_bands_label in enumerate(s3_bands_labels):
    #     # if s3_bands_label in ['s3rhorc_443']:
    #     #     continue
    #
    #     predictor = TabularPredictor(label=s3_bands_label, path=f'./AutogluonModels/ag-{s3_bands_label}').fit(
    #         train_data.drop(columns=[col for col in s3_bands_labels if col != s3_bands_label]),  # label仅保留要预测的波段
    #         hyperparameters='multimodal',
    #         num_stack_levels=1, num_bag_folds=5
    #     )

    # predict
    y_pred_allband = []
    for idx, s3_bands_label in enumerate(s3_bands_labels):
        predictor = TabularPredictor.load(os.path.join('AutogluonModels', f"ag-{s3_bands_label}"), require_version_match=False)
        # y_pred_train = predictor.predict(df_train.drop(columns=[col for col in labels if col != label]))
        y_pred = predictor.predict(df_test.drop(columns=[col for col in s3_bands_labels if col != s3_bands_label]))
        y_pred_allband.append(y_pred)
    y_pred_allband = pd.concat(y_pred_allband, axis=1)

    labels_test.to_csv('test-y_true.csv', index=False)
    y_pred_allband.to_csv('test-y_pred.csv', index=False)

    # draw
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), constrained_layout=True)
    axes = axes.flatten()
    colorbar_norm = Normalize(vmin=0, vmax=3000)
    for i, (s2_band, s3_band) in enumerate(zip(s2_bands, s3_bands)):
        x = labels_test.iloc[:, i]
        y = y_pred_allband.iloc[:, i]
        ax = axes[i]
        plot_scatter(ax, x, y, f'$\\rho_{{rc}}$({s3_band})', f'$\\rho_{{rc}}$({s2_band})', point_size=100,
                     is_density=True, colorbar_norm=colorbar_norm)

    # 添加共享颜色条
    cbar_ax = fig.add_subplot()  # 创建一个新轴用于颜色条
    sm = ScalarMappable(norm=colorbar_norm, cmap="RdBu_r")
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Density')

    # 调整颜色条的位置和大小
    cbar_ax.set_position([0.7, -0.05, 0.25, 0.03])  # [left, bottom, width, height]，设置到右下角

    plt.savefig(f'./assessment_test_autogluon_all.png', dpi=300, bbox_inches='tight')
    plt.show()


def autogl_test():

    features, labels = load_individual_test_data()

    df_test = pd.concat([features, labels], axis=1)

    test_data = TabularDataset(df_test)

    s3_bands = [443, 490, 560, 620, 709, 754, 779, 865]  # s3ab bands
    s2_bands = [443, 492, 560, 665, 704, 740, 783, 865]  # s2a bands

    s3_bands_labels = ['s3rhorc_' + str(band) for band in s3_bands]

    # predict
    y_pred_allband = []
    for idx, s3_bands_label in enumerate(s3_bands_labels):
        predictor = TabularPredictor.load(os.path.join('AutogluonModels', f"ag-{s3_bands_label}"))
        # y_pred_train = predictor.predict(df_train.drop(columns=[col for col in labels if col != label]))
        y_pred = predictor.predict(df_test.drop(columns=[col for col in s3_bands_labels if col != s3_bands_label]))
        y_pred_allband.append(y_pred)
    y_pred_allband = pd.concat(y_pred_allband, axis=1)

    # draw
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), constrained_layout=True)
    axes = axes.flatten()
    colorbar_norm = Normalize(vmin=0, vmax=3000)
    for i, (s2_band, s3_band) in enumerate(zip(s2_bands, s3_bands)):
        x = labels.iloc[:, i]
        y = y_pred_allband.iloc[:, i]
        ax = axes[i]
        plot_scatter(ax, x, y, f'$\\rho_{{rc}}$({s3_band})', f'$\\rho_{{rc}}$({s2_band})', point_size=100,
                     is_density=True, colorbar_norm=colorbar_norm)

    # 添加共享颜色条
    cbar_ax = fig.add_subplot()  # 创建一个新轴用于颜色条
    sm = ScalarMappable(norm=colorbar_norm, cmap="RdBu_r")
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Density')

    # 调整颜色条的位置和大小
    cbar_ax.set_position([0.7, -0.05, 0.25, 0.03])  # [left, bottom, width, height]，设置到右下角

    plt.savefig(f'./assessment_individual_test_autogluon_density.png', dpi=300, bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    autogl_train()

    # autogl_test()
    # pass

