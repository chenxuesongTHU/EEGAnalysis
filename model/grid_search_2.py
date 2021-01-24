#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   grid_search  
@Time        :   2020/4/16 10:58 上午
@Author      :   Xuesong Chen
@Description :   
"""

# todo: STEP 3
# idx = 0
# feature_type_idx = 2
import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing

from common.constants import *
from common.utils import mkdir
import random
from sklearn.utils import shuffle

def map_sat(x):
    if x > 2:
        return 1
    elif x < 2:
        return 0
    else:
        return -1

# file_name = models_type[feature_type_idx]

def getData(user_idx, feature_type_idx):
    if feature_type_idx in [3, 11]:
        return get_eye_movement_data(user_idx, feature_type_idx)
    if feature_type_idx == 5:
        return get_psd_de_eye_movement_area_data(user_idx, feature_type_idx)
    if feature_type_idx in [6, 7, 8]:
        return getData_with_blog_uid(user_idx, feature_type_idx)
    if feature_type_idx in [9]:
        return getData_with_blog_uid_header(user_idx, feature_type_idx)
    if feature_type_idx == 10:
        return get_two_type_features_data(user_idx, feat_type_1=8, feat_type_2=9)
    if feature_type_idx == 12:
        return get_two_type_features_data(user_idx, feat_type_1=11, feat_type_2=9)
    if feature_type_idx == 13:
        return get_three_type_features_data(user_idx, feat_type_1=8, feat_type_2=11, feat_type_3=9)
    if feature_type_idx == 14:
        return get_two_type_features_data(user_idx, feat_type_1=3, feat_type_2=9)
    if feature_type_idx == 15:
        return get_three_type_features_data(user_idx, feat_type_1=8, feat_type_2=3, feat_type_3=9)
    file_name = models_type[feature_type_idx]
    # df = pd.read_csv(f'{prj_path}/dataset/{user_name_list[idx]}_psd_in_diff_bands_stft_n_windows=3_step=1.csv', header=None)
    df = pd.read_csv(f'{prj_path}/dataset/{user_name_list[user_idx]}/{file_name}.csv', header=None)
    df = df[df[0] != 2]
    y = df[0].apply(lambda x: map_sat(x))
    X = df.drop(0, axis=1, inplace=False)
    random.seed(2021)
    X = X.values.tolist()
    y = list(y)
    cc = list(zip(X, y))
    random.shuffle(cc)
    X[:], y[:] = zip(*cc)
    # print(aa, bb)
    return X, y


def getData_with_blog_uid(user_idx, feature_type_idx):
    file_name = models_type[feature_type_idx]
    df = pd.read_csv(f'{prj_path}/dataset/{user_name_list[user_idx]}/{file_name}.csv', index_col=0, header=None)
    df = df[df[1] != 2]
    y = df[1].apply(lambda x: map_sat(x))
    X = df.drop(1, axis=1, inplace=False)
    # X = preprocessing.normalize(X, axis=1)
    random.seed(2021)
    X = X.values.tolist()
    y = list(y)
    cc = list(zip(X, y))
    random.shuffle(cc)
    X[:], y[:] = zip(*cc)
    # print(aa, bb)
    return X, y


def getData_with_blog_uid_header(user_idx, feature_type_idx):
    file_name = models_type[feature_type_idx]
    df = pd.read_csv(f'{prj_path}/dataset/{user_name_list[user_idx]}/{file_name}.csv', index_col=0)
    df = df[df['satisfaction'] != 2]
    y = df['satisfaction'].apply(lambda x: map_sat(x))
    X = df.drop(['satisfaction'], axis=1)
    X = preprocessing.normalize(X, axis=0)
    random.seed(2021)
    X = list(X)
    y = list(y)
    cc = list(zip(X, y))
    random.shuffle(cc)
    X[:], y[:] = zip(*cc)
    # print(aa, bb)
    return X, y


def get_eye_movement_data(user_idx, feature_type_idx):
    file_name = models_type[feature_type_idx]
    df = pd.read_csv(f'{prj_path}/dataset/{user_name_list[user_idx]}/{file_name}.csv', index_col=0)
    df = df[df['satisfaction'] != 2]
    y = df['satisfaction'].apply(lambda x: map_sat(x))
    X = df.drop(['satisfaction'], axis=1)
    X = preprocessing.normalize(X, axis=1)
    random.seed(2021)
    X = list(X)
    y = list(y)
    cc = list(zip(X, y))
    random.shuffle(cc)
    X[:], y[:] = zip(*cc)
    # print(aa, bb)
    return X, y


def get_two_type_features_data(user_idx, feat_type_1, feat_type_2):

    eeg_file_name = models_type[feat_type_1]
    dwell_time_file_name = models_type[feat_type_2]
    if feat_type_1 in [3, 11]:
        eeg_df = pd.read_csv(
            f'{prj_path}/dataset/{user_name_list[user_idx]}/{eeg_file_name}.csv', index_col=0
        )
        eeg_df.drop(['satisfaction'], axis=1, inplace=True)
    else:
        eeg_df = pd.read_csv(
            f'{prj_path}/dataset/{user_name_list[user_idx]}/{eeg_file_name}.csv', index_col=0, header=None
        )
    dwell_time_df = pd.read_csv(
        f'{prj_path}/dataset/{user_name_list[user_idx]}/{dwell_time_file_name}.csv', index_col=0
    )

    df = pd.concat([eeg_df, dwell_time_df], axis=1)
    df.dropna(axis=0, how='any', inplace=True)
    df = df[df['satisfaction'] != 2]
    y = df['satisfaction'].apply(lambda x: map_sat(x))
    if feat_type_1 in [3, 11]:
        X = df.drop(['satisfaction'], axis=1)
    else:
        X = df.drop([1, 'satisfaction'], axis=1)
    norm_dwell_time = preprocessing.normalize(X['dwellTime'].to_frame(), axis=0)
    if feat_type_1 in [3, 11]:
        norm_eeg_X = preprocessing.normalize(X[columns], axis=1)
    else:
        norm_eeg_X = X.iloc[:, 0:-1].values
    X = np.concatenate([norm_eeg_X, norm_dwell_time], axis=1)
    random.seed(2021)
    X = list(X)
    y = list(y)
    cc = list(zip(X, y))
    random.shuffle(cc)
    X[:], y[:] = zip(*cc)
    # print(aa, bb)
    return X, y

def get_three_type_features_data(user_idx, feat_type_1, feat_type_2, feat_type_3):

    eeg_file_name = models_type[feat_type_1]
    eye_movement_file_name = models_type[feat_type_2]
    dwell_time_file_name = models_type[feat_type_3]

    eeg_df = pd.read_csv(
        f'{prj_path}/dataset/{user_name_list[user_idx]}/{eeg_file_name}.csv', index_col=0, header=None
    )

    eye_movement_df = pd.read_csv(
        f'{prj_path}/dataset/{user_name_list[user_idx]}/{eye_movement_file_name}.csv', index_col=0,
    )
    eye_movement_df.drop(['satisfaction'], axis=1, inplace=True)

    dwell_time_df = pd.read_csv(
        f'{prj_path}/dataset/{user_name_list[user_idx]}/{dwell_time_file_name}.csv', index_col=0
    )

    df = pd.concat([eeg_df, eye_movement_df, dwell_time_df], axis=1)
    df.dropna(axis=0, how='any', inplace=True)
    df = df[df['satisfaction'] != 2]
    y = df['satisfaction'].apply(lambda x: map_sat(x))
    X = df.drop([1, 'satisfaction'], axis=1)
    norm_dwell_time = preprocessing.normalize(X['dwellTime'].to_frame(), axis=0)
    norm_eeg_X = X.iloc[:, 0:50].values
    norm_eye_movement_X = preprocessing.normalize(X[columns], axis=1)
    X = np.concatenate([norm_eeg_X, norm_eye_movement_X, norm_dwell_time], axis=1)
    random.seed(2021)
    X = list(X)
    y = list(y)
    cc = list(zip(X, y))
    random.shuffle(cc)
    X[:], y[:] = zip(*cc)
    # print(aa, bb)
    return X, y

def get_psd_de_eye_movement_area_data(user_idx, feature_type_idx):

    eeg_file_name = models_type[4]
    eye_movement_file_name = models_type[3]

    eeg_df = pd.read_csv(
        f'{prj_path}/dataset/{user_name_list[user_idx]}/{eeg_file_name}.csv', index_col=0, header=None
    )
    eye_movement_df = pd.read_csv(
        f'{prj_path}/dataset/{user_name_list[user_idx]}/{eye_movement_file_name}.csv', index_col=0
    )
    df = pd.concat([eeg_df, eye_movement_df], axis=1)
    df.dropna(axis=0, how='any', inplace=True)
    df = df[df['satisfaction'] != 2]
    y = df['satisfaction'].apply(lambda x: map_sat(x))
    X = df.drop([1, 'satisfaction'], axis=1)
    norm_eye_movement_X = preprocessing.normalize(X[columns], axis=1)
    norm_eeg_X = X.iloc[:, 0:-4].values
    X = np.concatenate([norm_eeg_X, norm_eye_movement_X], axis=1)
    random.seed(2021)
    X = list(X)
    y = list(y)
    cc = list(zip(X, y))
    random.shuffle(cc)
    X[:], y[:] = zip(*cc)
    # print(aa, bb)
    return X, y

# get_psd_de_eye_movement_area_data(0, 5)

#K近邻（K Nearest Neighbor）
def KNN():
    clf = neighbors.KNeighborsClassifier()
    return clf

#线性鉴别分析（Linear Discriminant Analysis）
def LDA():
    clf = LinearDiscriminantAnalysis()
    return clf

#支持向量机（Support Vector Machine）
def SVM():
    clf = svm.SVC()
    return clf

#逻辑回归（Logistic Regression）
def LR():
    clf = LogisticRegression()
    return clf

#随机森林决策树（Random Forest）
def RF():
    clf = RandomForestClassifier()
    return clf

#多项式朴素贝叶斯分类器
def native_bayes_classifier():
    clf = MultinomialNB(alpha = 0.01)
    return clf

#决策树
def decision_tree_classifier():
    clf = tree.DecisionTreeClassifier()
    return clf

#GBDT
def gradient_boosting_classifier():
    clf = GradientBoostingClassifier()
    return clf

#计算识别率
def getRecognitionRate(testPre, testClass):
    testNum = len(testPre)
    rightNum = 0
    for i in range(0, testNum):
        if testClass[i] == testPre[i]:
            rightNum += 1
    return float(rightNum) / float(testNum)


#report函数，将调参的详细结果存储到本地F盘（路径可自行修改，其中n_top是指定输出前多少个最优参数组合以及该组合的模型得分）
def report(results, model_name, user_idx, file_name, n_top=100):
    # f = open('results/robust_scalar_grid_search_RF.txt', 'w')
    path = f'result/auc/{user_name_list[user_idx]}/{file_name}'
    mkdir(path)
    cross_detail_f = open(f'{path}/result_{model_name}.txt', 'w')
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            cross_detail_f.write(
                "%f\t%f\t"%(
                    results['mean_test_score'][candidate],  results['std_test_score'][candidate]
                )
            )
            for i in range(5):
                cross_detail_f.write(
                    "%f"%results['split%d_test_score'%i][candidate]
                )
                cross_detail_f.write(
                    '\t'
                )
            cross_detail_f.write('\n')
    #         f.write("Model with rank: {0}".format(i) + '\n')
    #         f.write("Mean validation score: {0:.3f} (std: {1:.3f})".format(
    #               results['mean_test_score'][candidate],
    #               results['std_test_score'][candidate]) + '\n')
    #         f.write("Parameters: {0}".format(results['params'][candidate]) + '\n')
    #         f.write("\n")
    # f.close()
    cross_detail_f.close()

#自动调参（以随机森林为例）
def selectRFParam(user_idx, feature_type_idx, file_name):
    # clf_RF = KNN()
    clf_RF = RandomForestClassifier()
    clf_LR = LogisticRegression()
    clf_SVM = SVM()
    clf_KNN = KNN()
    clf_GBDT = GradientBoostingClassifier()
    # GBDT
    param_grid_GBDT = {
                  "max_depth": [1, 2, 3, 5],
                  # "min_samples_split": range(50, 130, 25),
                  # "min_samples_leaf": range(5,80,15),
                  "criterion": ["friedman_mse"],
                  "n_estimators": range(10, 90, 15),
                 }

    # RF
    param_grid_RF = {
                  "max_depth": [1,3,5,7, 10],
                  "min_samples_split": [3,5,10, 20, 30],
                  "min_samples_leaf": [3,5,10, 20],
                  "max_features": ['auto', 'sqrt', 'log2', None],
                  "bootstrap": [True],
                  "criterion": ["gini", "entropy"],
                  "n_estimators": [200, 300],
                 }

    best_param_of_RF = {
                  "max_depth": 3,
                  "min_samples_split": 3,
                  "min_samples_leaf": 5,
                  "max_features": 'auto',
                  "bootstrap": True,
                  # "criterion": ["gini", "entropy"],
                  "n_estimators": 300,
                 }

    # SVM
    param_grid_SVM = {
                  "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                  "gamma": ['scale', 'auto', 'auto_deprecated'],
                  "decision_function_shape": ['ovo', 'ovr'],
                 }

    # # LR
    param_grid_LR = {
                  "penalty": ['l1', 'l2'],
                  "multi_class": ['ovr', 'auto'],
                 }

    # KNN
    param_grid_KNN = {
        # "n_neighbors": range(10, 800, 10),
        "weights": ['distance'],
        "algorithm": ['auto'],
        # "leaf_size": range(10, 150, 50),
        "p": range(1, 5, 1)
    }
    T = getData(user_idx, feature_type_idx)
    for model_name, clf, param_grid in zip(
            ['LR', 'SVM', 'KNN', 'RF', 'GBDT'],
            [clf_LR, clf_SVM, clf_KNN, clf_RF, clf_GBDT],
            [param_grid_LR, param_grid_SVM, param_grid_KNN, param_grid_RF, param_grid_GBDT]
        ):
        grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
        start = time()
        grid_search.fit(T[0], T[1]) #传入训练集矩阵和训练样本类标
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(grid_search.cv_results_['params'])))
        report(grid_search.cv_results_, model_name, user_idx, file_name)


def print_importance_RF():
    # max_fq = 30
    best_param_of_RF = {
                  "max_depth": 3,
                  "min_samples_split": 3,
                  "min_samples_leaf": 5,
                  "max_features": 'auto',
                  "bootstrap": True,
                  # "criterion": ["gini", "entropy"],
                  "n_estimators": 300,
                 }

    best_param_of_RF = {
                  "max_depth": 3,
                  "min_samples_split": 20,
                  "min_samples_leaf": 10,
                  "max_features": 'log2',
                  "bootstrap": True,
                  # "criterion": ["gini", "entropy"],
                  "n_estimators": 200,
                 }

    clf_RF = RandomForestClassifier(**best_param_of_RF)
    T = getData()
    clf_RF.fit(T[0], np.ravel(T[1]))
    print(clf_RF.feature_importances_)


if __name__ == '__main__':

    # for user_idx in range(2, 6):
    #     for feature_type_idx in range(1, 3):
    # selectRFParam(user_idx, feature_type_idx)
    for user_idx in range(0, 16):
        for feature_type_idx in [12, 14, 15]:
            print(models_type[feature_type_idx])
            file_name = models_type[feature_type_idx]
            selectRFParam(user_idx, feature_type_idx, file_name)
    print('随机森林参数调优完成！')

    # 输出不同特征的importance
    # print_importance_RF()