import csv
import os

import sklearn
import time

import numpy as np
import xlwt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve  # roc计算曲线
from sklearn.model_selection import StratifiedShuffleSplit  # 创建随机数并打乱
import tensorflow as tf
from models import LstmGanModel
from data import read_data, DataSet


class ExperimentSetup(object):
    kfold = 5
    batch_size = 64
    encoder_size = 50
    lstm_size = 128
    learning_rate = 0.001
    epochs = 10
    output_n_epochs = 1
    max_times_of_visits = 3
    ridge_l2 = 0.001


def evaluate(test_index, y_label, y_score, file_name):
    """
    对模型的预测性能进行评估
    :param test_index
    :param y_label: 测试样本的真实标签 true label of test-set
    :param y_score: 测试样本的预测概率 predicted probability of test-set
    :param file_name: 输出文件路径    path of output file
    """
    # TODO 全部算完再写入
    wb = xlwt.Workbook(file_name + '.xls')
    table = wb.add_sheet('Sheet1')
    table_title = ["test_index", "label", "prob", "pre", " ", "fpr", "tpr", "thresholds", " ",
                   "acc", "auc", "recall", "precision", "f1-score", "threshold"]
    for i in range(len(table_title)):
        table.write(0, i, table_title[i])

    auc = roc_auc_score(y_label, y_score)

    fpr, tpr, thresholds = roc_curve(y_label, y_score, pos_label=1)
    threshold = thresholds[np.argmax(tpr - fpr)]

    for i in range(len(fpr)):
        table.write(i + 1, table_title.index("fpr"), fpr[i])
        table.write(i + 1, table_title.index("tpr"), tpr[i])
        table.write(i + 1, table_title.index("thresholds"), float(thresholds[i]))
    table.write(1, table_title.index("threshold"), float(threshold))

    y_pred_label = (y_score >= threshold) * 1
    acc = accuracy_score(y_label, y_pred_label)
    recall = recall_score(y_label, y_pred_label)
    precision = precision_score(y_label, y_pred_label)
    f1 = f1_score(y_label, y_pred_label)

    for i in range(len(test_index)):
        table.write(i + 1, table_title.index("test_index"), int(test_index[i]))
        table.write(i + 1, table_title.index("label"), int(y_label[i]))
        table.write(i + 1, table_title.index("prob"), float(y_score[i]))
        table.write(i + 1, table_title.index("pre"), int(y_pred_label[i]))

    # write metrics
    table.write(1, table_title.index("auc"), float(auc))
    table.write(1, table_title.index("acc"), float(acc))
    table.write(1, table_title.index("recall"), float(recall))
    table.write(1, table_title.index("precision"), float(precision))
    table.write(1, table_title.index("f1-score"), float(f1))

    wb.save(file_name + ".xls")


def model_experiments(model, data_set, result_file):
    dynamic_features = data_set.dynamic_features
    labels = data_set.labels
    last_features = data_set.last_features
    kf = sklearn.model_selection.StratifiedKFold(n_splits=ExperimentSetup.kfold, shuffle=False)

    # n_output = labels.shape[1]  # classes
    n_output = 1  # classes

    tol_test_idx = np.zeros(0, dtype=np.int32)
    tol_pred = np.zeros(shape=(0, n_output))
    tol_label = np.zeros(shape=(0, n_output), dtype=np.int32)
    i = 1
    for train_idx, test_idx in kf.split(X=data_set.dynamic_features, y=data_set.labels[:, -1]):
        train_dynamic = dynamic_features[train_idx]
        train_y = labels[train_idx]
        train_last_features = last_features[train_idx]
        train_set = DataSet(train_dynamic, train_y, train_last_features)

        test_dynamic = dynamic_features[test_idx]
        test_y = labels[test_idx]
        test_last_features = last_features[test_idx]
        test_set = DataSet(test_dynamic, test_y, test_last_features)
        print("learning_rate = ", ExperimentSetup.learning_rate)
        model.fit(train_set, test_set)

        y_score = model.predict(test_set)
        tol_test_idx = np.concatenate((tol_test_idx, test_idx))
        tol_pred = np.vstack((tol_pred, y_score))
        tol_label = np.vstack((tol_label, np.expand_dims(test_y[:, -1], 1)))
        print("Cross validation: {} of {}".format(i, ExperimentSetup.kfold),
              time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        i += 1
        # evaluate(test_y, y_score, result_file)

    model.close()
    # with open(result_file, 'a', newline='') as csv_file:
    #     f_writer = csv.writer(csv_file, delimiter=',')
    #     f_writer.writerow([])
    # return evaluate(tol_label, tol_pred, result_file)
    return evaluate(tol_test_idx, tol_label, tol_pred, result_file)


def denoising_auto_encoder_lstm_decoder_model_experiments(result_file):
    data_set = read_data(ExperimentSetup.max_times_of_visits)
    dynamic_features = data_set.dynamic_features
    labels = data_set.labels

    num_features = dynamic_features.shape[2]
    time_steps = dynamic_features.shape[1]
    n_output = labels.shape[1]

    model = LstmGanModel(num_features=num_features,
                         time_steps=time_steps,
                         lstm_size=ExperimentSetup.lstm_size,
                         n_output=n_output,
                         batch_size=ExperimentSetup.batch_size,
                         epochs=ExperimentSetup.epochs,
                         output_n_epoch=ExperimentSetup.output_n_epochs,
                         learning_rate=ExperimentSetup.learning_rate,
                         ridge=ExperimentSetup.ridge_l2,
                         optimizer=tf.train.AdamOptimizer(ExperimentSetup.learning_rate))
    return model_experiments(model, data_set, result_file)


if __name__ == '__main__':
    # basic_lstm_model_experiments('resources/save/basic_lstm.csv')
    # lstm_with_static_feature_model_experiments("resources/save/lstm_with_static.csv")
    # bidirectional_lstm_model_experiments('resources/save/bidirectional_lstm.csv')
    for i_times in range(20):
        # print("mlp_bi-lstm_att")
        # bi_lstm_attention_model_experiments('result_qx/MLA1-' + str(i_times + 1), True, True)
        save_path = "heart_3mon_3times_10epoch_1gen_LR0.001_L0.001"
        # save_path = "all_cause_24mon"
        print("save to " + save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # print("DAE-LSTM")
        # denoising_auto_encoder_lstm_model_experiments(
        #     save_path + '/DAE-LSTM-DROPOUT-50DAEsize-20epoch-0.001lr-1-' + str(i_times + 1))
        #
        print("LSTM_GAN")
        denoising_auto_encoder_lstm_decoder_model_experiments(
            save_path + '/LSTM_GAN_1-' + str(i_times + 1))
