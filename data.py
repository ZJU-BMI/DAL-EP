import pandas as pd
import numpy as np


class DataSet(object):
    def __init__(self, dynamic_features, labels, last_features):
        self._dynamic_features = dynamic_features
        self._labels = labels
        self._last_features = last_features
        self._num_examples = labels.shape[0]  # 这是什么意思
        self._epoch_completed = 0
        self._batch_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        if batch_size > self.num_examples or batch_size <= 0:
            # raise ValueError('The size of one batch: {} should be less than the total number of '
            #                  'data: {}'.format(batch_size, self.num_examples))
            batch_size = self._labels.shape[0]
        if self._batch_completed == 0:
            self._shuffle()
        self._batch_completed += 1
        start = self._index_in_epoch
        if start + batch_size >= self.num_examples:
            self._epoch_completed += 1
            feature_rest_part = self._dynamic_features[start:self._num_examples]
            label_rest_part = self._labels[start:self._num_examples]
            last_feature_rest_part = self._last_features[start:self._num_examples]

            self._shuffle()  # 打乱,在一个新的epoch里重新打乱
            self._index_in_epoch = 0
            return feature_rest_part, label_rest_part, last_feature_rest_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._dynamic_features[start:end], self._labels[start:end], self._last_features[start:end]

    def _shuffle(self):
        index = np.arange(self._num_examples)
        np.random.shuffle(index)
        self._dynamic_features = self._dynamic_features[index]
        self._labels = self._labels[index]
        self._last_features = self._last_features[index]

    @property
    def dynamic_features(self):
        return self._dynamic_features

    @property
    def labels(self):
        return self._labels

    @property
    def last_features(self):
        return self._last_features

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epoch_completed(self):
        return self._epoch_completed

    @property
    def batch_completed(self):
        return self._batch_completed

    @epoch_completed.setter
    def epoch_completed(self, value):
        self._epoch_completed = value


def read_data(max_times_of_visits=20):
    dataset_label = pd.read_csv("resources/preprocessed_label.csv", encoding='gbk')
    patient_id_list_repeat = dataset_label.iloc[:, 0]  # 切片
    patient_id_list = []
    [patient_id_list.append(i) for i in patient_id_list_repeat if not i in patient_id_list]

    dataset_feature = pd.read_csv("resources/preprocessed_feature.csv", encoding='gbk')
    labels_on_visits = []
    features_on_visit = []
    features_of_last = []
    for patient_id in patient_id_list:
        one_visit_labels = dataset_label.loc[dataset_label['patient_id'] == patient_id].iloc[:, 2:].as_matrix()
        one_visit_feature = dataset_feature.loc[dataset_feature['patient_id'] == patient_id].iloc[:, 3:].as_matrix()
        # todo 这里还要再取一下最后一次的患者特征
        if max_times_of_visits < one_visit_labels.shape[0] - 1:
            labels_on_visits.append(one_visit_labels[max_times_of_visits - 1])  # 取第max个标签，即max+1次的入院原因
            features_on_visit.append(one_visit_feature[0:max_times_of_visits])  # 取前max次的入院特征
            features_of_last.append(one_visit_feature[max_times_of_visits])  # 取第max+1次的入院特征
        else:
            labels_on_visits.append(one_visit_labels[one_visit_labels.shape[0] - 2])  # 取倒数第二个标签
            features_on_visit.append(one_visit_feature[0:one_visit_labels.shape[0] - 1])  # 取到倒数第二次入院的特征
            features_of_last.append(one_visit_feature[one_visit_feature.shape[0] - 1])  # 取最后一次入院的特征

    all_labels_on_visits = np.array(labels_on_visits)
    # todo 这里改成多分类了
    binary_labels_of_revisits = np.expand_dims(np.sign(np.sum(all_labels_on_visits[:, 5 + 0 * 11:10 + 0 * 11], 1)), 1)
    # binary_labels_of_revisits = np.expand_dims(np.sign(np.sum(all_labels_on_visits[:, 10 + 0 * 11:11 + 0 * 11], 1)), 1)
    # labels_of_revisits = all_labels_on_visits[:, 5:11]
    # labels_of_revisits = np.c_[labels_of_revisits, binary_labels_of_revisits]
    features_on_visit = list(
        map(lambda x: np.pad(x, ((0, max_times_of_visits - x.shape[0]), (0, 0)), 'constant', constant_values=0),
            features_on_visit))
    features_on_visit = np.stack(features_on_visit)
    features_of_last = np.stack(features_of_last)
    return DataSet(features_on_visit, binary_labels_of_revisits, features_of_last)


if __name__ == "__main__":
    x = read_data(7)
    print("ok")
