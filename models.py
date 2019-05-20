import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
import time


class LstmGanModel(object):
    def __init__(self, num_features, time_steps,
                 lstm_size=200, n_output=1, batch_size=64,
                 epochs=1000,
                 output_n_epoch=10,
                 learning_rate=0.01, max_loss=0.5, max_pace=0.01, lasso=0.0, ridge=0.0,
                 optimizer=tf.train.AdamOptimizer, name='LSTM-GAN'):
        self._num_features = num_features
        self._epochs = epochs
        self._name = name
        self._batch_size = batch_size
        self._output_n_epoch = output_n_epoch
        self._lstm_size = lstm_size
        self._n_output = n_output
        self._time_steps = time_steps
        self._max_loss = max_loss
        self._max_pace = max_pace
        self._lasso = lasso
        self._ridge = ridge
        self._optimizer = optimizer

        print("learning_rate=", learning_rate, "max_loss=", max_loss, "max_pace=", max_pace, "lasso=", lasso, "ridge=",
              ridge)

        self._graph_definition()

    def _graph_definition(self):
        self._placeholder_definition()  # 定义占位符
        self._sess = tf.Session()  # 会话
        with tf.variable_scope('generator'):
            self._hidden_layer()  # 定义中间层（输入至输出全流程）
            self._output = tf.contrib.layers.fully_connected(self._hidden_rep, self._n_output,
                                                             activation_fn=tf.identity)
            self._pred = tf.nn.sigmoid(self._output, name="pred")  # 预测概率
            self._hidden_state_decoder()  # 定义解码器
            # 添加正则

        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            self._fake_logits, self._fake_pred = self._hidden_layer_of_discriminator(self._predicted_last_x)
            self._real_logits, self._real_pred = self._hidden_layer_of_discriminator(self._last_x)

        self._gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self._fake_logits,
                                                                                labels=tf.ones_like(self._fake_logits)))
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self._fake_logits, labels=tf.zeros_like(self._fake_logits)))
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self._real_logits,
                                                                           labels=tf.ones_like(self._real_logits)))
        self._loss_of_discriminator = tf.add(fake_loss, real_loss)

        self._loss()
        self._loss_regulation()

        train_vars = tf.trainable_variables()
        # 生成器变量
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        # 判别器变量
        dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]
        self._generator_train_op = self._optimizer.minimize(self._loss_of_whole_generator, var_list=gen_vars)  # 定义训练
        self._discriminator_train_op = self._optimizer.minimize(self._loss_of_discriminator, var_list=dis_vars)

    def _hidden_layer_of_discriminator(self, samples):
        logits = tf.layers.dense(samples, 1)
        pred = tf.nn.sigmoid(logits)
        return logits, pred

    def _placeholder_definition(self):
        self._x = tf.placeholder(tf.float32, [None, self._time_steps, self._num_features], 'input')
        self._y = tf.placeholder(tf.float32, [None, self._n_output], 'label')
        self._last_x = tf.placeholder(tf.float32, [None, self._num_features], "true_last_x")
        self._keep_prob = tf.placeholder(tf.float32)

    def _hidden_layer(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)
        init_state = lstm.zero_state(tf.shape(self._x)[0], tf.float32)  # 全零向量

        mask, length = self._length()  # 每个病人的实际天数
        self._hidden, self._final_state = tf.nn.dynamic_rnn(lstm,
                                                            self._x,
                                                            sequence_length=length,
                                                            initial_state=init_state)
        self._hidden_rep = self._final_state.h

    def _hidden_state_decoder(self):
        # W_decoders = tf.Variable(xavier_init(self._n_output, self._lstm_size, self._num_features))
        # b_decoders = tf.Variable(tf.zeros(self._n_output))
        # decoders = tf.keras.backend.dot(self._hidden_rep, W_decoders) + tf.tile(tf.expand_dims(b_decoders, 1),
        #                                                                         [1, self._num_features])
        # self._predicted_last_x = tf.reshape(tf.matmul(tf.expand_dims(self._pred, 1), decoders),
        #                                     [-1, self._num_features])
        self._predicted_last_x = tf.contrib.layers.fully_connected(self._hidden_rep, self._num_features,
                                                                   activation_fn=tf.identity)

    def _loss(self):
        self._loss_classification = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y, logits=self._output))
        self._loss_of_whole_generator = self._loss_classification + 1*self._gen_loss

    def _loss_regulation(self):
        if self._lasso != 0:
            for trainable_variables in tf.trainable_variables("generator"):
                self._loss_of_whole_generator += tf.contrib.layers.l1_regularizer(self._lasso)(trainable_variables)
            for trainable_variables in tf.trainable_variables("discriminator"):
                self._loss_of_discriminator += tf.contrib.layers.l1_regularizer(self._lasso)(trainable_variables)
        if self._ridge != 0:
            for trainable_variables in tf.trainable_variables("generator"):
                self._loss_of_whole_generator += tf.contrib.layers.l2_regularizer(self._ridge)(trainable_variables)
            for trainable_variables in tf.trainable_variables("discriminator"):
                self._loss_of_discriminator += tf.contrib.layers.l2_regularizer(self._ridge)(trainable_variables)

    def _length(self):
        mask = tf.sign(tf.reduce_max(tf.abs(self._x), 2))  # 每个step若有实际数据则为1，只有填零数据则为0
        length = tf.reduce_sum(mask, 1)  # 每个sample的实际step数
        length = tf.cast(length, tf.int32)  # 类型转换
        return mask, length

    def _train_single_batch(self, dynamic_features, labels, last_features):
        self._sess.run(self._generator_train_op, feed_dict={self._x: dynamic_features,
                                                            self._y: labels,
                                                            self._last_x: last_features})
        self._sess.run(self._discriminator_train_op, feed_dict={self._x: dynamic_features,
                                                                self._y: labels,
                                                                self._last_x: last_features})

    def _loss_on_training_set(self, data_set):
        return self._sess.run(
            (self._loss_of_whole_generator, self._loss_classification, self._gen_loss, self._loss_of_discriminator),
            feed_dict={self._x: data_set.dynamic_features,
                       self._y: data_set.labels,
                       self._last_x: data_set.last_features})

    def fit(self, train_set, test_set):
        self._sess.run(tf.global_variables_initializer())
        train_set.epoch_completed = 0

        for c in tf.trainable_variables(self._name):
            print(c.name)

        print("auc\tepoch\tloss\tloss_diff\tloss_classification\tloss_generator\tloss_discriminator")
        logged = set()
        loss = 0
        while train_set.epoch_completed < self._epochs:
            dynamic_features, labels, last_features = train_set.next_batch(self._batch_size)

            if train_set.batch_completed == 1:
                loss = self.show_training(train_set, test_set, loss)  # 展示初始loss

            self._train_single_batch(dynamic_features, labels, last_features)
            if train_set.epoch_completed != 0 and train_set.epoch_completed % self._output_n_epoch == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)
                loss = self.show_training(train_set, test_set, loss)

    def predict(self, test_set):
        return np.expand_dims(
            self._sess.run(self._pred, feed_dict={self._x: test_set.dynamic_features, self._keep_prob: 1})[:, -1],
            1)

    def show_training(self, train_set, test_set, loss):
        loss_prev = loss
        loss, loss_classification, loss_generator, loss_discriminator = self._loss_on_training_set(train_set)
        loss_diff = loss_prev - loss
        y_score = self.predict(test_set)  # 此处计算和打印auc仅供调参时观察auc变化用，可删除，与最终输出并无关系
        auc = roc_auc_score(test_set.labels[:, -1], y_score)
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(auc, train_set.epoch_completed, loss, loss_diff,
                                                      loss_classification, loss_generator, loss_discriminator,
                                                      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        return loss

    @property
    def name(self):
        return self._name

    def close(self):
        self._sess.close()
        tf.reset_default_graph()
