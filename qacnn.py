# -*- coding: utf-8 -*-
import tensorflow as tf


class QACNN(object):
    def __init__(self, batch_size, filter_size, num_filters, sequence_length, hidden_size, embeddings, embedding_size, margin):
        self.batch_size = batch_size
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.margin = margin

        self.q = tf.placeholder(tf.int32, shape=[None, self.sequence_length])  # question
        self.ap = tf.placeholder(tf.int32, shape=[None, self.sequence_length])  # positive answer
        self.an = tf.placeholder(tf.int32, shape=[None, self.sequence_length])  # negative answer
        self.lr = tf.placeholder(tf.float32)

        with tf.variable_scope('embedding'):
            q_embed, ap_embed, an_embed = self.embedding_layer()
        with tf.variable_scope('HL', reuse=tf.AUTO_REUSE):
            h_q = self.hidden_layer(q_embed)
            h_ap = self.hidden_layer(ap_embed)
            h_an = self.hidden_layer(an_embed)
        with tf.variable_scope('CNN', reuse=tf.AUTO_REUSE):
            conv_q = self.convolutional_layer(h_q)
            conv_ap = self.convolutional_layer(h_ap)
            conv_an = self.convolutional_layer(h_an)

        cos_q_ap = self.calc_cosine(conv_q, conv_ap)
        cos_q_an = self.calc_cosine(conv_q, conv_an)

        self.loss, self.acc = self.calc_loss_and_acc(cos_q_ap, cos_q_an)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def embedding_layer(self):
        embeddings = tf.Variable(tf.to_float(self.embeddings), trainable=False, name="embeddings")
        q_embed = tf.nn.embedding_lookup(embeddings, self.q)
        ap_embed = tf.nn.embedding_lookup(embeddings, self.ap)
        an_embed = tf.nn.embedding_lookup(embeddings, self.an)
        return q_embed, ap_embed, an_embed

    def hidden_layer(self, x_embed):
        W = tf.get_variable('weights', shape=[self.embedding_size, self.hidden_size], initializer=tf.uniform_unit_scaling_initializer())
        b = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[self.hidden_size]))
        h_x = tf.reshape(tf.nn.tanh(tf.matmul(tf.reshape(x_embed, [-1, self.embedding_size]), W) + b), [self.batch_size, self.sequence_length, -1])
        return h_x

    def convolutional_layer(self, h_x):
        h_x = tf.expand_dims(h_x, -1)
        filter_shape = [self.filter_size, self.hidden_size, 1, self.num_filters]
        W = tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.get_variable(initializer=tf.constant(0.1, shape=[self.num_filters]), name="b")
        conv = tf.nn.conv2d(h_x, W, strides=[1, 1, 1, 1], padding='VALID', name="conv")
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        output = tf.nn.max_pool(h, ksize=[1, self.sequence_length - self.filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
        output = tf.nn.tanh(output)
        output = tf.reshape(output, [-1, self.num_filters])
        return output

    def calc_cosine(self, q, a):
        norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul_q_a = tf.reduce_sum(tf.multiply(q, a), 1)
        cosine = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))
        return cosine

    def calc_loss_and_acc(self, cos_q_ap, cos_q_an):
        zero = tf.fill(tf.shape(cos_q_ap), 0.0)
        margin = tf.fill(tf.shape(cos_q_ap), self.margin)
        losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(cos_q_ap, cos_q_an)))
        loss = tf.reduce_sum(losses)

        correct = tf.equal(zero, losses)
        acc = tf.reduce_mean(tf.cast(correct, "float"), name="acc")
        return loss, acc
