# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow as tf
import data_helpers
from qacnn import QACNN


def main():
    trained_model = "checkpoints/model.ckpt"
    embedding_size = 100  # Word embedding dimension
    epochs = 10
    batch_size = 128  # Batch data size
    filter_size = 3
    num_filters = 256
    sequence_length = 300  # Sentence length
    hidden_size = 128  # Number of hidden layer neurons
    learning_rate = 0.01  # Learning rate
    lrdown_rate = 0.9
    margin = 0.1
    gpu_mem_usage = 0.75
    gpu_device = "/gpu:0"
    cpu_device = "/cpu:0"

    embeddings, word2idx = data_helpers.load_embedding('vectors.nobin')
    voc = data_helpers.load_vocab('D:\\DataMining\\Datasets\\insuranceQA\\V1\\vocabulary')
    all_answers = data_helpers.load_answers('D:\\DataMining\\Datasets\\insuranceQA\\V1\\answers.label.token_idx', voc)
    questions, pos_answers, neg_answers = data_helpers.load_train_data('D:\\DataMining\\Datasets\\insuranceQA\\V1\\question.train.token_idx.label', all_answers, voc, word2idx, 300)
    data_size = len(questions)
    permutation = np.random.permutation(data_size)
    questions = questions[permutation, :]
    pos_answers = pos_answers[permutation, :]
    neg_answers = neg_answers[permutation, :]
    with tf.Graph().as_default(), tf.device(gpu_device):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_usage)
        session_conf = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        model = QACNN(batch_size, filter_size, num_filters, sequence_length, hidden_size, embeddings, embedding_size, margin)
        with tf.Session(config=session_conf).as_default() as sess:  # config=session_conf
            saver = tf.train.Saver()

            print("Start training")
            sess.run(tf.global_variables_initializer())  # Initialize all variables
            for epoch in range(epochs):
                print("The training of the %s iteration is underway" % (epoch + 1))
                batch_number = 1
                for question, pos_answer, neg_answer in data_helpers.batch_iter(questions, pos_answers, neg_answers, batch_size):
                    start_time = time.time()
                    feed_dict = {
                        model.q: question,
                        model.ap: pos_answer,
                        model.an: neg_answer,
                        model.lr: learning_rate
                    }
                    _, loss, acc = sess.run([model.train_op, model.loss, model.acc], feed_dict)
                    duration = time.time() - start_time
                    print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tAcc %2.3f' % (epoch + 1, batch_number * batch_size, data_size, duration, loss, acc))
                    batch_number += 1
                learning_rate *= lrdown_rate
                saver.save(sess, trained_model)
            print("End of the training")


if __name__ == '__main__':
    main()
