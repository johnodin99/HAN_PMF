import pickle
import pmf_model
import numpy as np
import han_model
import tensorflow as tf
import os
import pandas as pd
import time
from sklearn.utils import shuffle
from data_manager import Data_Factory
import pmf_model
from cnn_gru_model import CNN_GRU_module
from util import index_to_input
import win_unicode_console
win_unicode_console.enable()

# Data loading params
# CNN GRU parameter TF.FLAGS
tf.flags.DEFINE_integer("vocab_size", 8000, "vocabulary size")
tf.flags.DEFINE_integer("embedding_size", 60, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("nb_filters", 100, "nb_filters")
tf.flags.DEFINE_float("dropout_rate", 0, "dropout_rate")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
# Control thw max word and the max sentence by threshold
tf.flags.DEFINE_float("threshold_length_document", 0.5, "threshold_length_document")
tf.flags.DEFINE_float("threshold_length_sentence", 0.8, "threshold_length_sentence")
# PMF parameter
tf.flags.DEFINE_integer("latent_size", 5, "latent_size")
tf.flags.DEFINE_integer("pmf_iteration", 200, "pmf_iteration")
tf.flags.DEFINE_float("pmf_lambda_u", 0.01, "lambda_u")
tf.flags.DEFINE_float("pmf_lambda_v", 0.01, "lambda_v")
tf.flags.DEFINE_float("pmf_learning_rate", 0.0005, "pmf learning rate")
tf.flags.DEFINE_float("momentum", 0.8, "momentum")
# Experiment Result Directory
tf.flags.DEFINE_string("res_dir", "./res_dir", "result directory")
# Control Experiment Data
tf.flags.DEFINE_list("datasets", ["Amazon_Instant_Video_5",
                                  "Automotive_5",
                                  "Digital_Music_5",
                                  "Musical_Instruments_5",
                                  "Office_Products_5",
                                  "Patio_Lawn_and_Garden_5",
                                  "Apps_for_Android_5",
                                  "Kindle_Store_5"],
                     "datasets")
tf.flags.DEFINE_list("seeds", [55, 66, 77, 88, 99], "seeds")

tf.flags.DEFINE_integer("get_result_times", 1, "get_result_num")
FLAGS = tf.flags.FLAGS


#Data set Choose
# Amazon_Instant_Video_5 User:5130 Item:1685 Rating 37126 Density 0.429%
# Automotive_5  User:2928 Item:1835 Rating 20473 Density 0.381%
# Digital_Music_5 User:5541 Item:3568 Rating 64706 Density 0.327%
# Musical_Instruments_5 User:1429 Item:900 Rating 10261 Density 0.798%
# Office_Products_5 User:4905 Item:2420 Rating 53258 Density 0.449%
# Patio_Lawn_and_Garden_5  User:1686 Item:962 Rating 13272 Density 0.818%

# Apps_for_Android_5 User:87271 Item:13209 Rating 752937 Density 0.065%
# Kindle_Store_5  User:68223 Item:61934 Rating 982619 Density 0.023%

Test_Experiment ="CNN_GRU_User"
record_path = "record.csv"

datasets = FLAGS.datasets
# override
datasets = ["Musical_Instruments_5"]

seeds = FLAGS.seeds
# override
seeds = [55]

for get_experiment_result in range(FLAGS.get_result_times):
    for dataset in datasets:
        for seed in seeds:
            from numpy.random import seed as set_seed

            set_seed(seed=seed)
            from tensorflow import set_random_seed

            set_random_seed(seed=seed)

            src_document_all = os.path.join(os.getcwd(), "data", dataset, "document.all")
            src_ratings_data = os.path.join(os.getcwd(), "data", dataset, "ratings.dat")
            np.random.seed(seed)

            ratings = pd.read_csv(src_ratings_data,
                                  sep="::",
                                  names=["user", "item", "rating", "timestamp"],
                                  engine='python')
            ratings_array = ratings.values
            ratings_array = shuffle(ratings_array, random_state=seed)
            ratio = 0.8
            train_data = ratings_array[:int(ratio * ratings_array.shape[0])]
            valid_data = ratings_array[
                         int(ratio * ratings_array.shape[0]):int((ratio + (1 - ratio) / 2) * ratings_array.shape[0])]
            test_data = ratings_array[int((ratio + (1 - ratio) / 2) * ratings_array.shape[0]):]

            # Dataset read
            with open(src_document_all, 'rb') as document:
                document_all = pickle.load(document)

            # User
            user_sequence2D = document_all["Y_sequence2D"]
            len_doc_user = [len(profile) for profile in user_sequence2D]
            len_sent_user = []
            for profile in user_sequence2D:
                for review in profile:
                    len_sent_user.append(len(review))
            len_doc_user = sorted(len_doc_user)
            len_sent_user = sorted(len_sent_user)
            maxlen_doc_user = len_doc_user[int(len(len_doc_user) * FLAGS.threshold_length_document)]
            maxlen_sent_user = len_sent_user[int(len(len_sent_user) * FLAGS.threshold_length_sentence)]

            print("user count")
            print(len(user_sequence2D))
            print("maxlen_doc_user:")
            print(maxlen_doc_user)
            print("maxlen_sent_user:")
            print(maxlen_sent_user)

            R_zeros = np.zeros((ratings["user"].max(), ratings["item"].max()))
            for element in train_data:
                R_zeros[int(element[0]) - 1, int(element[1]) - 1] = float(element[2])

            user_description = np.zeros((len(user_sequence2D), maxlen_doc_user, maxlen_sent_user), dtype=np.int32)

            # user description data clean shape
            for i in range(len(user_sequence2D)):
                for j in range(len(user_sequence2D[i])):
                    for k in range(len(user_sequence2D[i][j])):
                        if j < maxlen_doc_user and k < maxlen_sent_user:
                            user_description[i][j][k] = user_sequence2D[i][j][k]

            pmf_model_1 = pmf_model.PMF(R=R_zeros, U=None, V=None,
                                        latent_size=FLAGS.latent_size,
                                        iterations=FLAGS.pmf_iteration,
                                        lambda_alpha=FLAGS.pmf_lambda_u,
                                        lambda_beta=FLAGS.pmf_lambda_v,
                                        lr=FLAGS.pmf_learning_rate,
                                        momentum=FLAGS.momentum,
                                        seed=seed)
            U, V, train_loss_list, valid_rmse_list = pmf_model_1.train(train_data=train_data, valid_data=valid_data)
            test_data_rmse_list = []
            last_test_data_rmse = None

            if not os.path.exists(FLAGS.res_dir):
                os.makedirs(FLAGS.res_dir)
            f1 = open(FLAGS.res_dir + '/' + Test_Experiment + '_' + "dataset_" + dataset + '_' + "seed_" + str(
                seed) + "_" + 'state.log', 'a')
            f1.write("###" + Test_Experiment + "###\n\n")
            f1.write("Dataset={}\n\n".format(dataset))
            f1.write("Seed={}\n\n".format(seed))
            f1.write("user count={}\n\n".format(len(user_sequence2D)))
            f1.write(" latent_size={}\n\n".format(FLAGS.latent_size))
            f1.write("maxlen_doc_user={}\n\n".format(maxlen_doc_user))
            f1.write("maxlen_sent_user={}\n\n".format(maxlen_sent_user))
            f1.write("===Configuration===\n")
            f1.write("vocab_size={},embedding_size={}, "
                     .format(FLAGS.vocab_size, FLAGS.embedding_size))
            f1.write("PMF: lambda_u={}, lambda_v={}\n\n".format(FLAGS.pmf_lambda_u, FLAGS.pmf_lambda_v))
            f1.write("learning_rate={}\n\n".format(FLAGS.pmf_learning_rate))
            f1.write("momentum={}\n\n".format(FLAGS.momentum))

            model_user = CNN_GRU_module(output_dimesion=FLAGS.latent_size,
                                        vocab_size=FLAGS.vocab_size,
                                        dropout_rate=FLAGS.dropout_rate,
                                        emb_dim=FLAGS.embedding_size,
                                        gru_outdim=FLAGS.latent_size,
                                        maxlen_doc=maxlen_doc_user,
                                        maxlen_sent=maxlen_sent_user,
                                        nb_filters=FLAGS.nb_filters,
                                        init_W=None)
            user_sequence2D = index_to_input(user_sequence2D, maxlen_doc_user, maxlen_sent_user)
            user_weight = np.ones(len(user_sequence2D), dtype=float)

            for epoch in range(FLAGS.num_epochs):
                f1.write("Epoch:{}\n".format(epoch + 1))
                print('current epoch %s' % (epoch + 1))

                model_user.train(user_sequence2D, U, user_weight, seed)
                U = model_user.get_projection_layer(user_sequence2D)

                pmf_model_2 = pmf_model.PMF(R=R_zeros, U=U, V=V,
                                            latent_size=FLAGS.latent_size,
                                            iterations=FLAGS.pmf_iteration,
                                            lambda_alpha=FLAGS.pmf_lambda_u,
                                            lambda_beta=FLAGS.pmf_lambda_v,
                                            lr=FLAGS.pmf_learning_rate,
                                            momentum=FLAGS.momentum,
                                            seed=seed)

                U, V, train_loss_list, valid_rmse_list = pmf_model_2.train(train_data=train_data, valid_data=valid_data)

                f1.write("valid_rmse:{}\n".format(valid_rmse_list[-2]))

                predicts = pmf_model_2.predict(data=test_data)
                test_data_rmse = pmf_model_2.RMSE(predicts=predicts, truth=test_data[:, 2])
                print('current epoch %s test data rmse' % (epoch + 1))
                print(test_data_rmse)
                test_data_rmse_list.append(test_data_rmse)

                if (last_test_data_rmse is not None) and (last_test_data_rmse - test_data_rmse <= 0):
                    print("Convergence at epoch:%s" % str(epoch + 1))
                    f1.write("test_rmse :{}\n".format(test_data_rmse_list[-1]))
                    f1.write("\n\nconvergence at epoch:{}".format(epoch + 1))
                    f1.write("\n\nThe Best Performance at epoch:{}\n".format(epoch))

                    rmse_raise = test_data_rmse_list[0] - min(test_data_rmse_list[:])
                    min_test_data_rmse = min(test_data_rmse_list[:])

                    f1.write("Epoch 1 rmse - Epoch{} rmse = {}\n".format(epoch, rmse_raise))
                    print("Epoch 1 rmse - Epoch{} rmse = {}".format(epoch, rmse_raise))
                    f1.close()
                    if not os.path.isfile(record_path):
                        record = pd.DataFrame(columns=["Experiment", "dataset", "Seed", "embedding_size",
                                                       "nb_filters", "batch_size", "latent_size", "pmf_lambda_u",
                                                       "pmf_lambda_v", "pmf_learning_rate", "momentum", "Test"])
                    else:
                        record = pd.read_csv(record_path)

                    record = record.append({"Experiment": Test_Experiment,
                                            "dataset": dataset,
                                            "Seed": seed,
                                            "embedding_size": FLAGS.embedding_size,
                                            "nb_filters": FLAGS.nb_filters,
                                            "batch_size": None,
                                            "latent_size": FLAGS.latent_size,
                                            "pmf_lambda_u": FLAGS.pmf_lambda_u,
                                            "pmf_lambda_v": FLAGS.pmf_lambda_v,
                                            "pmf_learning_rate": FLAGS.pmf_learning_rate,
                                            "momentum": FLAGS.momentum,
                                            "Test": min_test_data_rmse,
                                            },
                                           ignore_index=True)

                    record.to_csv(record_path, index=False)

                    break

                else:
                    last_test_data_rmse = test_data_rmse
                    f1.write("test_rmse_:{}\n".format(last_test_data_rmse))
                    print("last_test_data_rmse")
                    print(last_test_data_rmse)







