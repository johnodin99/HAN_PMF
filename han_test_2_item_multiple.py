from sklearn.utils import shuffle
import pickle
import numpy as np
import han_model
import pmf_model
import tensorflow as tf
import os
import pandas as pd
import time
import win_unicode_console
win_unicode_console.enable()


# Data loading params
# Han parameter TF.FLAGS
tf.flags.DEFINE_integer("vocab_size", 8000, "vocabulary size")
tf.flags.DEFINE_integer("embedding_size", 30, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 35, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_float("han_learning_rate", 0.001, "han learning rate")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
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
#################################
#Data set Choose
# Amazon_Instant_Video_5 User:5130 Item:1685 Rating 37126 Density 0.429%
# Automotive_5  User:2928 Item:1835 Rating 20473 Density 0.381%
# Digital_Music_5 User:5541 Item:3568 Rating 64706 Density 0.327%
# Musical_Instruments_5 User:1429 Item:900 Rating 10261 Density 0.798%
# Office_Products_5 User:4905 Item:2420 Rating 53258 Density 0.449%
# Patio_Lawn_and_Garden_5  User:1686 Item:962 Rating 13272 Density 0.818%

# Apps_for_Android_5 User:87271 Item:13209 Rating 752937 Density 0.065%
# Kindle_Store_5  User:68223 Item:61934 Rating 982619 Density 0.023%

Test_Experiment ="Han_Item"
record_path = "record.csv"

datasets = FLAGS.datasets
# override
datasets = ["Apps_for_Android_5"]

seeds = FLAGS.seeds
# override
# seeds = [55, 66, 77, 88, 99]

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

            # item
            item_sequence2D = document_all["X_sequence2D"]
            len_doc_item = [len(profile) for profile in item_sequence2D]
            len_sent_item = []
            for profile in item_sequence2D:
                for review in profile:
                    len_sent_item.append(len(review))
            len_doc_item = sorted(len_doc_item)
            len_sent_item = sorted(len_sent_item)
            maxlen_doc_item = len_doc_item[int(len(len_doc_item) * FLAGS.threshold_length_document)]
            maxlen_sent_item = len_sent_item[int(len(len_sent_item) * FLAGS.threshold_length_sentence)]

            print("item count")
            print(len(item_sequence2D))
            print("maxlen_doc_item:")
            print(maxlen_doc_item)
            print("maxlen_sent_item:")
            print(maxlen_sent_item)

            R_zeros = np.zeros((ratings["user"].max(), ratings["item"].max()))
            for element in train_data:
                R_zeros[int(element[0]) - 1, int(element[1]) - 1] = float(element[2])

            item_description = np.zeros((len(item_sequence2D), maxlen_doc_item, maxlen_sent_item), dtype=np.int32)

            # item description data clean shape
            for i in range(len(item_sequence2D)):
                for j in range(len(item_sequence2D[i])):
                    for k in range(len(item_sequence2D[i][j])):
                        if j < maxlen_doc_item and k < maxlen_sent_item:
                            item_description[i][j][k] = item_sequence2D[i][j][k]

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

            tf.reset_default_graph()
            with tf.Session() as sess:
                # User
                han_user = han_model.HAN(vocab_size=FLAGS.vocab_size,
                                         num_classes=FLAGS.latent_size,
                                         embedding_size=FLAGS.embedding_size,
                                         hidden_size=FLAGS.hidden_size,
                                         seed=seed)

                with tf.name_scope('loss_user'):
                    loss_user = tf.losses.mean_squared_error(labels=han_user.input_y, predictions=han_user.out)

                with tf.name_scope('accuracy_user'):
                    predict_user = han_user.out
                    label_user = han_user.input_y
                    acc_user = tf.reduce_mean(tf.cast(tf.equal(predict_user, label_user), tf.float32))

                with tf.name_scope('predict_user'):
                    predict_user = han_user.out

                global_step_user = tf.Variable(0, trainable=False)
                optimizer = tf.train.AdamOptimizer(FLAGS.han_learning_rate)

                # Item
                han_item = han_model.HAN(vocab_size=FLAGS.vocab_size,
                                         num_classes=FLAGS.latent_size,
                                         embedding_size=FLAGS.embedding_size,
                                         hidden_size=FLAGS.hidden_size,
                                         seed=seed)

                with tf.name_scope('loss_item'):
                    loss_item = tf.losses.mean_squared_error(labels=han_item.input_y, predictions=han_item.out)

                with tf.name_scope('accuracy_item'):
                    predict_item = han_item.out
                    label_item = han_item.input_y
                    acc_item = tf.reduce_mean(tf.cast(tf.equal(predict_item, label_item), tf.float32))

                with tf.name_scope('predict_item'):
                    predict_item = han_item.out

                global_step_item = tf.Variable(0, trainable=False)

                tvars_item = tf.trainable_variables()
                grads_item, _ = tf.clip_by_global_norm(tf.gradients(loss_item, tvars_item), FLAGS.grad_clip)
                grads_and_vars_item = tuple(zip(grads_item, tvars_item))
                train_optimizer_item = optimizer.apply_gradients(grads_and_vars_item, global_step=global_step_item)


                def train_step_item(x_batch, y_batch):
                    feed_dict = {
                        han_item.input_x: x_batch,
                        han_item.input_y: y_batch,
                        han_item.max_sentence_num: maxlen_doc_item,
                        han_item.max_sentence_length: maxlen_sent_item,
                        han_item.batch_size: FLAGS.batch_size
                    }
                    _, step, cost, accuracy = sess.run([train_optimizer_item, global_step_item, loss_item, acc_item],
                                                       feed_dict)
                    time_str = str(int(time.time()))
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))

                    return step


                def get_Prediction_item(x_batch, y_batch):
                    feed_dict = {
                        han_item.input_x: x_batch,
                        han_item.input_y: y_batch,
                        han_item.max_sentence_num: maxlen_doc_item,
                        han_item.max_sentence_length: maxlen_sent_item,
                        han_item.batch_size: FLAGS.batch_size
                    }
                    prediction_result = sess.run([predict_item], feed_dict)

                    return prediction_result


                sess.run(tf.global_variables_initializer())

                if not os.path.exists(FLAGS.res_dir):
                    os.makedirs(FLAGS.res_dir)
                f1 = open(FLAGS.res_dir + '/' + Test_Experiment + '_' + "dataset_" + dataset + '_' + "seed_" + str(
                    seed) + "_" + 'state.log', 'a')
                f1.write("###" + Test_Experiment + "###\n\n")
                f1.write("Dataset={}\n\n".format(dataset))
                f1.write("Seed={}\n\n".format(seed))
                f1.write("item count={}\n\n".format(len(item_sequence2D)))
                f1.write(" latent_size={}\n\n".format(FLAGS.latent_size))
                f1.write("maxlen_doc_item={}\n\n".format(maxlen_doc_item))
                f1.write("maxlen_sent_item={}\n\n".format(maxlen_sent_item))
                f1.write("===Configuration===\n")
                f1.write("HAN vocab_size={},embedding_size={},hidden_size={},batch_size={}, "
                         .format(FLAGS.vocab_size, FLAGS.embedding_size, FLAGS.hidden_size, FLAGS.batch_size))
                f1.write("HAN_learning_rate={}\n\n".format(FLAGS.han_learning_rate))
                f1.write("PMF: lambda_u={}, lambda_v={}\n\n".format(FLAGS.pmf_lambda_u, FLAGS.pmf_lambda_v))
                f1.write("PMF_learning_rate={}\n\n".format(FLAGS.pmf_learning_rate))
                f1.write("PMF momentum={}\n\n".format(FLAGS.momentum))

                ###########################################################################################################
                # epoch
                for epoch in range(FLAGS.num_epochs):
                    f1.write("Epoch:{}\n".format(epoch + 1))
                    # item
                    train_item_description = item_description
                    train_item_latent_vector = V
                    train_predict_item_all_list = []

                    print('current epoch %s' % (epoch + 1))

                    for i in range(0, len(train_item_description), FLAGS.batch_size):
                        x = train_item_description[i:i + FLAGS.batch_size]
                        y = train_item_latent_vector[i:i + FLAGS.batch_size]
                        step = train_step_item(x, y)

                    for i in range(0, len(train_item_description), FLAGS.batch_size):
                        x = train_item_description[i:i + FLAGS.batch_size]
                        y = train_item_latent_vector[i:i + FLAGS.batch_size]
                        train_predict_result = get_Prediction_item(x, y)
                        temp_train_predict_result = list(train_predict_result[0])
                        train_predict_item_all_list.extend(temp_train_predict_result)

                    V = np.asarray(train_predict_item_all_list)

                    pmf_model_2 = pmf_model.PMF(R=R_zeros, U=U, V=V,
                                                latent_size=FLAGS.latent_size,
                                                iterations=FLAGS.pmf_iteration,
                                                lambda_alpha=FLAGS.pmf_lambda_u,
                                                lambda_beta=FLAGS.pmf_lambda_v,
                                                lr=FLAGS.pmf_learning_rate,
                                                momentum=FLAGS.momentum,
                                                seed=seed)

                    U, V, train_loss_list, valid_rmse_list = pmf_model_2.train(train_data=train_data,
                                                                               valid_data=valid_data)

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
                            record = pd.DataFrame(columns=["Experiment", "dataset", "Seed", "Test"])
                        else:
                            record = pd.read_csv(record_path)

                        record = record.append({"Experiment": Test_Experiment,
                                                "dataset": dataset,
                                                "Seed": seed,
                                                "Test": min_test_data_rmse},
                                               ignore_index=True)

                        record.to_csv(record_path, index=False)

                        break
                    else:
                        last_test_data_rmse = test_data_rmse
                        f1.write("test_rmse_:{}\n".format(last_test_data_rmse))
                        print("last_test_data_rmse")
                        print(last_test_data_rmse)






