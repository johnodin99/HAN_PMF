import pickle
from pmf_model import *
import numpy as np
import os
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf

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

#parameter control
res_dir = FLAGS.res_dir
latent_size = FLAGS.latent_size
iterations = FLAGS.pmf_iteration
pmf_lambda_u = FLAGS.pmf_lambda_u
pmf_lambda_v = FLAGS.pmf_lambda_v
momentum = FLAGS.momentum
learning_rate = FLAGS.pmf_learning_rate

# split ratio
ratio = 0.8
Test_Experiment = "PMF"
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
            train_data = ratings_array[:int(ratio * ratings_array.shape[0])]
            valid_data = ratings_array[
                         int(ratio * ratings_array.shape[0]):int((ratio + (1 - ratio) / 2) * ratings_array.shape[0])]
            test_data = ratings_array[int((ratio + (1 - ratio) / 2) * ratings_array.shape[0]):]

            R_zeros = np.zeros((ratings["user"].max(), ratings["item"].max()))

            for element in train_data:
                R_zeros[int(element[0]) - 1, int(element[1]) - 1] = float(element[2])

            model_1 = PMF(R=R_zeros,
                          U=None,
                          V=None,
                          latent_size=latent_size,
                          iterations=iterations,
                          lambda_alpha=pmf_lambda_u,
                          lambda_beta=pmf_lambda_v,
                          lr=learning_rate,
                          momentum=momentum,
                          seed=seed)

            U, V, train_loss_list, valid_rmse_list = model_1.train(train_data=train_data, valid_data=valid_data)

            preds = model_1.predict(data=test_data)
            test_data_rmse = model_1.RMSE(predicts=preds, truth=test_data[:, 2])

            f1 = open(res_dir + '/' + Test_Experiment + '_' + "dataset_" + dataset + '_' + "seed_" + str(
                seed) + "_" + 'state.log', 'a')
            f1.write("###" + Test_Experiment + "###\n\n")
            f1.write("Dataset={}\n\n".format(dataset))
            f1.write("Seed={}\n\n".format(seed))
            f1.write("===Configuration===\n")
            f1.write("PMF: lambda_u={}, lambda_v={}\n\n".format(pmf_lambda_u, pmf_lambda_v))
            f1.write("learning_rate={}\n\n".format(learning_rate))
            f1.write("momentum={}\n\n".format(momentum))
            f1.write("latent_size={}\n\n".format(latent_size))
            f1.write("valid_rmse:{}\n".format(valid_rmse_list[-2]))
            f1.write("test_rmse :{}\n".format(test_data_rmse))
            f1.close()

            if not os.path.isfile(record_path):
                record = pd.DataFrame(columns=["Experiment", "dataset", "Seed", "Test"])
            else:
                record = pd.read_csv(record_path)

            record = record.append({"Experiment": Test_Experiment,
                                    "dataset": dataset,
                                    "Seed": seed,
                                    "Test": test_data_rmse},
                                   ignore_index=True)

            record.to_csv(record_path, index=False)

            print("model PMF test_data_rmse:")
            print(test_data_rmse)


