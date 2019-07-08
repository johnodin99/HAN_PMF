from __future__ import print_function
import numpy as np
from numpy.random import RandomState
import pickle
import os
import copy


class PMF():
    '''
    a class for this Double Co-occurence Factorization model
    '''
    # initialize some paprameters

    def __init__(self, R, U, V, lambda_alpha=1e-2, lambda_beta=1e-2, latent_size=5, momentum=0.8,
                 lr=0.0001, iterations=20, seed=0):

        self.lambda_alpha = lambda_alpha
        self.lambda_beta = lambda_beta
        self.momentum = momentum
        self.R = R
        self.random_state = RandomState(seed)
        self.iterations = iterations
        self.lr = lr
        self.I = copy.deepcopy(self.R)
        self.I[self.I != 0] = 1

        self.count_users = np.size(R, 0)
        self.count_items = np.size(R, 1)
        self.U = U
        self.V = V
        self.seed =seed

        if self.U is None and self.V is None:
            #Random
            self.U = self.random_state.rand(self.count_users, latent_size)
            self.V = self.random_state.rand(self.count_items, latent_size)
            # Random np.random.normal(mu, sigma, 1000)
            #self.U = self.random_state.normal((0, 0.5), size=(self.count_users, latent_size))
            #self.V = self.random_state.normal((0, 0.5), size=(self.count_items, latent_size))
            #uniform
            #self.U = np.random.uniform(size=(self.count_users, latent_size))
            #self.V = np.random.uniform(size=(self.count_items, latent_size))

        else:
            self.U = U
            self.V = V

    def RMSE(self, predicts, truth):
        return np.sqrt(np.mean(np.square(predicts - truth)))

    def loss(self):
        # the loss function of the model
        loss = np.sum(self.I*(self.R-np.dot(self.U, self.V.T))**2) + self.lambda_alpha*np.sum(np.square(self.U))\
               + self.lambda_beta*np.sum(np.square(self.V))
        return loss

    def predict(self, data):
        index_data = np.array([[int(ele[0])-1, int(ele[1])-1] for ele in data], dtype=int)
        u_features = self.U.take(index_data.take(0, axis=1), axis=0)
        v_features = self.V.take(index_data.take(1, axis=1), axis=0)
        preds_value_array = np.sum(u_features*v_features, 1)
        return preds_value_array

    def train(self, train_data=None, valid_data =None):
        '''
        # training process
        :param train_data: train data with [[i,j],...] and this indacates that K[i,j]=rating
        :param lr: learning rate
        :param iterations: number of iterations
        :return: learned V, T and loss_list during iterations
        '''

        train_loss_list = []
        valid_rmse_list = []
        last_valid_rmse = None

        # momentum
        momentum_u = np.zeros(self.U.shape)
        momentum_v = np.zeros(self.V.shape)

        for it in range(self.iterations):
            # derivation of Vi
            grads_u = np.dot(self.I*(self.R-np.dot(self.U, self.V.T)), -self.V) + self.lambda_alpha*self.U

            # derivation of Tj
            grads_v = np.dot((self.I*(self.R-np.dot(self.U, self.V.T))).T, -self.U) + self.lambda_beta*self.V

            # update the parameters
            momentum_u = (self.momentum * momentum_u) + self.lr * grads_u
            momentum_v = (self.momentum * momentum_v) + self.lr * grads_v

            self.U = self.U - momentum_u
            self.V = self.V - momentum_v

            # training evaluation
            train_loss = self.loss()
            train_loss_list.append(train_loss)

            valid_predicts = self.predict(valid_data)
            # type numpy array index:0 user index:1 item index:2 rating index:3 timestamp
            valid_rmse = self.RMSE(valid_data[:, 2], valid_predicts)
            valid_rmse_list.append(valid_rmse)

            print('traning iteration:{: d} ,loss:{: f}, vali_rmse:{: f}'.format(it, train_loss, valid_rmse))

            if last_valid_rmse and (last_valid_rmse - valid_rmse) <= 0:
                print('convergence at iterations:{: d}'.format(it))
                break
            else:
                last_valid_rmse = valid_rmse

        return self.U, self.V, train_loss_list, valid_rmse_list


