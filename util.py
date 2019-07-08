'''
Modified on June 20, 2017 
@author: Hao Wu, Zhengxin Zhang

'''

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

class Query:
    def __init__(self, user_id):
        self.user_id = user_id
        self.ground_truth = []
        self.selected_items = []
        
    def extendGroundTruth(self, item_ids):
        self.ground_truth.extend(item_ids)
        
    def appendGroundTruth(self, item_id):
        self.ground_truth.append(item_id)
        
    def appendSelectedItems(self, item_id):
        self.selected_items.append(item_id)
        
    def extendSelectedItems(self, item_ids):
        self.selected_items.extend(item_ids)
    def print(self):
        print("%d" % self.user_id)
        print(self.ground_truth)
        print(self.selected_items)

def eval_RATING(R, U, V, TS):
    num_user = U.shape[0]
    sub_mae = np.zeros(num_user)
    sub_mse = np.zeros(num_user)
    
    TS_count = 0
    for i in range(num_user):
        idx_item = TS[i]
        if len(idx_item) == 0:
            continue
        TS_count = TS_count + len(idx_item)
        approx_R_i = U[i].dot(V[idx_item].T)  # approx_R[i, idx_item]
        R_i = R[i]

        sub_mse[i] = np.square(approx_R_i - R_i).sum()
        sub_mae[i] = np.abs(approx_R_i - R_i).sum()

    
    mae = sub_mae.sum() / TS_count
    mse = sub_mse.sum() / TS_count
    rmse = np.sqrt(mse)
    return mae, mse, rmse


def eval_RATING_biased(R, U, V, TS, mu, bu, bi):
    num_user = U.shape[0]
    sub_mae = np.zeros(num_user)
    sub_mse = np.zeros(num_user)
    
    TS_count = 0
    for i in range(num_user):
        idx_item = TS[i]
        if len(idx_item) == 0:
            continue
        TS_count = TS_count + len(idx_item)
        approx_R_i = U[i].dot(V[idx_item].T)  # approx_R[i, idx_item]
        R_i = R[i]
        sub_mse[i] = np.square(mu + bu[i] + bi[idx_item] + approx_R_i - R_i).sum()
        sub_mae[i] = np.abs(mu + bu[i] + bi[idx_item] + approx_R_i - R_i).sum()
    
    mae = sub_mae.sum() / TS_count
    mse = sub_mse.sum() / TS_count
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def AP(ground_truth, test_decision):
    p_at_k = [0.0] * len(test_decision)
    C = 0
    for i in range(1, len(test_decision) + 1):
        rel = 0
        if test_decision[i - 1] in ground_truth:
            rel = 1
            C += 1
        p_at_k[i - 1] = rel * C / i
    if C==0:
        return 0.0        
    else:
        return np.sum(p_at_k) / C 

def NDCG(ground_truth, test_decision):
    dcg = 0
    C = 0
    for i in range(1, len(test_decision) + 1):
        rel = 0
        if test_decision[i - 1] in ground_truth:
            rel = 1
            C += 1        
        dcg += (np.power(2, rel) - 1) / np.log(i + 1)
    if C == 0:
        return 0
    idcg = 0
    for i in range(1, C + 1):
        idcg += (1 / np.log(i + 1))
    return dcg / idcg

def PrecisionRecall(ground_truth, test_decision):
    hit_set = list(set(ground_truth) & set(test_decision))
    precision = len(hit_set) / float(len(test_decision))
    recall = len(hit_set) / float(len(ground_truth))
    return precision, recall

def val_format(value_list, str_format):
    _str_list = []
    for val in value_list:
        _str_list.append(format(val, str_format))
    return _str_list
    
def eval_RANKING(U, V, QList, RankPos):
    _precision = [0] * len(RankPos)
    _recall = [0] * len(RankPos)
    _ndcg = [0]
    _map = [0]
    num_case = 0
    for q in QList:
        if len(q.ground_truth)==0:
            continue
        # build TOP_N ranking list 
        idx_item = q.selected_items
        ranklist = sorted(zip(U[q.user_id].dot(V[idx_item].T), idx_item), reverse=True)
        for n in range(len(RankPos)):
            sublist = ranklist[0:RankPos[n]]
            score, test_decision = zip(*sublist)
            p_at_k, r_at_k = PrecisionRecall(q.ground_truth, test_decision)
        # sum the metrics of decision
            _precision[n] += p_at_k
            _recall[n] += r_at_k
        sublist = ranklist[0:RankPos[-1]]
        score, test_decision = zip(*sublist)
        _ndcg[0] += NDCG(q.ground_truth, test_decision)
        _map[0] += AP(q.ground_truth, test_decision)
        # counting the number of test cases
        num_case += 1
        # calculate the final scores of metrics      
    for n in range(len(RankPos)):
        _precision[n] /= num_case
        _recall[n] /= num_case
    _ndcg[0] /= num_case
    _map[0] /= num_case
    return _precision, _recall, _ndcg, _map

def eval_RANKING_biased(U, V, QList, RankPos, mu, bu, bi):
    _precision = [0] * len(RankPos)
    _recall = [0] * len(RankPos)
    _ndcg = [0]
    _map = [0]
    num_case = 0
    for q in QList:
        if len(q.ground_truth) ==0:
            continue
        # build TOP_N ranking list 
        test_decision = []
        idx_item = q.selected_items
        R_H = U[q.user_id].dot(V[idx_item].T) + bi[idx_item]
        R_H += mu + bu[q.user_id]
        ranklist = sorted(zip(R_H, idx_item), reverse=True)
        for n in range(len(RankPos)):
            sublist = ranklist[0:RankPos[n]]
            score, test_decision = zip(*sublist)
            p_at_k, r_at_k = PrecisionRecall(q.ground_truth, test_decision)
        # sum the metrics of decision
            _precision[n] += p_at_k
            _recall[n] += r_at_k
        sublist = ranklist[0:RankPos[-1]]
        score, test_decision = zip(*sublist)
        _ndcg[0] += NDCG(q.ground_truth, test_decision)
        _map[0] += AP(q.ground_truth, test_decision)
        # counting the number of test cases
        num_case += 1
        # calculate the final scores of metrics      
    for n in range(len(RankPos)):
        _precision[n] /= num_case
        _recall[n] /= num_case
    _ndcg[0] /= num_case
    _map[0] /= num_case
    return _precision, _recall, _ndcg, _map           

def histogram(xlabel, data=[]):
    (mu, sigma) = norm.fit(data)
    # the histogram of the data
    n, bins, patches = plt.hist(data, 50, normed=1)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--', linewidth=2)
    # plot
    plt.xlabel(xlabel)
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ Length:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
    plt.grid(True)
    plt.show()

def expand_label(y):
    """
    transform an 1-d y array of (length, ) to shape(length, 2)
    :param y:
    :return:
    """
    return np.array([np.ones_like(y) - y, y]).T

def index_to_input(X, maxlen_sent, maxlen_doc):
    """
    transform the index-list based input to some data that can be fed into the CNN_GRU_module class
    """
    X = [pad_sequences(i, maxlen=maxlen_sent) for i in X]
    X = pad_2Dsequences(X, maxlen=maxlen_doc)
    return np.array([x.flatten() for x in X])
    
    
def pad_2Dsequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    '''modify the keras.preprocessing.sequence to make it padding on doc level'''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    nb_dims = sequences[0].shape[1]
    
    if maxlen is None:
        maxlen = np.max(lengths)
    
    x = (np.ones((nb_samples, maxlen, nb_dims)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)
    
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
    
    return x


def make_CDL_format(X_base, path):
    max_X = X_base.max(1).toarray()
    for i in range(max_X.shape[0]):
        if max_X[i, 0] == 0:
            max_X[i, 0] = 1
    max_X_rep = np.tile(max_X, (1, X_base.shape[1]))
    X_nor = X_base / max_X_rep
    np.savetxt(path + '/mult_nor.dat', X_nor, fmt='%.5f')
    