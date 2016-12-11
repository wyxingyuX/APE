import numpy as np
import random

import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Layer, Input, Dense, Embedding, Flatten, Merge, AveragePooling1D, Merge, Permute, merge
from keras.regularizers import WeightRegularizer, l1, l2, activity_l2, activity_l1


def get_ranking_loss(loss, batch_size_p, num_negatives, conf=None):
    # Losses defined here are sensitive to the number of negatives
    # assume first batch_size_p are positive, and left batch_size_p * num_negatives are negative

    # loss_base helps turn mean loss from num_negatives pairs into (1+num_negatives) losses so that keras can correctly match the shape
    loss_base = np.ones(((1 + num_negatives) * batch_size_p, 1), dtype=np.float32) / ((1 + num_negatives) * batch_size_p)

    def ranking_loss(y_true, y_pred):
        pos_interact_score = y_pred[:batch_size_p, :]
        neg_interact_score = y_pred[batch_size_p:, :]
        # weight_pos = y_true[:batch_size_p, 0]
        # weight_neg = y_true[batch_size_p:, 0]
        if loss == 'max-margin':
            print '[INFO!] y_true is not used so it may raise some warning.'
            try:
                margin = conf.loss_margin
            except:
                margin = 1
                print '[warning] loss margin not defined, %d is used.' % margin
            task_loss = K.mean(K.relu(neg_interact_score
                                      - K.reshape(K.repeat(pos_interact_score, num_negatives), (-1, 1))
                                      + margin))
        elif loss == 'log-loss':
            print '[INFO!] y_true is not used so it may raise some warning.'
            # a.k.a., bayesian personalized ranking
            task_loss = K.mean(-K.log(K.sigmoid(K.reshape(K.repeat(pos_interact_score, num_negatives), (-1, 1))
                                                - neg_interact_score)))
        elif loss == 'skip-gram':
            # average over mini-batch, for each (pos, k negs) it is log\sigma(s) + E[log\sigma(-s)]
            #task_loss = K.mean(-K.log(K.sigmoid(pos_interact_score))) + num_negatives * K.mean(-K.log(K.sigmoid(-neg_interact_score)))
            task_loss = K.sum(K.log(K.sigmoid(y_true * y_pred)))
            task_loss /= - batch_size_p
        else:
            assert False, '[ERROR!] loss %s not specified.' % loss
        return task_loss * loss_base

    return ranking_loss