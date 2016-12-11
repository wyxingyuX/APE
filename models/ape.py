import numpy as np
import random

import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Layer, Input, Dense, Embedding, Flatten, Merge, AveragePooling1D, Merge, Permute, merge, Lambda
from keras.regularizers import WeightRegularizer, l1, l2, activity_l2, activity_l1
from keras.optimizers import SGD, RMSprop, Adam

import objectives
reload(objectives)
from objectives import get_ranking_loss


class Triangularize(Layer):
    def __init__(self, size, **kwargs):
        self.size = size
        self.filter = np.ones((size, size), dtype=np.float32)
        self.filter = np.triu(self.filter, 1)  # upper triangle with zero diagonal
        self.filter = self.filter.reshape((1, size, size))
        super(Triangularize, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (None, size, size)
        pass

    def call(self, x, mask=None):
        return x * self.filter

    def get_output_shape_for(self, input_shape):
        # input_shape: (None, size, size)
        return input_shape


class Bias(Layer):
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (None, dim)
        assert len(input_shape) == 2
        output_dim = input_shape[-1]
        self.b = K.zeros((output_dim,), name='{}_b'.format(self.name))
        self.trainable_weights = [self.b]

    def call(self, x, mask=None):
        return x + self.b

    def get_output_shape_for(self, input_shape):
        # input_shape: (None, dim)
        return input_shape


def get_model(conf, data_spec):
    num_entity_type = data_spec.num_entity_type
    num_entity = data_spec.num_entity

    emb_dim = conf.emb_dim
    batch_size = conf.batch_size
    num_negatives = conf.num_negatives
    loss = conf.loss

    X = Input(shape=(num_entity_type,), dtype='int32')
    sn = Input(shape=(1,), dtype='float32')  # noise score

    Emb_entity = Embedding(num_entity, emb_dim, input_length=num_entity_type)
    entity_emb = Emb_entity(X)  # (None, num_entity_type, emb_dim)
    entity_emb_t = Permute((2, 1))(entity_emb) # (None, emb_dim, num_entity_type)
    h = Merge(mode='dot', dot_axes=[2, 1])([entity_emb, entity_emb_t])  # (None, num_entity_type, num_entity_type)
    #h = merge([entity_emb, entity_emb_t], mode='dot', dot_axes=[1, 2])  # (None, num_entity_type, num_entity_type)
    if conf.no_weight:
        h = Triangularize(num_entity_type)(h)
        h = Flatten()(h)
        y_pred = Dense(1, trainable=False, init='one', activation='linear')(h)
        y_pred = Bias()(y_pred)
    else:
        h = Triangularize(num_entity_type)(h)
        h = Flatten()(h)
        y_pred = Dense(1, init='one', W_constraint='nonneg')(h)
    if not conf.ignore_noise_dist:
        y_pred = Lambda(lambda x: x - sn)(y_pred)

    model = Model(input=[X, sn], output=[y_pred])
    #model.compile(optimizer='adam', loss='kld')
    #model.compile(optimizer=SGD(10), loss=get_ranking_loss(loss, batch_size, num_negatives))
    model.compile(optimizer=Adam(0.01), loss=get_ranking_loss(loss, batch_size, num_negatives))

    return model