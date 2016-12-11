import numpy as np
import pandas as pd
import random
import os
import sys
from collections import Counter


#############################
# Data preparations
#############################


def data_transformation(table):
    # map entity into continuous idx
    # input:
    #  table - pandas data frame, all columns are of interests
    # return: 
    #  table_transformed
    #  id2type_and_name
    #  type_and_name_2id
    #  type2range

    id2type_and_name, type_and_name2id = {}, {}
    type2range = {}
    max_id = -1
    table_transformed = table.copy()
    for col in table:
        name2id_local = {}
        entity_type = str(col)
        for entity_name_ in set(table[entity_type]):
            entity_name = str(entity_name_)
            type_and_name = (entity_type, entity_name)
            if not type_and_name in type_and_name2id:
                max_id += 1
                type_and_name2id[type_and_name] = max_id
                name2id_local[entity_name] = max_id
                id2type_and_name[max_id] = type_and_name
                if entity_type not in type2range:
                    type2range[entity_type] = [max_id, max_id]
                else:
                    min_, max_ = type2range[entity_type]
                    if max_id > max_: type2range[entity_type][1] = max_id
                    elif max_id < min_: type2range[entity_type][0] = max_id
        table_transformed[col] = table_transformed[col].apply(lambda x: name2id_local[str(x)])
        
    return table_transformed, id2type_and_name, type_and_name2id, type2range


def transform_with_keys(table, columns, type_and_name2id):
    # transform raw csv table into table of indexs using existing keys
    # type_and_name2id is produced by data_transformation()
    # note: if the key is missing, it will be indexed as np.nan
    type_and_name2id = dict([('%s.%s' % (k[0], k[1]), v) for k, v in type_and_name2id.iteritems()])
    table_new = table.copy()
    for col in columns:
        table_new[col] = table[col].apply(lambda x: type_and_name2id['%s.%s' % (col, x)] if '%s.%s' % (col, x) in type_and_name2id else np.nan)
    return table_new


def generate_mixed_events(events_batch, neg_entity_typeid, num_negatives, type2sampler, typeid2type, noise_prob, mixed_layout='separate'):
    # sample negative events that replace positive events_batch in neg_entity_typeid, and compute the noise prob score
    # concatenate to obtain mixed events:
    #   1. mixed_layout=='mixed': (positive, k negatives, ...)
    #   2. mixed_layout=='separate': (positive, ..., k negatives, ...)
    # return events_batch_mixed, events_noise_prob, events_label
    batch_size = events_batch.shape[0]
    total_num_negatives = batch_size * num_negatives
    mixed_event_size = batch_size * (num_negatives + 1)
    
    assert mixed_layout == 'separate' or mixed_layout == 'mixed', mixed_layout
    
    neg_entities = type2sampler[typeid2type[neg_entity_typeid]](total_num_negatives)
    if mixed_layout == 'mixed':
        events_batch_mixed = np.repeat(events_batch, num_negatives + 1, axis=0)
        for i in range(num_negatives):
            events_batch_mixed[i+1::num_negatives + 1, neg_entity_typeid] = neg_entities[i::num_negatives]
        events_noise_prob = noise_prob[events_batch_mixed[:, neg_entity_typeid]]

        events_label = np.zeros(mixed_event_size)
        events_label[::num_negatives + 1] = 1
        events_label = (events_label - 0.5) * 2  # turn into +1/-1   
    elif mixed_layout == 'separate':
        events_batch_negative = np.repeat(events_batch, num_negatives, axis=0)
        events_batch_negative[:, neg_entity_typeid] = neg_entities
        events_batch_mixed = np.vstack((events_batch, events_batch_negative))
        events_noise_prob = noise_prob[events_batch_mixed[:, neg_entity_typeid]]

        '''
        # positive samples using average over entity types - similar
        events_noise_prob[:batch_size] = 0
        for c in range(events_batch.shape[1]):
            events_noise_prob[:batch_size] += noise_prob[events_batch[:, c]]
        events_noise_prob[:batch_size] /= events_batch.shape[1]

        # all samples using average over entity types - bad
        events_noise_prob[:] = 0
        for c in range(events_batch.shape[1]):
            events_noise_prob += noise_prob[events_batch_mixed[:, c]]
        events_noise_prob /= events_batch_mixed.shape[1]
        '''
        
        events_label = np.zeros(mixed_event_size)
        events_label[:batch_size] = 1
        events_label = (events_label - 0.5) * 2  # turn into +1/-1   

    return events_batch_mixed, events_noise_prob, events_label


############################
# Sampler preparations
############################


from sampler import MultinomialSampler

def get_entity_sampler(data_column, neg_dist='unigram', batch_mode=True, neg_sampling_power=0.75, rand_seed=0):
    '''
    Return a sampling function
    '''
    assert neg_dist == 'uniform' or neg_dist == 'unigram', [neg_dist]

    min_element = np.min(data_column)
    max_element = np.max(data_column)
    size = max_element - min_element + 1
    data_column_new = data_column - min_element
    
    dist = np.array([0.] * size)
    if neg_dist == 'uniform':
        dist[data_column_new] = 1
    else:
        for d in data_column_new: dist[d] += 1        
    remap = np.array(range(size), dtype=np.int32) + min_element
    
    s = MultinomialSampler(dist, size, neg_sampling_power, rand_seed, remap)
    dist = (dist / dist.sum()).astype(np.float32)
    if batch_mode:
        return s.sample_batch, (dist, min_element, max_element)
    else:
        return s.sample, (dist, min_element, max_element)


def get_entity_samplers_and_noise_prob(table_transformed, noise_prob_cal='logkPn@10', neg_dist='unigram', batch_mode=True, neg_sampling_power=0.75, rand_seed=0):
    # return sampling functions indexed by type, also the noise_prob can provide noise score for sampled (raw) idx
    # table_transformed should satisfy:
    #  1. column is entity type
    #  2. all entities are mapped into the same id space
    #  3. id continuous within type and increasing according table columns
    type2sampler = {}
    dist_and_boundries = []
    for col in table_transformed:
        sample, (dist, min_element, max_element) = get_entity_sampler(table_transformed[col], neg_dist, batch_mode, neg_sampling_power, rand_seed)
        type2sampler[col] = sample
        dist_and_boundries.append([dist, min_element, max_element])

    last_max_element = -1
    dists_to_stack = []
    for dist, min_element, max_element in dist_and_boundries:
        gap = min_element - (last_max_element + 1)
        assert gap >= 0, 'gap (%d) should be positive, make sure id continuous within type and increasing according table columns' % gap
        if gap == 0:
            dists_to_stack.append(dist)
        else:
            print '[Warning] Adding zero vector to gap of length %d' % gap
            dist_gap = np.zeros(gap)
            dists_to_stack.append(dist_gap)
            dists_to_stack.append(dist)
        last_max_element = max_element
    noise_prob = np.hstack(dists_to_stack)
    if noise_prob_cal.startswith('logkPn'):
        k = int(noise_prob_cal.split('@')[1])  # doesn't matter what this number is
        noise_prob = np.log(k * noise_prob)
    else:
        assert '[ERROR] noise_prob_cal %s is unknown.' % noise_prob_cal
    return type2sampler, noise_prob


def get_entity_type_sampler_and_mappings(table_transformed, neg_dist='unigram', neg_sampling_power=0.75, rand_seed=0, batch_mode=True):
    # sample entity types
    assert neg_dist == 'uniform' or neg_dist == 'unigram', [neg_dist]
    
    typeid2type, type2typeid = {}, {}
    max_typeid = -1
    type_cad_dist = []
    for col in table_transformed.columns:
        max_typeid += 1
        typeid2type[max_typeid] = col
        type2typeid[col] = max_typeid
        #cardinality = type2range[col][1] - type2range[col][0] + 1
        cardinality = np.max(table_transformed[col]) - np.min(table_transformed[col]) + 1
        type_cad_dist.append(cardinality)
    type_cad_dist = np.array(type_cad_dist, dtype='float') / np.sum(type_cad_dist)
    if neg_dist == 'uniform':
        type_cad_dist[type_cad_dist > 0] = 1
        
    s = MultinomialSampler(type_cad_dist, type_cad_dist.size, neg_sampling_power, rand_seed)
    if batch_mode:
        sample = s.sample_batch
    else:
        sample = s.sample
    return type2typeid, typeid2type, sample, type_cad_dist



