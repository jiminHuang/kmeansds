# -*- coding: utf-8 -*-
#
# Author: huang
#
'''
    The implementation of the framework of combining kmeans with distant supervision
'''
import argparse
import logging
import time
import random
import collections
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics.pairwise import euclidean_distances


NEG_RATIO = 0.05 # the ratio of subsample negatives
SUBSAMPLE = True # Subsample the cluster or not


class MentionDatum(object):
    '''
        The Class of Mention in the Datum
    '''
    ENTITY = {} # First entity of the entity pair
    TYPE = {} # Type of first entity
    NE = {} # Type of second entity
    SLOT = {} # Second entity of the entity pair
    RELATION = {} # Belonged Relation in DS
    FEATURE = {}
    FEATURE_APPEARENCE = []

    # Class variable of the counts of values
    entity_number = 0
    type_number = 0
    ne_number = 0
    slot_number = 0
    relation_number = 0
    feature_number = 0

    # Initialization for @property
    _entity_id = None
    _entity_type = None
    _ne_type = None
    _slot_value = None
    _relation = []
    _features = []

    def __init__(self, args):
        self.entity_id = args[0]
        self.entity_type = args[1]
        self.ne_type = args[2]
        self.slot_value = args[3]
        self.relation = args[4]
        self.features = args[5:]
        self.relabel_relation = []

    @property
    def entity_id(self):
        return self._entity_id

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def ne_type(self):
        return self._ne_type

    @property
    def slot_value(self):
        return self._slot_value

    @property
    def relation(self):
        return self._relation

    @property
    def features(self):
        return self._features

    @entity_id.setter
    def entity_id(self, value):
        if value not in MentionDatum.ENTITY:
            MentionDatum.ENTITY[value] = self.entity_number
            MentionDatum.entity_number += 1
        self._entity_id = MentionDatum.ENTITY.get(value)

    @entity_type.setter
    def entity_type(self, value):
        if value not in MentionDatum.TYPE:
            MentionDatum.TYPE[value] = self.type_number
            MentionDatum.type_number += 1
        self._entity_type = MentionDatum.TYPE.get(value)

    @ne_type.setter
    def ne_type(self, value):
        if value not in MentionDatum.NE:
            MentionDatum.NE[value] = self.ne_number
            MentionDatum.ne_number += 1
        self._ne_type = MentionDatum.NE.get(value)

    @slot_value.setter
    def slot_value(self, value):
        if value not in MentionDatum.SLOT:
            MentionDatum.SLOT[value] = self.slot_number
            MentionDatum.slot_number += 1
        self._slot_value = MentionDatum.SLOT.get(value)

    @relation.setter
    def relation(self, value):
        value = value.split('|')
        reform_relation = []
        for rel in value:
            if rel not in MentionDatum.RELATION:
                MentionDatum.RELATION[rel] = self.relation_number
                MentionDatum.relation_number += 1
            reform_relation.append(MentionDatum.RELATION.get(rel))
        self._relation = reform_relation

    @features.setter
    def features(self, value):
        reform_feature = []
        for feature in value:
            if feature not in MentionDatum.FEATURE:
                MentionDatum.FEATURE[feature] = self.feature_number
                MentionDatum.feature_number += 1
                MentionDatum.FEATURE_APPEARENCE.append(0)
            feature_index = MentionDatum.FEATURE.get(feature)
            MentionDatum.FEATURE_APPEARENCE[feature_index] += 1
            reform_feature.append(feature_index)
        self._features = reform_feature

    def __str__(self):
        relation = self.relation if not self.relabel_relation else self.relabel_relation
        mention_str =\
            (
                '{0} {1} {2} {3} {4} {5}'
            ).format(
                MentionDatum.ENTITY.get(self.entity_id),
                MentionDatum.TYPE.get(self.entity_type),
                MentionDatum.NE.get(self.ne_type),
                MentionDatum.SLOT.get(self.slot_value),
                '|'.join([MentionDatum.RELATION.get(rel) for rel in relation]),
                ' '.join([MentionDatum.FEATURE.get(fea) for fea in self.features]),
            )
        return mention_str

    @classmethod
    def shrink_features(cls, threshold=5):
        '''
            Shrink the features whose appearence is less than the setting threshold.
        '''

        shrinked_index = 0
        shrinked_feature = {}
        cls.FEATURE_INDEX = {} # Regenerate index for shrinked feature space

        for fea, index in cls.FEATURE.iteritems():
            if cls.FEATURE_APPEARENCE[index] >= threshold:
                shrinked_feature[fea] = index
                cls.FEATURE_INDEX[index] = shrinked_index
                shrinked_index += 1

        shrinked_feature_number = cls.feature_number - shrinked_index
        cls.feature_number = shrinked_index
        cls.FEATURE_APPEARENCE = None
        logging.info('[OK]...Feature Shrinking')
        logging.info('---# of shrinked Features: {0}'.format(shrinked_feature_number))


    def _feature_vector_generation(self):
        '''
            Generate the feature vector in the shrinked feature space.
        '''
        return dict(
            [
                (str(MentionDatum.FEATURE_INDEX[index]), 1)
                for index in self.features
                if index in MentionDatum.FEATURE_INDEX
            ]
        )

    @classmethod
    def regenerate_feature(cls, mentions):
        '''
            Generate feature vectors for all relation mentions
        '''
        return [mention._feature_vector_generation() for mention in mentions]

    @classmethod
    def transpose_values(cls):
        '''
            Transpose all value dicts for the generation of datum files.
        '''
        cls.ENTITY = dict(
            zip(cls.ENTITY.values(), cls.ENTITY.keys())
        )
        cls.TYPE = dict(zip(cls.TYPE.values(), cls.TYPE.keys()))
        cls.NE = dict(zip(cls.NE.values(), cls.NE.keys()))
        cls.SLOT = dict(zip(cls.SLOT.values(), cls.SLOT.keys()))
        cls.RELATION = dict(
            zip(cls.RELATION.values(), cls.RELATION.keys())
        )
        cls.FEATURE = dict(
            zip(cls.FEATURE.values(), cls.FEATURE.keys())
        )


def _subsample_negatives(mention):
    '''
        Subsample negatives from mention.
        :type mention: MentionDatum
        :rtype boolean
    '''
    nr = MentionDatum.RELATION.get('_NR', None)
    if nr is not None\
        and [nr] == mention.relation\
            and random.uniform(0, 1) > NEG_RATIO:
        return False
    return True
    

def _read_datum_file(file_path):
        '''
            Load the datum from the datum file
            :type file_path: basestring
            :type neg_ratio: double in [0,1]
            :rtype List[MentionDatum]
        '''
        mentions = []

        with open(file_path) as f:
            for line in f:
                mention = MentionDatum(line.split())
                if not _subsample_negatives(mention):
                    continue
                mentions.append(mention)

        logging.debug(
            '---[OK]...Datum File {0} Loaded | {1} Mentions Loaded'.format(
                file_path,
                len(mentions),
            )
        )
        return mentions


def datums_read(directory, number=88):
    '''
        Load datums from NUMBER of datum files in the DIRECTORY
        :type directory: basestring
        :type number: int in [0, # of datum file in the DIRECTORY]
        :rtype List[MentionDatum]
    '''
    def _generate_file_path(index, generate_mode='{0}/kb_part-00{1:0>2d}.datums'):
        '''
            Generate the file path in the directory
        '''
        return generate_mode.format(directory, index)
    start = time.clock()
    loaded_mentions = []

    for datum_number in xrange(number):
        loaded_mentions += _read_datum_file(_generate_file_path(datum_number+1))

    time_cost = time.clock() - start
    logging.info(
        (
            '[OK]...All Datums Loaded\n'
            '---Cost Time: {0} | Average Per File: {1}\n'
            '---# of Loaded Mentions: {2}\n'
            '---# of Loaded Entities: {3}\n'
            '---# of Loaded Entity Types: {4}\n'
            '---# of Loaded NE Types: {5}\n'
            '---# of Loaded Slots: {6}\n'
            '---# of Loaded Relations: {7}\n'
            '---# of Loaded Features: {8}\n'
        ).format(
            time_cost,
            time_cost/number,
            len(loaded_mentions),
            MentionDatum.entity_number,
            MentionDatum.type_number,
            MentionDatum.ne_number,
            MentionDatum.slot_number,
            MentionDatum.relation_number,
            MentionDatum.feature_number,
        )
    )

    return loaded_mentions


def _generate_feature_space(mentions):
    '''
        Generate the features space.
        ---------------------------------
        :type mentions: List[MentionDatum]
        :rtype: numpy.ndarray
    '''
    start = time.clock()
    # Shrink the features
    MentionDatum.shrink_features(threshold=5)
    
    # Regenerate feature vectors
    feature_space = MentionDatum.regenerate_feature(mentions)

    # Generate feature space
    feature_space =\
        FeatureHasher(
            n_features=MentionDatum.feature_number
        ).transform(feature_space)

    time_cost = time.clock() - start
    logging.info('[OK]...Generate Feature Space in {0}s'.format(time_cost))
    
    return feature_space


def _minibatchkmeans(feature_space, cluster_number):
    '''
        Use MinibatchKkmeans to divide the feature_space into cluster_number bags.
        -------------------------------------------------------------------------
        :type feature_space: numpy.ndarray
        :type cluster_number: int
        :rtype: numpy.ndarray[n_mentions,] labels of the mentions 
    '''
    start = time.clock()
    model =\
        MiniBatchKMeans(
            n_clusters=cluster_number,
            n_init=22,
            batch_size=5700
        )

    predicts = model.fit_predict(feature_space)
    
    
    logging.info('[OK]...Kmeans Clustering | Cost {0}s'.format(time.clock()-start))
    return predicts


def _predict_to_cluster(predicts, mentions):
    '''
        Transform predicts to clusters.
        -------------------------------
        :type predicts: numpy.ndarray[n_samples,]
        :type mentions: List[MentionDatum]
        :rtype: List[[int,]] 
    '''
    cluster_number = len(set(predicts))
    clusters = [[] for size in xrange(cluster_number)]
    for index, predict in enumerate(predicts):
        clusters[predict]+=mentions[index].relation
    logging.info('------[OK]...Labels Transform To Clusters')
    return clusters


def _assign_cluster_relation(predicts, mentions):
    '''
        Assign each cluster the most similar relation according to the assumption.
        --------------------------------------------------------------------------
        :type predicts: numpy.ndarray[n_samples,]
        :type mentions: List[MentionDatum]
        :rtype: List[(int, double)]
    '''
    start = time.clock()
    relation_for_clusters = []

    # Predicts -> clusters
    clusters = _predict_to_cluster(predicts, mentions)
    
    for cluster in clusters:
        relation_counter = collections.Counter(cluster)
        logging.info('---Cluster assign: {0}'.format(relation_counter))
        assign_relation = relation_counter.most_common(1)[0]
        relation_for_clusters.append(
            (
                assign_relation[0],
                (assign_relation[1]+0.0)/len(cluster),
            )
        )
    
    time_cost = time.clock() - start
    logging.info('---[OK]...Assign cluster relations cost of {0}'.format(time_cost)) 
    return relation_for_clusters


def _subsample_mention(predicts, clusters, mentions):
    '''
        Subsample mentions in a cluster based on the probability of the relation.
        -------------------------------------------------------------------------
        :type predicts: numpy.ndarray[n_samples,]
        :type clusters: List[(int, double)]
        :type mentions: List[MentionDatum]
        :rtype: None
    '''
    start = time.clock()
    subsample_number = 0
    
    for index, predict in enumerate(predicts):
        relation, probability = clusters[predict]
	if not SUBSAMPLE or random.random() < probability:
	    mentions[index].relabel_relation.append(relation)
	    subsample_number += 1
    
    time_cost = time.clock() - start
    logging.info('---[OK]...Subsample mentions cost of {0}'.format(time_cost)) 
    logging.info('------# of subsamples: {0}'.format(subsample_number))


def kmeans_predict(mentions, cluster_number=100):
    '''
        The framework predicts labels of mentions as following:
        1. Generate the feature space
        2. Kmeans divides the feature space into k clusters
        3. Reassign each cluster a relation based on DS
        4. Subsample mentions in the cluster to be labeled with corresponding relation
        NOTE: Usually k is much higher than the # of known relations.
        ---------------------------------------------------
        :type mentions:List[DatumMention]
        :type cluster_number:int
        :rtype None
    '''
    start = time.clock()
    feature_space = _generate_feature_space(mentions)

    predicts = _minibatchkmeans(feature_space, cluster_number)
    
    relation_for_clusters = _assign_cluster_relation(predicts, mentions)

    _generate_cluster(predicts, relation_for_clusters, mentions)
    
    _subsample_mention(predicts, relation_for_clusters, mentions)

    logging.info('[OK]...Framework | Cost {0}s'.format(time.clock()-start))


def regenerate_datums(mentions, filepath):
    '''
        Regenerate datums with the new relation
        -------------------------------------------------
        :type mentions: List[MentionDatum]
        :type filepath: basestring
        :rtype: None
    '''
    start = time.clock()
    file_number = len(mentions) / 90000 + 1
    negative_number = 0
    nr = MentionDatum.RELATION.get('_NR')
    
    #transpose values
    MentionDatum.transpose_values()

    for index in xrange(file_number):
        with open(filepath + '/{0:0>2d}.datums'.format(index), 'w') as f:
            for mention in mentions[index*90000:(index+1)*90000]:
                if nr in mention.relabel_relation:
                    negative_number += 1
	        f.write(str(mention))
		f.write('\n')
        logging.debug('---[OK]...Generate {0:0>2d}.datums'.format(index))

    spend = time.clock() - start
    logging.info('[OK]...Generate {0} Datums File'.format(file_number))
    logging.info('[OK]...Negative number: {0}'.format(negative_number))
    logging.info('---Cost time: {0} | Average per file: {1}'.format(spend, spend/file_number))


def _generate_cluster(predicts, clusters, mentions):
    '''
        Generate clusters from predicts.
        =======================================
        :type predicts: numpy.ndarray[n_samples,]
        :type clusters: List[(int, double)]
        :type mentions: List[MentionDatum]
        :rtype: None
    '''
    entity_index = dict(
        zip(MentionDatum.ENTITY.values(), MentionDatum.ENTITY.keys())
    )
    slot_index = dict(
        zip(MentionDatum.SLOT.values(), MentionDatum.SLOT.keys())
    )
    relation_index = dict(
        zip(MentionDatum.RELATION.values(), MentionDatum.RELATION.keys())
    )
    cluster_results = [[] for index in xrange(100)]
    for index, predict in enumerate(predicts):
        relation, probability = clusters[predict]
        cluster_results[predict].append(
            (
                entity_index[mentions[index].entity_id],
                slot_index[mentions[index].slot_value],
                relation_index[mentions[index].relation[0]],
                relation_index[relation],
            )
        )
    for index, cluster_result in enumerate(cluster_results):
        with open('result/'+str(index), 'w') as f:
            f.write('\n'.join([str(result) for result in cluster_result]))
    with open('result/index', 'w') as f:
        f.write('\n'.join([str(index) for index in sorted(enumerate(clusters), key=lambda x:x[1][1])]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('METHOD START')

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int)
    parser.add_argument('-r', type=float)
    parser.add_argument('-o', type=str)
    parser.add_argument('-d', type=str)
    parser.add_argument('-s', type=bool)
    parser.add_argument('-f', type=int)
    args = parser.parse_args()

    start = time.clock()
    cluster_number = args.n
    NEG_RATIO = (args.r + 0.0) / 100
    SUBSAMPLE = True if args.s else False

    logging.info('CLUSTER NUMBER:{0}'.format(cluster_number))
    logging.info('NEG_RATIO:{0}'.format(NEG_RATIO))
    logging.info('OUTPUT_DIR:{0}'.format(args.o))
    logging.info('DATA_DIR:{0}'.format(args.d))
    logging.info('SUBSAMPLE:{0}'.format(SUBSAMPLE))

    mentions = datums_read(args.d, number=args.f)
    kmeans_predict(mentions, cluster_number)
    regenerate_datums(
        mentions,
        args.o,
    )
    logging.info('Method End With {0}s'.format(time.clock()-start))
