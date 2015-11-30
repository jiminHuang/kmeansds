# -*- coding: utf-8 -*-
#
# Author: jimin.huang
#
# Created Time: 2015年11月12日 星期四 15时59分38秒
#
'''
    对读出的DATUM进行Kmeans聚类，再根据聚类结果重新标记
'''
import logging
import time
import random
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import FeatureHasher


class MentionDatum(object):
    '''
        mention 在datum的数据结构
    '''
    ENTITY = {}
    TYPE = {}
    NE = {}
    SLOT = {}
    RELATION = {}
    FEATURE = {}
    FEATURE_APPEARENCE = []

    entity_number = 0
    type_number = 0
    ne_number = 0
    slot_number = 0
    relation_number = 0
    feature_number = 0

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
        mention_str =\
            (
                u'实体: {0}\n'
                u'实体类型: {1}\n'
                u'命名实体类型: {2}\n'
                u'填空实体值: {3}\n'
                u'关系类型: {4} \n'
                u'特征集: {5}\n'
            ).format(
                self.entity_id,
                self.entity_type,
                self.ne_type,
                self.slot_value,
                self.relation,
                self.features,
            )

        return mention_str.encode('utf8')

    @classmethod
    def shrink_features(cls, threshold=5):
        '''
            清洗特征，出现次数小于阈值的被洗掉，重新确定索引
        '''
        logging.info(u'开始清洗特征')
        shrinked_index = 0
        shrinked_feature = {}
        cls.FEATURE_INDEX = {}
        for fea, index in cls.FEATURE.iteritems():
            if cls.FEATURE_APPEARENCE[index] >= threshold:
                shrinked_feature[fea] = index
                cls.FEATURE_INDEX[index] = shrinked_index
                shrinked_index += 1
        shrink_feature = cls.feature_number - shrinked_index
        cls.feature_number = shrinked_index
        cls.FEATURE_APPEARENCE = None

        logging.info(
            u'清洗了{0}个特征,还剩{1}个特征'.format(
                shrink_feature,
                cls.feature_number
            )
        )

    def _feature_vector_generation(self):
        '''
            生成numpy array形式特征向量
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
            清洗后重新生成特征向量
        '''
        logging.info(u'重新生成特征')
        return [mention._feature_vector_generation() for mention in mentions]


def datums_read(file_path, number=88, neg_ratio=0.05):
    start = time.clock()
    mentions = []
    loaded_mention_number = 0
    logging.info(u'开始读取datums文件')
    for datum_number in xrange(number):
        with open(
            '{0}/kb_part-00{1:0>2d}.datums'.format(
                file_path,
                datum_number+1,
            )
        ) as f:
            logging.debug(u'开始处理文件{0:0>2d}'.format(datum_number+1))
            present_loaded = 0
            for line in f:
                mention = MentionDatum(line.split())
                nr = MentionDatum.RELATION.get('_NR', None)
                if nr is not None\
                    and [nr] == mention.relation\
                        and random.uniform(0, 1) > neg_ratio:
                    continue
                mentions.append(mention)
                logging.debug(mention)
                present_loaded += 1
            logging.debug('载入了{0}个mention'.format(present_loaded))
            loaded_mention_number += present_loaded
    spend = time.clock() - start
    logging.info(u'载入耗时{0} | 平均每个文件载入时间为{1}'.format(spend, spend/number))
    logging.info(u'载入mention数量{0}'.format(loaded_mention_number))
    logging.info(u'载入实体数量{0}'.format(MentionDatum.entity_number))
    logging.info(u'载入实体类型数量{0}'.format(MentionDatum.type_number))
    logging.info(u'载入NE类型数量{0}'.format(MentionDatum.ne_number))
    logging.info(u'载入SLOT数量{0}'.format(MentionDatum.slot_number))
    logging.info(u'载入关系数量{0}'.format(MentionDatum.relation_number))
    logging.info(
        u'载入特征大小{0}'.format(
            MentionDatum.feature_number
        )
    )
    return mentions


def predict_to_cluster(predicts, cluster_number):
    '''
        预测标记转为类
    '''
    logging.info(u'开始根据标记构造类')
    clusters = [[] for size in xrange(cluster_number)]
    for index, predict in enumerate(predicts):
        clusters[predict].append(index)
    return clusters


def allocate_cluster(clusters, mentions):
    '''
        根据DS假设预先标记分配类
    '''
    logging.info(u'根据DS假设划分类到各个关系重新标记')
    cluster_allocation =\
        [
            [0 for rel_size in xrange(MentionDatum.relation_number)]
            for size in xrange(len(clusters))
        ]
    for cluster_index, cluster in enumerate(clusters):
        for index in cluster:
            for relation in mentions[index].relation:
                cluster_allocation[cluster_index][relation] += 1

    cluster_allocation =\
        [
            [(relation+0.0)/sum(cluster) for relation in cluster]
            for cluster in cluster_allocation
        ]
    cluster_relation = [list(cluster) for cluster in zip(*cluster_allocation)]

    cluster_score = [[] for size in xrange(MentionDatum.relation_number)]

    for relation_index, relation in enumerate(cluster_relation):
        for index, cluster in enumerate(relation):
            cluster_score[relation_index].append((cluster, index))

    cluster_score =\
        [
            sorted(relation, reverse=True)[:5] for relation in cluster_score
        ]

    logging.info(u'计算各个类相对于关系的信心')
    logging.info(u'\n'.join((str(score) for score in cluster_score)))

    logging.info(u'划分结束')

    return [
        (
            cluster.index(max(cluster)),
            (max(cluster)+0.0)/sum(cluster)
        )
        for cluster in cluster_allocation
    ]


def kmeans_predict(mentions, cluster_number=100):
    '''
        生成特征空间，使用Kmeans聚类，再利用聚类结果重新标记
    '''
    MentionDatum.shrink_features(threshold=5)
    feature_space = MentionDatum.regenerate_feature(mentions)
    logging.info(u'开始生成特征空间')
    start = time.clock()
    feature_space =\
        FeatureHasher(
            n_features=MentionDatum.feature_number
        ).transform(feature_space)
    logging.info(u'生成特征空间结束，耗时{0}'.format(time.clock()-start))
    logging.info('开始Kmeans聚类')
    start = time.clock()
    predicts =\
        MiniBatchKMeans(
            n_clusters=cluster_number,
            n_init=1,
            batch_size=3000
        ).fit_predict(feature_space)
    logging.info(u'聚类结束，耗时{0}'.format(time.clock()-start))
    clusters = predict_to_cluster(predicts, cluster_number)
    start = time.clock()
    allocate = allocate_cluster(clusters, mentions)
    logging.info(u'划分结束，耗时{0}'.format(time.clock()-start))
    return (clusters, allocate)


def relabel(allocate, mentions, clusters):
    '''
        根据划分和分类结果重新标记
    '''
    logging.info(u'开始重新标记')
    relabel_number = 0
    for cluster, alloc in enumerate(allocate):
        relation = alloc[0]
        score = alloc[1]
        for index in clusters[cluster]:
            if random.uniform(0, 1) <= score:
                mentions[index].relabel_relation.append(relation)
                relabel_number += 1
    logging.info(
        u'重新标记结束 耗时{0} | 重新标记了{1}'.format(
            time.clock()-start,
            relabel_number
        )
    )


def regenerate_datums(mentions, filepath, cluster_number):
    '''
        根据重新生成的标签重新生成datums
    '''
    entity = dict(
        zip(MentionDatum.ENTITY.values(), MentionDatum.ENTITY.keys())
    )
    en_type = dict(zip(MentionDatum.TYPE.values(), MentionDatum.TYPE.keys()))
    ne = dict(zip(MentionDatum.NE.values(), MentionDatum.NE.keys()))
    slot = dict(zip(MentionDatum.SLOT.values(), MentionDatum.SLOT.keys()))
    relation = dict(
        zip(MentionDatum.RELATION.values(), MentionDatum.RELATION.keys())
    )
    feature = dict(
        zip(MentionDatum.FEATURE.values(), MentionDatum.FEATURE.keys())
    )

    file_number = len(mentions) / 90000 + 1
    start = time.clock()
    logging.info(u'开始生成{0}个datums文件'.format(file_number))
    for index in xrange(file_number):
        logging.info(u'生成{0:0>2d}.datums文件'.format(index))
        with open(filepath + '/{0:0>2d}_{1}.datums'.format(index, cluster_number), 'w') as f:
            for mention in mentions[index*90000:(index+1)*90000]:
                mention_relation = mention.relabel_relation\
                    if mention.relabel_relation else mention.relation
                f.write(
                    ' '.join(
                        (
                            entity.get(mention.entity_id),
                            en_type.get(mention.entity_type),
                            ne.get(mention.ne_type),
                            slot.get(mention.slot_value),
                            '|'.join(
                                (relation.get(rel) for rel in mention_relation)
                            ),
                            ' '.join(
                                (feature.get(fea) for fea in mention.features)
                            ),
                        )
                    )
                )
                f.write('\n')
    spend = time.clock() - start
    logging.debug(
        u'生成结束 耗时{0}|平均每个文件{1}'.format(
            spend,
            (spend+0.0)/file_number
        )
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info(u'开始处理')
    start = time.clock()
    cluster_number = 110
    mentions = datums_read('/home/jimin/mimlre-2012-11-27/corpora/kbp/train')
    clusters, allocate = kmeans_predict(mentions, cluster_number)
    relabel(allocate, mentions, clusters)
    regenerate_datums(
        mentions,
        '/home/jimin/mimlre-2012-11-27/corpora/kbp/train/genTest',
        cluster_number,
    )
    logging.info(u'处理结束 共耗时{0}'.format(time.clock()-start))
