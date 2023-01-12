import torch
import torch.nn as nn
import numpy as np
from model.KGAT import KGAT


class News_embedding(nn.Module):

    def __init__(self, config, doc_feature_dict, entity_embedding, relation_embedding, adj_entity, adj_relation,
                 entity_num, position_num, type_num, entity_category_num, entity_sec_category_num, device):
        super(News_embedding, self).__init__()
        self.config = config
        self.doc_feature_dict = doc_feature_dict
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.device = device
        self.kgat = KGAT(config, doc_feature_dict, entity_embedding, relation_embedding, adj_entity, adj_relation,
                         device)

        self.entity_num = entity_num
        self.position_num = position_num
        self.type_num = type_num
        self.entity_category_num = entity_category_num
        self.entity_sec_category_num = entity_sec_category_num

        self.final_embedding1 = nn.Linear(
            self.config['model']['document_embedding_dim'] + self.config['model']['embedding_dim'],
            self.config['model']['layer_dim'])
        self.final_embedding2 = nn.Linear(self.config['model']['layer_dim'],
                                          self.config['model']['embedding_dim'])
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.title_embeddings = nn.Embedding(1000, self.config['model']['entity_embedding_dim'])
        self.type_embeddings = nn.Embedding(type_num, self.config['model']['entity_embedding_dim'])
        self.entity_num_embeddings = nn.Embedding(entity_num, self.config['model']['entity_embedding_dim'])
        self.entity_category_embeddings = nn.Embedding(entity_category_num, self.config['model']['entity_embedding_dim'])
        self.entity_second_category_embeddings = nn.Embedding(entity_sec_category_num, self.config['model']['entity_embedding_dim'])

        # Use xavier initialization method to initialize embeddings of entities and relations
        title_weight = torch.FloatTensor(1000, self.config['model']['entity_embedding_dim'])
        type_weight = torch.FloatTensor(self.type_num, self.config['model']['entity_embedding_dim'])
        entity_num_weight = torch.FloatTensor(entity_num, self.config['model']['entity_embedding_dim'])

        nn.init.xavier_normal_(title_weight, gain=0.01)
        nn.init.xavier_normal_(type_weight, gain=0.01)
        nn.init.xavier_normal_(entity_num_weight, gain=0.01)

        self.title_embeddings.weight = nn.Parameter(title_weight)
        self.type_embeddings.weight = nn.Parameter(type_weight)
        self.entity_num_embeddings.weight = nn.Parameter(entity_num_weight)

        self.attention_embedding_layer1 = nn.Linear(
            self.config['model']['document_embedding_dim'] + self.config['model']['entity_embedding_dim'],
            self.config['model']['layer_dim'])
        self.attention_embedding_layer2 = nn.Linear(self.config['model']['layer_dim'], 1)
        self.softmax = nn.Softmax(dim=-2)

        # Multi-Head attention initialization
        self.mh = nn.MultiheadAttention(embed_dim=self.config['model']['document_embedding_dim'],
                                        num_heads=self.config['model']['mh_number_of_heads']
                                        , kdim=self.config['model']['entity_embedding_dim'],
                                        vdim=self.config['model']['entity_embedding_dim']
                                        , batch_first=True)

        # Dynamically choose the order of the news features inside the tuple
        # This is based on how it's formed the MIND dataset on util.py
        self.news_features_order = {'entity': 0, 'frequency': 1, 'position': 2, 'type': 3}
        if 'use_entity_category' in self.config['data'] and self.config['data']['use_entity_category']:
            self.news_features_order['entity_category'] = max(list(self.news_features_order.values())) + 1
        if 'use_second_entity_category' in self.config['data'] and self.config['data']['use_second_entity_category']:
            self.news_features_order['entity_second_category'] = max(list(self.news_features_order.values())) + 1
        self.news_features_order['context_vector'] = max(list(self.news_features_order.values())) + 1

    def multihead_attention_layer(self, entity_embeddings, context_vecs):
        # context_vecs is the Q, entity_embeddings is K and V
        key = torch.sum(entity_embeddings, dim=-2)
        value = torch.sum(entity_embeddings, dim=-2)
        attn_output, attn_output_weights = self.mh(context_vecs, key, value)
        return attn_output, attn_output_weights

    def attention_layer(self, entity_embeddings, context_vecs):
        if len(entity_embeddings.shape) == 4:
            context_vecs = torch.unsqueeze(context_vecs, -2)
            context_vecs = context_vecs.expand(context_vecs.shape[0], context_vecs.shape[1], entity_embeddings.shape[2],
                                               context_vecs.shape[3])
        else:
            context_vecs = torch.unsqueeze(context_vecs, -2)
            context_vecs = context_vecs.expand(context_vecs.shape[0], entity_embeddings.shape[1], context_vecs.shape[2])

        att_value1 = self.relu(self.attention_embedding_layer1(torch.cat([entity_embeddings, context_vecs], dim=-1)))
        att_value = self.relu(self.attention_embedding_layer2(att_value1))
        soft_att_value = self.softmax(att_value)
        weighted_entity_embedding = soft_att_value * entity_embeddings
        weighted_entity_embedding_sum = torch.sum(weighted_entity_embedding, dim=-2)
        topk_weights = torch.topk(soft_att_value, 20, dim=-2)
        if len(soft_att_value.shape) == 3:
            topk_index = topk_weights[1].reshape(topk_weights[1].shape[0],
                                                 topk_weights[1].shape[1] * topk_weights[1].shape[2])
        else:
            topk_index = topk_weights[1].reshape(topk_weights[1].shape[0],
                                                 topk_weights[1].shape[1] * topk_weights[1].shape[2] *
                                                 topk_weights[1].shape[3])
        return weighted_entity_embedding_sum, soft_att_value

    def get_news_features(self, news_id, feature_index):
        features = []
        for news in news_id:
            if type(news) == str:
                features.append(self.doc_feature_dict[news][feature_index])
            else:
                features.append([])
                for news_i in news:
                    features[-1].append(self.doc_feature_dict[news_i][feature_index])
        return features

    def get_news_features_as_embedding(self, features, embedding):
        features_embedding = embedding(torch.tensor(features).to(self.device))
        return features_embedding

    def forward(self, news_id):
        entities = self.get_news_features(news_id, self.news_features_order['entity'])
        entity_nums = self.get_news_features(news_id, self.news_features_order['frequency'])
        istitle = self.get_news_features(news_id, self.news_features_order['position'])
        types = self.get_news_features(news_id, self.news_features_order['type'])
        context_vecs = self.get_news_features(news_id, self.news_features_order['context_vector'])

        entity_num_embedding = self.get_news_features_as_embedding(entity_nums, self.entity_num_embeddings)
        istitle_embedding = self.get_news_features_as_embedding(istitle, self.title_embeddings)
        type_embedding = self.get_news_features_as_embedding(types, self.type_embeddings)
        kgat_entity_embeddings = self.kgat(entities)  # batch(news num) * entity num
        news_entity_embedding = kgat_entity_embeddings + entity_num_embedding + istitle_embedding + type_embedding

        # Adding entity category and second category
        if 'use_entity_category' in self.config['data'] and self.config['data']['use_entity_category']:
            entity_categories = self.get_news_features(news_id, self.news_features_order['entity_category'])
            entity_categories_embedding = self.get_news_features_as_embedding(entity_categories,
                                                                              self.entity_category_embeddings)
            news_entity_embedding = news_entity_embedding + entity_categories_embedding
        if 'use_second_entity_category' in self.config['data'] and self.config['data']['use_second_entity_category']:
            second_entity_categories = self.get_news_features(news_id,
                                                              self.news_features_order['entity_second_category'])
            second_entity_categories_embedding = self.get_news_features_as_embedding(second_entity_categories,
                                                                                     self.entity_second_category_embeddings)
            news_entity_embedding = news_entity_embedding + second_entity_categories_embedding

        # Choose type of attention
        if self.config['model']['use_mh_attention']:
            attention_context, topk_index = self.multihead_attention_layer(news_entity_embedding,
                                                                           torch.FloatTensor(np.array(context_vecs)).to(
                                                                               self.device))
            concat_embedding = torch.cat(
                [attention_context, torch.sum(news_entity_embedding, dim=-2)],
                len(attention_context.shape) - 1)
        else:
            aggregate_embedding, topk_index = self.attention_layer(news_entity_embedding,
                                                                   torch.FloatTensor(np.array(context_vecs)).to(
                                                                       self.device))
            concat_embedding = torch.cat(
                [aggregate_embedding, torch.FloatTensor(np.array(context_vecs)).to(self.device)],
                len(aggregate_embedding.shape) - 1)

        news_embeddings = self.tanh(self.final_embedding2(self.relu(self.final_embedding1(concat_embedding))))

        return news_embeddings, topk_index
