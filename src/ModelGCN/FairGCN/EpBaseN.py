import os

import torch

from ModelGCN import GCNBaseN
from consts import MODELS_PATH


class EpBaseN(GCNBaseN):
    """
    Embedding projection GCN Base class

    Method from "Monet: Debiasing graph embeddings via the metadata-orthogonal training unit."

    Abstract class that adds Edge Projection of resulting embeddings.
    Since nothing changes during the training itself, will try to load a trained base network if possible.
    """
    unfair_run_name = None
    fair_prefix = "Ep"

    def set_run_config(self, run_name, lr, weight_decay):
        GCNBaseN.set_run_config(self, run_name, lr, weight_decay)
        if run_name is None:
            run_name = ''
        if not self.validation:
            run_name += 'Noval'
        if self.inductive:
            run_name += 'Ind'
        self.unfair_run_name = '{}/{}_{}_0_0'.format(self.get_unfair_name(), self.data.split_name, run_name)
        self.model_path = MODELS_PATH + self.unfair_run_name

    def get_unfair_name(self):
        return self.model.name

    def get_embeddings(self, split='all'):
        """
        The actual Embedding Projection happens here, we're calculating a bias vector
         and then project all our vectors (embeddings) on a vector perpendicular to the bias vector

        :param split: can be all, train, val or test
        :return:
        """
        data = self.get_data_for_split(split=split)
        embeds = self.model.get_embedding(data, data.ndata['feat'])
        sens_mask = self.data.ndata['sens']
        embeds_0 = embeds[sens_mask == 0]
        embeds_1 = embeds[sens_mask == 1]
        sum_0 = torch.sum(embeds_0, 0)
        sum_1 = torch.sum(embeds_1, 0)
        avg_0 = sum_0 / torch.norm(sum_0)
        avg_1 = sum_1 / torch.norm(sum_1)
        bias = (avg_0 - avg_1) / torch.norm(avg_0 - avg_1)
        fair_embeds = embeds - ((embeds @ bias).unsqueeze(1) * bias)
        return fair_embeds

    def get_predictions(self, split='all'):
        embeds = self.get_embeddings(split)
        return self.model.classify_emb(embeds)
