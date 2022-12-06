import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from networks.rank_cp import *
from networks.E2E_PextE import *
from networks.E2E_PextC import *

class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.e2e = E2E_PextE(configs)
        # self.e2e = E2E_PextC(configs)
        self.rank_cp = RankCP(configs)
        self.middle_layer = nn.Linear(4*configs.n_hidden, 2*configs.n_hidden)
    
    def forward(self, x):
        # Clause Embedding
        s = self.e2e.get_clause_embedding(x)

        # PextE
        x_pos, pred_pos = self.e2e.get_emotion_prediction(s)
        s_pred_pos = torch.cat([s, pred_pos], 2)
        x_cause, pred_cause = self.e2e.get_cause_prediction(s_pred_pos)

        # PextC
        # x_cause, pred_cause = self.e2e.get_cause_prediction(s)
        # s_pred_cause = torch.cat([s, pred_cause], 2)
        # x_pos, pred_pos = self.e2e.get_emotion_prediction(s_pred_cause)

        cat_x = torch.cat([x_pos, x_cause], -1)
        doc_sents_h = self.middle_layer(cat_x)

        # Pair Extraction
        couples_pred, emo_cau_pos = self.rank_cp.rank(doc_sents_h)

        return couples_pred, emo_cau_pos, pred_pos, pred_cause