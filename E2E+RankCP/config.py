import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
TORCH_SEED = 129
DATA_DIR = '../data_combine_eng/'

SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses_eng.pkl'

class Config(object):
    def __init__(self):
        self.train_file_path = DATA_DIR + 'clause_keywords.csv'
        self.w2v_file = DATA_DIR + 'w2v_200.txt'
        self.embedding_dim = 200
        self.embedding_dim_pos = 50
        self.max_sen_len = 30
        self.max_doc_len = 41
        self.n_hidden = 100
        self.n_class = 2
        self.keep_prob1 = 0.8
        self.keep_prob2 = 1.0
        self.keep_prob3 = 1.0
        self.l2_reg = 0.00010
        self.diminish_factor = 0.400
        self.cause = 1.0
        self.pos = 1.0
        self.pair = 1.0
        self.learning_rate = 0.005

        self.split = 'split10'

        self.bert_cache_path = 'bert-base-cased'
        self.feat_dim = 768

        self.gnn_dims = '192'
        self.att_heads = '4'
        self.K = 12
        self.pos_emb_dim = 50
        self.pairwise_loss = False

        self.epochs = 15
        self.lr = 1e-5
        self.batch_size = 16
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.l2 = 1e-5
        self.l2_bert = 0.01
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8

