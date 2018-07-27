

class Config:

    def __init__(self):
        self.data_dir = ''

        self.task_id = 1

        self.loss_mode = 0

        self.target_file = 'trn'

        self.device = '/gpu:0'

        self.batch_size = 1
        self.epoch_num = 15
        self.save_step = 50

        self.restore = False

        # self.batch_size = 1
        # self.epoch_num = 10
        # self.save_step = 50

        # self.batch_size = 20
        # self.epoch_num = 10
        # self.save_step = 5

        # above are important parameters

        # default vocab_size
        self.vocab_size = -1

        # default batch_num
        self.batch_num = -1

        # default seq_length
        self.seq_length = -1

        self.is_shuffle = False

        self.da_types = None

        self.num_layers = 1

        # !!! embedding_size = embedding_size_0 + feature_size
        self.embedding_size = 0

        self.embedding_size_0 = 128

        self.feature_size = 128

        self.vocab_file = 'save/vocab.pkl'

        self.word2vec_path = 'word_embeddings'

        self.attnlog_path = 'attn_log/'

        # default slot size
        self.slot_size = -1

        self.pattern_dir = 'data/pattern.txt'
        #
        self.target_vecs = []

        self.grad_clip = 5.0

        self.learning_rate = 0.002

        self.decay_rate = 0.99

        self.save_path = 'save/'

        # memdm_cell variables sizes
        self.rnn_size = 128

        self.v_memory_size = 128

        self.k_memory_size = 128

        self.e_memory_size = 128

        self.e_memory_length = 8

        self.log_path = 'save/log/'

        self.summary_path = 'summary/'

        self.test_path = 'test/'

        self.cofficient = 1.0

    def create_properties(self):
        """
        create some properties
        :return:
        """
        da_types = self.da_types
        slot_value = self.slot_value

        target_vecs = [len(da_types)]
        for item in slot_value.items():
            target_vecs.append(len(item[1]))

        self.target_vecs = target_vecs
        self.k_memory_size = self.slot_size
        self.split_sizes = [len(self.da_types)]
        for slot in self.slots:
            value = self.slot_value[slot]
            self.split_sizes.append(len(value))
        sm = 0
        self.numpy_split_sizes = []
        for item in self.split_sizes:
            sm += item
            self.numpy_split_sizes.append(sm)
        print('\nconfig.split_sizes : {}'.format(self.split_sizes))
        # all vector dimensions are controlled by embedding_size(_0)
        self.embedding_size = self.embedding_size_0 + self.feature_size
        self.e_memory_size = self.embedding_size_0
        self.k_memory_size = self.embedding_size_0
        self.v_memory_size = self.embedding_size
        self.rnn_size = self.embedding_size_0
