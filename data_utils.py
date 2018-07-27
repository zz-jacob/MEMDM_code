"""
process the data of dialog bAbI data
convert yt from string into dialog act representation
give BoV of xt and yt
"""
import numpy as np
import os
import collections
from six.moves import cPickle
from random import shuffle
from config import Config


class DataLoader:
    """
    load data from dialog bAbI tasks
    """
    def __init__(self, args):
        self.args = args

        print('\nstart reading embeddings ...')
        self.embedding_words, self.word_dict, self.embeddings = self.word2vec()
        args.embeddings = self.embeddings
        print('... reading finished!')

        self.data_dir = args.data_dir

        print('\ngetting vocab ...')
        self.vocab, self.words = self.get_vocab()
        args.word_size = len(self.words)

        # test the consistency of self.embedding_words and self.words
        assert len(self.words) == len(self.embedding_words)
        missing_words = list()
        for word in self.words:
            if word not in self.embedding_words:
                missing_words.append(word)
        if len(missing_words) > 0:
            print('!! missing {} words in self.words'.format(len(missing_words)))
            with open('missing_words.txt', 'w+') as f:
                for w in missing_words:
                    f.write(w + '\n')
                f.close()
        with open('word.txt', 'w+') as f:
            for w in self.words:
                f.write(w + '\n')
            f.close()
            print('\nwords write to ./words.txt')

        self.vocab_size = len(self.vocab)
        print('vocab_size: {}'.format(self.vocab_size))

        print('\ngetting slot values ...')
        self.slots, self.slot_value, self.value_slot = self.get_slot_value()
        self.slot_size = len(self.slots)

        # slot_embedding size: shape[slot_size, embedding_size]
        self.slot_embedding = list()
        for s in self.slots:
            ebd = [0.0 for _ in range(args.embedding_size_0)]
            values = self.slot_value[s]
            for v in values:
                vebd = self.embeddings[self.word_dict[v]]
                ebd = [ebd[i] + vebd[i] for i in range(len(vebd))]
            ebd = [item / len(values) for item in ebd]
            self.slot_embedding.append(ebd)
        args.slot_embedding = np.array(self.slot_embedding)

        # get da_types and patterns
        self.pattern, self.da_types = self.get_pattern()
        print('\npattern size: {}'.format(len(self.pattern)))
        print('da types: {}'.format(self.da_types))

        # value dicts
        self.value_index = {}
        self.index_value = {}
        for item in self.slot_value.items():
            dic, dic_ = dict(zip(item[1], range(len(item[1])))), dict(zip(range(len(item[1])), item[1]))
            self.value_index[item[0]] = dic
            self.index_value[item[0]] = dic_

        print('\ndomain slots {}\n'.format(self.slots))
        for item in self.slot_value.items():
            print('slot [{}] has [{}] values, which are: {}'.format(item[0], len(item[1]), item[1]))

        # dat dict
        self.dat_index = dict(zip(self.da_types, range(len(self.da_types))))
        self.index_dat = dict(zip(range(len(self.da_types)), self.da_types))

        a, b = self.get_da('api_call british madrid six expensive')
        print('\nda_type_&_slots: {}; \nsentence: [{}]; pattern: [{}]'.format(
            a, 'api_call british madrid six expensive', b))
        a, b, msk = self.get_da_bov('api_call british madrid six expensive')
        print('\nda_type :{}'.format(a))
        for i, item in enumerate(b):
            if i < 100:
                print('slot[{}]: {}'.format(self.slots[i], item))
        print('mask of b: {}\n'.format(msk))

        a, b = self.get_da_by_bov(a, b)
        print('da_type from bov: {}; \nslot_values: {}'.format(a, b))

        # midify config.properties 2
        args.slot_value = self.slot_value
        args.da_types = self.da_types

        # generate special feature for slot value
        self.value_feature = []
        self.generate_value_feature()

        print('\ngetting data ...')
        self._data_x, self._data_y, self._data_target, self._data_mask, max_seq_length, \
            self.seq_length_list, max_sentence_length, self.attention, self.attention_mask, \
            self.raw_data_x, self.raw_data_y, self.x_bov, self.y_bov = self.create_data_0()
        print('\ntotal data amount: [{}]'.format(len(self._data_x)))
        print('\nmax of self.seq_length_list: {}'.format(max(self.seq_length_list)))
        # print('max seq length: {}'.format(max_seq_length))
        assert max_seq_length == max(self.seq_length_list)

        self.target_vec_size = len(self._data_target[0][0])

        # data_x is current batch_size * batch_num data, _data_x is raw data
        # self.data_x, self.data_y = self._data_x, self._data_y

        assert len(self._data_x) == len(self._data_y) == len(self._data_target) \
            == len(self._data_mask) == len(self.seq_length_list)

        # self.seq_length = 10 * (max_seq_length // 10)
        # if self.seq_length < max_seq_length:
        #     self.seq_length += 10
        self.seq_length = max_seq_length
        self.sentence_length = max_sentence_length

        # modify config.properties
        args.seq_length = self.seq_length
        args.slots = self.slots
        args.vocab_size = self.vocab_size
        args.slot_size = len(self.slots)
        args.target_vec_size = self.target_vec_size
        args.sentence_length = self.sentence_length

        # convert data from multi seq_length to self.seq_length and np.array type
        print('\nconverting data ...')
        self.x_sentence_length, self.y_sentence_length = list(), list()
        self.convert_data()
        # print('\nself.seq_length_list.shape = {}'.format(self.seq_length_list.shape))

        self.is_shuffle = args.is_shuffle

        print('\ncreating batches ...')
        self.x_batches, self.y_batches, self.target_batches, self.mask_batches = [], [], [], []
        self.x_bov_batches, self.y_bov_batches = [], []
        self.seq_length_batches = []
        self.x_sentence_length_batches = []
        self.y_sentence_length_batches = []
        self.attention_batches = []
        self.attention_mask_batches = []
        self.raw_data_x_batches = []
        self.raw_data_y_batches = []
        self.create_batches()
        self.batch_pointer = 0
        print('\nbatches list size: {} * [{}, {}, {}, {}, {}, {}, {}]'.format(
                len(self.x_batches), self.x_batches[0].shape,
                self.y_batches[0].shape, self.target_batches[0].shape,
                self.mask_batches[0].shape, self.seq_length_batches[0].shape,
                self.x_sentence_length_batches[0].shape, self.y_sentence_length_batches[0].shape))

    def word2vec(self):
        """
        get [words, word dict, embedding_matrix] from word2vec_path by embedding_size
        Args:
        Return:
            words: list, [word]
            word_dict: dict, {word: index}
            word_embeddings: np.array() word embeddings [len(words), embedding_size_0]
        """

        files = os.listdir(self.args.word2vec_path)
        files = [self.args.word2vec_path + '/' + f
                 for f in files if str(self.args.embedding_size_0) in f and 'txt' in f]
        assert len(files) == 1
        embedding_file = files[0]

        words = list()
        word_dict = dict()
        word_embedding = list()

        with open(embedding_file) as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if index % 1000 == 0 and index > 0:
                    print('\t\treading {} lines from embedding file ...'.format(index))
                line = line.strip()
                if len(line) == 0:
                    continue
                slist = line.split(' ')
                word = slist[0]
                word_embeddingi = list()
                for i in range(len(slist) - 1):
                    word_embeddingi.append(float(slist[i + 1]))

                word_embedding.append(word_embeddingi)
                word_dict[word] = len(words)
                words.append(word)
                word_embeddings = np.array(word_embedding)

        return words, word_dict, word_embeddings

    def get_vocab(self):
        """
        process the vocab of model
        Args:

        Returns:
            vocab: dict, {word: index}
            words: list, [word]
        """
        data_dir = self.data_dir

        files = os.listdir(data_dir)
        files = [data_dir + f for f in files]

        ite_str = 'dialog-babi-task'
        files = [f for f in files if ite_str in f]

        # for i, f in enumerate(files):
        #     print('reading from file{}: {}'.format(i, f))

        data = []
        for f in files:
            with open(f) as fin:
                lines = fin.readlines()
                for line in lines:
                    line = str.lower(line)
                    line = line.replace('\n', '')
                    if len(line) == 0:
                        continue
                    line = line.split(' ', 1)
                    assert line[0].isdigit()
                    line = line[1]
                    line = line.replace('\t', ' ')
                    data_ = line.split(' ')
                    data += data_
        counter = collections.Counter(data)
        counter_pairs = sorted(counter.items(), key=lambda x: -x[1])

        words, _ = zip(*counter_pairs)
        words_dict = dict(zip(words, range(len(words))))

        with open(self.args.vocab_file, 'w+') as f:
            cPickle.dump(words, f)

        return words_dict, words

    def get_slot_value(self):
        """
        get slot values from data_dir

        Args:
            self
        Returns:
            slots: list, [slot]
            slot_value: dict, {slot: [value]}
            value_slot: dict, {value: slot}
        """
        files = os.listdir(self.data_dir)
        files = [self.data_dir + f for f in files]
        files = [f for f in files if 'kb-all' in f]

        f_kb = files[0]

        # print('\nreading kb information from {}'.format(f_kb))

        slot_value = {}
        value_slot = {}
        with open(f_kb) as f:
            lines = f.readlines()
            for line in lines:
                line = str.lower(line)
                line = line.replace('\n', '')
                line = line.split(' ')
                kv = line[2].split('\t')
                slot = kv[0][2:]
                value = kv[1].strip()
                if value not in value_slot:
                    value_slot[value] = slot
                    value = [value]
                    if slot not in slot_value:
                        slot_value[slot] = value
                    else:
                        slot_value[slot] += value
                else:
                    slot_ = value_slot[value]
                    if slot_ != slot:
                        print('error: same value from {} and {}, value={}'.format(slot, slot_, value))
            f.close()

        slot_value.pop('address')
        slot_value.pop('phone')

        slots = slot_value.keys()

        return slots, slot_value, value_slot

    def create_data_0(self):
        """
        create raw id data, each word is repalce by id
        session will be lengthen upto max_seq_length
        currently, seq_length means length of session
        Args:
        Returns:
            x: list, [size, seq_length_None, sentence_length_None]
            y: list, same to x
            target: list, [size, seq_length_None, target_size]
            mask: list, [size, seq_length_None, slot_size, 2]
            max_seq_length: int, max seq length
            seq_length_list: list, [size]
        """
        args = self.args

        files = os.listdir(args.data_dir)
        task_id = args.task_id
        files = [args.data_dir + f for f in files
                 if 'task{}'.format(task_id) in f and args.target_file in f and 'OOV' not in f]
        assert len(files) == 1
        target_file = files[0]

        with open(target_file) as fin:
            print('\n creating data from "{}"'.format(target_file))
            max_seq_length = -1
            max_sentence_length = -1
            lines = fin.readlines()

            counter = 0
            x = []
            y = []
            x_bov = []
            y_bov = []
            target = []
            mask = []
            seq_length_list = []
            attention = []
            attention_mask = []
            raw_data_x = []
            raw_data_y = []

            x_session = []
            y_session = []
            x_bov_session = []
            y_bov_session = []
            target_session = []
            mask_session = []
            attn_session = []
            attn_mask_session = []
            raw_data_x_session = []
            raw_data_y_session = []

            for line in lines:
                line = line.replace('\n', '')
                line = line.strip()

                # end of session
                if len(line) == 0:
                    # update max_seq_length
                    if counter > max_seq_length:
                        max_seq_length = counter
                    seq_length_list.append(counter)
                    counter = 0
                    # append new session
                    x.append(x_session)
                    # y session should shift right 1 and pad [0]
                    y_session_t = [[0]]
                    y_session_t += [y_session[i] for i in range(len(y_session) - 1)]
                    y.append(y_session_t)
                    x_bov.append(x_bov_session)
                    y_bov_session_t = [[0]]
                    y_bov_session_t += [y_bov_session[i] for i in range(len(y_bov_session) - 1)]
                    y_bov.append(y_bov_session)
                    target.append(target_session)
                    mask.append(mask_session)
                    attention.append(attn_session)
                    attention_mask.append(attn_mask_session)
                    raw_data_x.append(raw_data_x_session)
                    raw_data_y.append(raw_data_y_session)
                    # clear sessions
                    x_session = []
                    y_session = []
                    x_bov_session = []
                    y_bov_session = []
                    target_session = []
                    mask_session = []
                    attn_session = []
                    attn_mask_session = []
                    raw_data_x_session = []
                    raw_data_y_session = []
                    continue

                # in session
                # preprocess line
                line = str.lower(line)
                line = line.split(' ', 1)
                assert line[0].isdigit()
                line = line[1]
                line = line.split('\t')
                if len(line) == 1:
                    continue
                assert len(line) == 2
                # add line to x_session and y_session
                original_line1 = line[1]
                line[1] = self.convert_line1(line[1])
                x_session.append(self.get_sentence_id(line[0]))
                y_session.append(self.get_sentence_id(line[1]))
                x_bov_session.append(self.get_bov(line[0]))
                y_bov_session.append(self.get_bov(original_line1))
                att, att_msk = self.get_attention(line[0])
                attn_session.append(att)
                attn_mask_session.append(att_msk)
                raw_data_x_session.append(line[0])
                raw_data_y_session.append(line[1])
                # update max_sentence_length
                length = max(len(x_session[-1]), len(y_session[-1]))
                if length > max_sentence_length:
                    max_sentence_length = length

                # generate target vector
                da_, slot_values_, submask = self.get_da_bov(line[1])
                # convert target data from [] + [[], [], [], ...] to []
                da_slot_values = da_
                for item in slot_values_:
                    da_slot_values += item
                # append target and mask to session
                # TODO add post value to target
                target_session.append(self.get_new_target(line[0], target_session, da_slot_values))
                mask_session.append(submask)

                counter += 1

            return x, y, target, mask, max_seq_length, seq_length_list, max_sentence_length, \
                attention, attention_mask, raw_data_x, raw_data_y, x_bov, y_bov

    def get_new_target(self, line0, target_session, target):
        """
        former target are only with information of this sentence
        the new target will remain history information
        Args:
            line0: user post
            target_session: as name
            target: system response target format (dimension 50+ vec)
        Return:
            new_target
        """
        last_target = list()
        if len(target_session) > 0:
            last_target = target_session[-1]
        else:
            last_target = target
        line0 = str.lower(line0)

        datl = len(self.da_types)
        prefix = target[0:datl]
        last_target = last_target[datl:]
        target = target[datl:]

        targets = []
        split_index = [0]
        for item in self.slots:
            x = split_index[-1]
            split_index.append(x + len(self.slot_value[item]))
        words = line0.split(' ')
        for i in range(self.slot_size):
            # get line0 value for slot[i]
            indext = -1
            ite = ''
            for item in words:
                if item in self.value_slot and self.value_slot[item] == self.slots[i]:
                    indext = self.value_index[self.slots[i]][item]
                    ite = item
                    break

            start = split_index[i]
            end = split_index[i+1]
            lt = last_target[start:end]
            t = target[start:end]
            # print('item: {}, len(lt){}'.format(ite, len(lt)))
            # print('slot {}:{} len{}, indext:{}'.format(i, self.slots[i], len(self.slot_value[self.slots[i]]), indext))
            if indext > -1:
                lt = [0 for _ in lt]
                lt[indext] = 1.0
            if 1 in t:
                targets += t
            else:
                targets += lt
        return prefix + targets

    def convert_line1(self, sen):
        """
        convert the hotel_name in response to values.
            e.g. resto_a_b_c_stars3 -> a b c
        Args:
            sen: string, a response sentence
        Return:
            sen: string, new sentence
        """
        task_id = self.args.task_id
        if task_id > 2 and sen.strip().startswith("what do you think of this option: resto"):
            print('convert sentence: {}'.format(sen))
            prefix = sen.split(': ', 1)[0] + ': '
            sen = sen.split('_', 1)
            sen = sen[1]
            slist = sen.split('_')
            sen = ''
            for ite in slist:
                if 'stars' not in ite:
                    sen += ite + ' '
            sen = prefix + sen.strip()
            print(sen)
        return sen

    def get_sentence_id(self, line):
        """
        get id representation of id
        Args:
            line: string, sentence
        Returns:
            sentence: list, [word_id]
        """
        line = line.strip()
        words = line.split(' ')
        sentence = list()
        for w in words:
            sentence.append(self.word_dict[w])
        return sentence

    def create_data(self):
        """
        create raw bov data, the sequence is None currently
        the result is to be processed to fixed seq length by convert_data()
        Args:
            self
        Returns:
            x_: list, [size, None, embedding_size]
            y_: list, [size, None, embedding_size]
            target_ : list, [size, None, target_size=(act_size + sum(value_sizes))]
            max_seq_length: int, max length of sequence
        """
        args = self.args

        files = os.listdir(args.data_dir)
        task_id = args.task_id

        files = [args.data_dir + f for f in files if 'task{}'.format(task_id) in f]
        # print('files: {}'.format(files))
        trn_file = [f for f in files if self.args.target_file in f][0]

        with open(trn_file) as fin:
            max_seq_length = -1

            lines = fin.readlines()

            counter = 0
            x_ = []
            y_ = []
            target_ = []
            mask_ = []
            seq_length_list = []

            x_session = []
            y_session = []
            target_session = []
            mask_session = []

            for line in lines:
                line = str.lower(line)
                line = line.replace('\n', '')

                if len(line) == 0:
                    if counter > max_seq_length:
                        max_seq_length = counter
                    seq_length_list += [counter]
                    counter = 0
                    x_.append(x_session)
                    y_.append(y_session)
                    target_.append(target_session)
                    mask_.append(mask_session)

                    x_session = []
                    y_session = []
                    target_session = []
                    mask_session = []
                    continue

                line = line.split(' ', 1)
                assert line[0].isdigit()
                line = line[1]

                line = line.split('\t')
                if len(line) != 2:
                    continue
                assert len(line) == 2

                x_session.append(self.get_bov(line[0]))
                y_session.append(self.get_bov(line[1]))

                # [1, 2, ...], [[1, 2, 3, ...], [1, 2, 3, ...], [...]]
                da_, slot_values_, submask = self.get_da_bov(line[1])

                # convert target data from [] + [[], [], [], ...] to []
                da_slot_values = da_
                for item in slot_values_:
                    da_slot_values += item
                # print(len(da_slot_values))
                target_session.append(da_slot_values)
                mask_session.append(submask)

                counter += 1
        # mask = [session_number, -1, slot_size, 2]
        return x_, y_, target_, mask_, max_seq_length, seq_length_list

    def convert_data(self):
        """
        convert raw bov data from create_data() to fixed seq length
        specifically, convert self._data_x, self._data_y and self._data_target to fixed seq length
        Args:
            self
        Returns:
        """
        # new code
        data_size = len(self._data_x)
        data_x = np.zeros([data_size, self.seq_length, self.sentence_length])
        data_y = np.zeros([data_size, self.seq_length, self.sentence_length])
        data_target = np.zeros([data_size, self.seq_length, self.target_vec_size])
        data_mask = np.zeros([data_size, self.seq_length, self.slot_size, 2])
        self.x_sentence_length = np.zeros([data_size, self.seq_length])
        self.y_sentence_length = np.zeros([data_size, self.seq_length])
        attention = np.zeros([data_size, self.seq_length, self.slot_size, self.sentence_length])
        attention_mask = np.zeros([data_size, self.seq_length, self.slot_size, 2])
        x_bov = np.zeros([data_size, self.seq_length, self.vocab_size])
        y_bov = np.zeros([data_size, self.seq_length, self.vocab_size])

        raw_x = np.zeros([data_size, self.seq_length]).tolist()
        raw_y = np.zeros([data_size, self.seq_length]).tolist()

        for i in range(len(self._data_x)):
            # for each session:
            for j in range(self.seq_length_list[i]):
                # for each sentence
                x_len = len(self._data_x[i][j])
                y_len = len(self._data_y[i][j])
                data_x[i][j][:x_len] = self._data_x[i][j][:x_len]
                data_y[i][j][:y_len] = self._data_y[i][j][:y_len]
                self.x_sentence_length[i][j] = x_len
                self.y_sentence_length[i][j] = y_len
                attention_len = len(self.attention[i][j][0])
                for k in range(self.slot_size):
                    attention[i][j][k][:attention_len] = self.attention[i][j][k]
            attention_mask[i][:len(self.attention_mask[i])] = np.array(self.attention_mask[i])
            data_target[i][:len(self._data_target[i])] = np.array(self._data_target[i])
            data_mask[i][:len(self._data_mask[i])] = np.array(self._data_mask[i])
            raw_x[i][:len(self.raw_data_x[i])] = self.raw_data_x[i]
            raw_y[i][:len(self.raw_data_y[i])] = self.raw_data_y[i]
            x_bov[i][:len(self.x_bov[i])] = np.array(self.x_bov[i])
            y_bov[i][:len(self.y_bov[i])] = np.array(self.y_bov[i])

        self._data_x = data_x
        self._data_y = data_y
        self._data_target = data_target
        self._data_mask = data_mask
        self.seq_length_list = np.array(self.seq_length_list)
        self.attention = attention
        self.attention_mask = attention_mask
        self.raw_data_x = raw_x
        self.raw_data_y = raw_y
        self.x_bov = x_bov
        self.y_bov = y_bov

        # previous code
        # data_x = np.zeros([len(self._data_x), self.seq_length])
        # data_y = np.zeros([len(self._data_x), self.seq_length])
        # data_target = np.zeros([len(self._data_target), self.seq_length, self.target_vec_size])
        # data_mask = np.zeros([len(self._data_mask), self.seq_length, len(self.slots), 2])
        #
        # for i, _ in enumerate(self._data_x):
        #     data_x[i][:len(self._data_x[i])] = np.array(self._data_x[i])
        #     data_y[i][:len(self._data_y[i])] = np.array(self._data_y[i])
        #     data_target[i][:len(self._data_target[i])] = np.array(self._data_target[i])
        #     data_mask[i][:len(self._data_mask[i])] = np.array(self._data_mask[i])
        #
        # self._data_x = data_x
        # self._data_y = data_y
        # self._data_target = data_target
        # self._data_mask = data_mask
        # self.seq_length_list = np.array(self.seq_length_list)

    def create_batches(self):
        """
        get x and y batches based on args.batch_size
        batch_num = max(i * batch_size = c where c <= data_size)
        Args:
            self
        Returns:
            x_batches: list, [batch_num, batch_size, seq_length, vocab_size]
            y_batches: same as x_batches
            target_batches: the same
        """

        args = self.args
        total_batch_size = len(self._data_x)

        batch_num = total_batch_size // args.batch_size
        if batch_num == 0:
            assert False, 'batch_size is to large!'

        # c = list(zip(self._data_x, self._data_y, self._data_target))
        # shuffle(c)
        # self._data_x, self._data_y, self._data_target = zip(*c)

        # shuffle self._data_s
        indexes = range(len(self._data_x))

        if self.is_shuffle:
            shuffle(indexes)

        # shuffle all data
        x_batches = self._data_x
        y_batches = self._data_y
        target_batches = self._data_target
        mask_batches = self._data_mask
        seq_length_batches = self.seq_length_list
        x_sentence_length_batches = self.x_sentence_length
        y_sentence_length_batches = self.y_sentence_length
        attention_batches = self.attention
        attention_mask_batches = self.attention_mask
        raw_x = self.raw_data_x
        raw_y = self.raw_data_y
        x_bov = self.x_bov
        y_bov = self.y_bov

        for i, index in enumerate(indexes):
            x_batches[i] = self._data_x[index]
            y_batches[i] = self._data_y[index]
            target_batches[i] = self._data_target[index]
            mask_batches[i] = self._data_mask[index]
            seq_length_batches[i] = self.seq_length_list[index]
            x_sentence_length_batches[i] = self.x_sentence_length[index]
            y_sentence_length_batches[i] = self.y_sentence_length[index]
            attention_batches[i] = self.attention[index]
            attention_mask_batches[i] = self.attention_mask[index]
            # print('{}, {}, {}, {}'.format(i, index, len(raw_x), len(self.raw_data_x)))
            raw_x[i] = self.raw_data_x[index]
            raw_y[i] = self.raw_data_y[index]
            x_bov[i] = self.x_bov[index]
            y_bov[i] = self.y_bov[index]

        # now all datas are numpy
        exact_size = batch_num * args.batch_size
        x_batches = x_batches[:exact_size]
        y_batches = y_batches[:exact_size]
        target_batches = target_batches[:exact_size]
        mask_batches = mask_batches[:exact_size]
        seq_length_batches = seq_length_batches[:exact_size]
        x_sentence_length_batches = x_sentence_length_batches[:exact_size]
        y_sentence_length_batches = y_sentence_length_batches[:exact_size]
        attention_batches = attention_batches[:exact_size]
        attention_mask_batches = attention_mask_batches[:exact_size]
        raw_x = raw_x[:exact_size]
        raw_y = raw_y[:exact_size]
        x_bov = x_bov[:exact_size]
        y_bov = y_bov[:exact_size]

        args.batch_num = batch_num

        self.x_batches = np.split(x_batches, args.batch_num, 0)
        self.y_batches = np.split(y_batches, args.batch_num, 0)
        self.target_batches = np.split(target_batches, args.batch_num, 0)
        self.mask_batches = np.split(mask_batches, args.batch_num, 0)
        self.seq_length_batches = np.split(seq_length_batches, args.batch_num, 0)
        self.x_sentence_length_batches = np.split(x_sentence_length_batches, args.batch_num, 0)
        self.y_sentence_length_batches = np.split(y_sentence_length_batches, args.batch_num, 0)
        self.attention_batches = np.split(attention_batches, args.batch_num, 0)
        self.attention_mask_batches = np.split(attention_mask_batches, args.batch_num, 0)
        self.raw_data_x_batches = raw_x
        self.raw_data_y_batches = raw_y
        self.x_bov_batches = np.split(x_bov, args.batch_num, 0)
        self.y_bov_batches = np.split(y_bov, args.batch_num, 0)

        self.batch_pointer = 0

    def next_batch(self):
        """
        get next batch from data_batches
        Args:
            self
        Returns:
            self.batch_pointer: int, the index of batch. when it is larger than batch_num,
                                it will be reset to 0.
            self.x_batches[i]
            self.y_bathces[i]
            self.target_batches[i]
        """
        i = self.batch_pointer
        self.batch_pointer += 1
        if self.batch_pointer >= self.args.batch_num:
            self.batch_pointer = 0
        start = i * self.args.batch_size
        assert i < self.args.batch_num
        return self.batch_pointer, self.x_batches[i], self.y_batches[i], \
            self.target_batches[i], self.mask_batches[i], self.seqlength2mask(self.seq_length_batches[i]), \
            self.senlenconvert(self.x_sentence_length_batches[i]), \
            self.senlenconvert(self.y_sentence_length_batches[i]), \
            self.attention_batches[i], self.attention_mask_batches[i], \
            self.raw_data_x_batches[start:start + self.args.batch_size], \
            self.raw_data_y_batches[start:start + self.args.batch_size], \
            self.x_bov_batches[i], self.y_bov_batches[i]


    def senlenconvert(self, sen_length):
        """
        :param sen_length:
        :return:
        """
        lst = sen_length.tolist()
        for i in range(len(lst)):
            for j in range(len(lst[i])):
                val = lst[i][j]
                lst[i][j] = [1.0 for _ in range(self.sentence_length)]
                for k in range(self.sentence_length):
                    if k >= val:
                        lst[i][j][k] = 0.0
                if val == 0:
                    lst[i][j][0] = 1.0
        return lst

    def seqlength2mask(self, seq_lengths):
        """
        convert from list to 2D
        Args:
            seq_lengths: list(int)
        Returns:
            seq_mask: list(list(int))
        """
        args = self.args
        ret = np.zeros([args.batch_size, args.seq_length])
        for i in range(args.batch_size):
            ret[i][:seq_lengths[i]] = np.ones([seq_lengths[i]])
        return ret

    def get_bov(self, sentence):
        """
        get the bov of sentence
        Args:
            sentence: string, sentence to be convert
        Returns:
            vector: list, bov of sentence
        """

        sentence = sentence.strip()
        words = sentence.split(' ')

        vec = [0 for _ in range(self.vocab_size)]

        for word in words:
            vec[self.word_dict[word]] = 1

        return vec

    def get_da(self, sentence):
        """
        get the da(da_type, slot_values) of sentence
        Args:
            sentence, string
        Returns:
            da_sv: [dat, sv]
                dat: string
                sv: [[slot1,value1], [s2, v2], ...]
            sentence: values are replaced by '#slot_name#'
        """
        pattern = self.pattern

        dat = ''
        sv = []

        for item in pattern.items():
            pre = item[0]
            if pre in sentence:
                dat = item[1]
                break

        if len(dat) == 0 and self.args.task_id == 6:
            with open('missing_da_sen.txt', 'a+') as fout:
                fout.write(sentence + '\n')
            fout.close()
        else:
            assert len(dat) != 0

        for item in self.value_slot.items():
            if sentence.find(item[0]) > 0:
                sv.append([item[1], item[0]])
                sentence = sentence.replace(item[0], '#{}#'.format(item[1]))

        da_sv = [dat, sv]

        return da_sv, sentence

    def get_attention(self, sentence):
        """
        Args:
            sentence: str
        Returns:
            attention: list shape[slot_size, sentence_length]
            attention_mask: list, shape[slot_size]
        """
        sentence = str.lower(sentence).strip()
        words = sentence.split(' ')
        attention_mask = list()
        attention = list()
        for i in range(self.slot_size):
            attn_slot = [0 for _ in range(len(words))]
            slot_mask = [0, 1.0]
            for j, word in enumerate(words):
                if word in self.value_slot and self.value_slot[word] == self.slots[i]:
                    attn_slot[j] = 1
                    slot_mask = [1.0, 0]
                    break
            attention_mask.append(slot_mask)
            attention.append(attn_slot)
        return attention, attention_mask

    def get_da_bov(self, sentence):
        """
        get bov representation of sentence
        Args:
            sentence: string, response
        Return:
            dav_vec: list, one hot, size=len(act_types)
            value_vectors: list, [ei] where ei is one hot on slot_i values
            mask: list, [slot_size, 2], [1, 0] for exist and [0, 1] for none
        """
        da, _ = self.get_da(sentence)

        dat = da[0]
        sv = da[1]

        da_vec = np.zeros(len(self.pattern)).tolist()

        da_vec[self.dat_index[dat]] = 1

        value_vectors = []
        mask = []
        for slot in self.slots:
            dic = self.value_index[slot]
            vec = np.zeros(len(dic)).tolist()
            submask = [0.0, 1.0]
            for s_v in sv:
                if slot == s_v[0]:
                    vec[dic[s_v[1]]] = 1
                    submask = [1.0, 0.0]
                    break
            mask.append(submask)
            value_vectors.append(vec)
        return da_vec, value_vectors, mask

    def get_da_by_bov(self, da_vec, value_vectors):
        """
        get da representation from bov representation
        Args:
            da_vec: list, da vector
            value_vectors: list, each element is on hot representation
        Returns:
            da_type: string,
            slot_values: list, [[s1, v1], [s2, v2], ...]
        """

        da_type = ''
        for i, item in enumerate(da_vec):
            if item == 1:
                da_type = self.index_dat[i]
                break
        assert da_type != ''

        slot_values = []
        for i, item in enumerate(value_vectors):
            s_, v_ = self.slots[i], ''
            dic = self.index_value[s_]
            for j, it in enumerate(item):
                if it == 1:
                    v_ = dic[j]
                    break
            slot_values.append([s_, v_])

        return da_type, slot_values

    def get_pattern(self):
        """
        get pattern dict of patterns
        Args:
            self
        Returns:
            dic: dict, {pattern_prefix: da_type}
            da_types: list, [da_type]
        """
        dic = {}

        args = self.args

        da_types = []

        pattern_file = args.pattern_dir
        if args.task_id == 6:
            pattern_file.replace('pattern', 'pattern_6')

        with open(pattern_file) as f:
            lines = f.readlines()
            for line in lines:
                line = str.lower(line)
                line = line.replace('\n', '')
                line = line.split('\t')
                assert len(line) == 2
                if line[0] not in dic:
                    dic[line[0]] = line[1]
                    da_types.append(line[1])
        return dic, da_types

    def generate_value_feature(self):
        """
        generate special features for slot value word
        modifiy self.value_feature
        Args:
            None
        Returns:
            None
        """
        args = self.args
        self.value_feature = np.zeros([len(self.embedding_words), args.feature_size])
        for i, slot in enumerate(self.slots):
            values = self.slot_value[slot]
            bias = int(args.feature_size / len(values))
            start = 0
            for v in values:
                index = self.word_dict[v]
                self.value_feature[index][start:start+bias] = [1 for _ in range(bias)]
                start += bias
        self.value_feature = np.array(self.value_feature)

# test DataLoader Class

# config = Config()
# # config.task_id = _task_id
# config.save_path = config.save_path + str(config.task_id) + '/'
# config.data_dir = 'data/dialog-bAbI-tasks/'
# config.target_file = 'trn'
# dl = DataLoader(config)
# ret = dl.create_batches()
# _, _, _, target, _, _, x_sen_len, y_sen_len, attn, attn_mask, data_x, _, _, _ = dl.next_batch()
# _, _, _, target, _, _, x_sen_len, y_sen_len, attn, attn_mask, data_x, _, _, _  = dl.next_batch()
# _, _, _, target, _, _, x_sen_len, y_sen_len, attn, attn_mask, data_x, _, _, _  = dl.next_batch()
# with open('attn_log/data_test.txt', 'w+') as fout:
#     for i in range(config.batch_size):  # batch
#         for j in range(config.seq_length):  # seq
#             fout.write(data_x[i][j] + '\n')
#             for k in range(config.sentence_length):
#                 fout.write(str(x_sen_len[i][j][k]) + ' ')
#             fout.write('\n')
#             for k in range(config.slot_size):  # slot_size
#                 fout.write('\t' + config.slots[k] + '\t')
#                 fout.write('\t[' + str(attn_mask[i][j][k]) + ']\t')
#                 for item in attn[i][j][k]:
#                     fout.write(str(item) + ' ')
#                 fout.write('\n')
#         fout.write('\n')
#     fout.close()
# a = 2