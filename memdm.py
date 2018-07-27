"""
build the memory network based dialog manager model
"""

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import variable_scope as vs
import numpy as np
import memdm_cell
from datetime import datetime

class Model:
    """
    MEMDM Model, based on BasicLSTMCell
    """
    def __init__(self, args, infer=False):

        # if infer, the batch_size and seq_length should be 1
        self.args = args
        if infer:
            args.batch_size = 1

        with tf.variable_scope('MemDM', initializer=tf.random_normal_initializer()) as scope:
            with tf.device(args.device):

                # model cell
                cell_fn = memdm_cell.MEMDMCell
                cell = cell_fn(args)

                self.initial_state = memdm_cell.get_initial_states(args)

                # model inputs, including x, y, targets
                self.input_x = tf.placeholder(tf.int32, [args.batch_size, args.seq_length, args.sentence_length])
                self.input_y = tf.placeholder(tf.int32, [args.batch_size, args.seq_length, args.sentence_length])

                self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length, args.target_vec_size])
                self.masks = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.slot_size, 2])
                self.seq_masks = tf.placeholder(tf.float32, [args.batch_size, args.seq_length])

                self.x_sentence_length = tf.placeholder(tf.float32,
                                                        [args.batch_size, args.seq_length, args.sentence_length])
                self.y_sentence_length = tf.placeholder(tf.float32,
                                                        [args.batch_size, args.seq_length, args.sentence_length])
                self.attention = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.slot_size,
                                                             args.sentence_length])
                self.attention_mask = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.slot_size, 2])
                self.loss_mask = tf.placeholder(tf.float32, [5])

                # read embedding from word2vec

                self.init_k_memory = tf.get_variable("init_k_memory", [args.slot_size, args.embedding_size_0],
                                                     trainable=False)

                self.value_feature0 = tf.get_variable("value_feature", [args.vocab_size, args.feature_size],
                                                     trainable=False)

                self.embedding = tf.get_variable("embedding", [args.vocab_size, args.embedding_size_0],
                                                 trainable=False)

                self.embedding_q = tf.get_variable("embedding_q", [args.vocab_size, args.embedding_size_0],
                                                   trainable=True)
                self.embedding_a = tf.get_variable("embedding_a", [args.vocab_size, args.embedding_size_0],
                                                   trainable=True)
                self.embedding_q = tf.nn.l2_normalize(self.embedding_q, 0)
                self.embedding_a = tf.nn.l2_normalize(self.embedding_a, 0)

                # tf.summary.histogram('embedding_q', self.embedding_q)
                # tf.summary.histogram('embedding_a', self.embedding_a)

                # get free embeddings
                # [batch, seq, sen, embed]
                input_x_bov = tf.nn.embedding_lookup(self.embedding_q, self.input_x)
                input_q_bov = tf.split(1, args.seq_length, input_x_bov)
                input_q_bov = [tf.squeeze(item, [1]) for item in input_q_bov]

                input_y_bov = tf.nn.embedding_lookup(self.embedding_a, self.input_y)
                input_a_bov = tf.split(1, args.seq_length, input_y_bov)
                input_a_bov = [tf.squeeze(item, [1]) for item in input_a_bov]

                self.value_feature = tf.nn.l2_normalize(self.value_feature0, 1)

                # both with shape [batch_size, seq_length, sentence_length, word_embedding_size]
                input_x_ebd = tf.nn.embedding_lookup(self.embedding, self.input_x)
                input_y_ebd = tf.nn.embedding_lookup(self.embedding, self.input_y)
                input_x_feature = tf.nn.embedding_lookup(self.value_feature, self.input_x)
                input_y_feature = tf.nn.embedding_lookup(self.value_feature, self.input_y)
                input_x_ebd = tf.concat(3, [input_x_ebd, input_x_feature])
                input_y_ebd = tf.concat(3, [input_y_ebd, input_y_feature])

                input_q = tf.split(1, args.seq_length, input_x_ebd)
                input_a = tf.split(1, args.seq_length, input_y_ebd)
                # shape [seq_length * tensor(batch_size, 1, sentence_length, word_embedding_size)]
                x_sen_length = tf.split(1, args.seq_length, self.x_sentence_length)
                y_sen_length = tf.split(1, args.seq_length, self.y_sentence_length)

                input_q = [tf.squeeze(input_, [1]) for input_ in input_q]
                input_a = [tf.squeeze(input_, [1]) for input_ in input_a]
                # shape [seq_length, tensor(batch_size, sentence_length, word_embedding_size)]
                x_sen_length = [tf.squeeze(item, [1]) for item in x_sen_length]
                y_sen_length = [tf.squeeze(item, [1]) for item in y_sen_length]

                # shape: seq_length tensor[batch_size, element]
                # attention shape [seq_length, ]
                self.initial_state = memdm_cell.update_init_state(self.initial_state, self.init_k_memory)
                # run seq2seq model
                outputs, k_memorys, v_memorys, attention, last_state, tmp_result, g_result, e_memorys = rnn_decoder(
                    input_q, input_a, input_q_bov, input_a_bov, x_sen_length, y_sen_length, self.initial_state, cell)

                _outputs = np.zeros([args.batch_size, args.seq_length]).tolist()
                _v_memorys = np.zeros([args.batch_size, args.seq_length]).tolist()
                _e_memorys = np.zeros([args.batch_size, args.seq_length]).tolist()
                attn = np.zeros([args.batch_size, args.seq_length]).tolist()
                tmp_val = np.zeros([args.batch_size, args.seq_length]).tolist()
                g_val = np.zeros([args.batch_size, args.seq_length]).tolist()

                for i in range(args.seq_length):
                    outputi = outputs[i]
                    v_memoryi = v_memorys[i]
                    e_memoryi = e_memorys[i]

                    outputi = tf.split(0, args.batch_size, outputi)
                    outputi = [tf.squeeze(item) for item in outputi]

                    v_memoryi = tf.split(0, args.batch_size, v_memoryi)
                    v_memoryi = [tf.squeeze(item) for item in v_memoryi]

                    e_memoryi = tf.split(0, args.batch_size, e_memoryi)
                    e_memoryi = [tf.squeeze(item) for item in e_memoryi]

                    for j in range(args.batch_size):
                        _outputs[j][i] = outputi[j]
                        _v_memorys[j][i] = v_memoryi[j]
                        _e_memorys[j][i] = e_memoryi[j]
                        attn[j][i] = attention[i][j]
                        tmp_val[j][i] = tmp_result[i][j]
                        g_val[j][i] = g_result[i][j]
                self.attention_result = attn
                self.tmp_val = tmp_val
                # shape [batch_size, seq_length, slot_size]
                self.g_val = g_val
                self.reuse_tag = 1

                # declaration of weight, bias of both da type and slots
                width = 0
                for item in args.slots:
                    value_num = len(args.slot_value[item])
                    width += value_num
                # self.weight_da = tf.get_variable('weight_da', [args.e_memory_length * args.e_memory_size,
                #                                                len(self.args.da_types)])
                le = args.e_memory_length * args.e_memory_size
                lv = args.slot_size * args.v_memory_size
                self.weight_da = tf.get_variable('weight_da', [le,
                                                               len(self.args.da_types)])
                self.bias_da = tf.get_variable('bias_da', [len(self.args.da_types)])
                self.weight_slot = []
                self.bias_slot = []
                self.weight_mask = []
                self.bias_mask = []
                for i, item in enumerate(args.slots):
                    value_num = len(args.slot_value[item])
                    self.weight_slot.append(tf.get_variable(
                            'weight_slot' + str(i), [self.args.v_memory_size, value_num]))
                    self.bias_slot.append(tf.get_variable('bias_slot' + str(i), [value_num]))

                    # self.weight_mask.append(tf.get_variable(
                    #         'weight_mask' + str(i), [args.e_memory_length * args.e_memory_size, 2]))
                    self.weight_mask.append(tf.get_variable(
                            'weight_mask' + str(i), [le, 2]))
                    self.bias_mask.append(tf.get_variable('bias_mask' + str(i), [2]))

                # result prob and msk to be calculated in loss_batch
                self.prob = []
                self.slot_prob = []
                self.mask_result = []

                # using outputs to predit prob
                # then use prob and target to calculate loss
                loss1, loss2, loss3, loss4, loss5 = self.loss_batch(_outputs, _v_memorys, _e_memorys, attn)
                loss1 = tf.reshape(loss1, [])
                loss2 = tf.reshape(loss2, [])
                loss3 = tf.reshape(loss3, [])
                loss4 = tf.reshape(loss4, [])
                loss5 = tf.reshape(loss5, [])

                # normalize loss
                # TODO: the loss number should be correctly calculated
                loss1 /= args.batch_size * args.seq_length
                loss2 /= args.batch_size * args.slot_size
                loss3 /= args.batch_size * args.seq_length
                loss4 /= args.batch_size * args.seq_length * args.slot_size
                loss5 /= args.batch_size * args.seq_length * args.slot_size

                # different loss mode, controlled by config.moss_mode
                loss = loss1 * self.loss_mask[0] + \
                       loss2 * self.loss_mask[1] + \
                       loss3 * self.loss_mask[2] + \
                       loss4 * self.loss_mask[3] + \
                       loss5 * self.loss_mask[4]

                if args.loss_mode == 1:
                    loss = loss1
                elif args.loss_mode == 2:
                    loss = loss2
                elif args.loss_mode == 3:
                    loss = loss3
                elif args.loss_mode == 4:
                    loss = loss4

                # assign to self variables
                self.cost = loss
                self.cost1 = loss1
                self.cost2 = loss2
                self.cost3 = loss3
                self.cost4 = loss4
                self.cost5 = loss5

                # summary 4 losses
                tf.summary.histogram('total_cost', self.cost)
                tf.summary.histogram('act_type_cost', self.cost1)
                tf.summary.histogram('slot_value_cost', self.cost2)
                tf.summary.histogram('attention_lost', self.cost3)
                tf.summary.histogram('mask_lost', self.cost4)

                self.final_state = last_state

                # train parameter setting
                self.lr = tf.Variable(0.0, trainable=False)
                t_vars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, t_vars),
                                                  args.grad_clip)
                optimizer = tf.train.AdamOptimizer(self.lr)
                self.train_op = optimizer.apply_gradients(zip(grads, t_vars))

                slot_vars = self.weight_slot + self.bias_slot
                grads2, _ = tf.clip_by_global_norm(tf.gradients(self.cost, slot_vars),
                                                   args.grad_clip)
                optimizer2 = tf.train.AdamOptimizer(self.lr)
                self.train_op2 = optimizer2.apply_gradients(zip(grads2, slot_vars))

    # TODO: repaire loss function
    # these variables will be used self.targets, self.masks, self.seq_masks):
    def loss_batch(self, outputs, v_memorys, e_memorys, attn):
        """compute the total loss of model
        Args:
            outputs: list, [batch_size, seq_length] * element
            delete k_memorys: list, [batch_size, seq_length] * element
            v_memorys: list, [batch_size, seq_length] * element
        """
        targets = tf.split(0, self.args.batch_size, self.targets)
        targets = [tf.squeeze(item) for item in targets]
        # shape [batch_size, tensor(seq_length, target_size)]

        masks = tf.split(0, self.args.batch_size, self.masks)
        masks = [tf.squeeze(item) for item in masks]
        # shape [batch_size, tensor(seq_length, slot_size 2)]

        seq_masks = tf.split(0, self.args.batch_size, self.seq_masks)
        seq_masks = [tf.reshape(item, [self.args.seq_length]) for item in seq_masks]
        # shape = [batch_size, tensor(1)]

        attention = tf.split(0, self.args.batch_size, self.attention)
        attention = [tf.squeeze(item) for item in attention]
        # shape [batch_size, tensor(seq_length, slot_size, sentence_length)]

        attention_mask = tf.split(0, self.args.batch_size, self.attention_mask)
        attention_mask = [tf.squeeze(item) for item in attention_mask]
        # shape [batch_size, tensor(seq_length, slot_size)]

        _loss1 = 0.0
        _loss2 = 0.0
        _loss3 = 0.0
        _loss4 = 0.0
        _loss5 = 0.0
        for i in range(self.args.batch_size):
            l1, l2, l3, l4, l5, p, sp, msk = self.loss_seq(outputs[i], v_memorys[i], e_memorys[i], targets[i], masks[i],
                                                           seq_masks[i], attn[i], attention[i], attention_mask[i],
                                                           self.g_val[i])
            _loss1 += l1
            _loss2 += l2
            _loss3 += l3
            _loss4 += l4
            _loss5 += l5
            self.prob.append(p)
            self.slot_prob.append(sp)
            self.mask_result.append(msk)
        return _loss1, _loss2, _loss3, _loss4, _loss5

    def loss_seq(self, outputs, v_memorys, e_memorys, targets, masks, seq_masks, attni, attentioni, attention_maski, g_val):
        """
        cpt loss by for s seq
        Args:
            outputs: list, [seq_length, tensor(rnn_size)]
            v_memorys: list, [seq_length, tensor(slot_size, v_mem_size]
            targets: tensor, [seq_length, target_size]
            masks: tensor, [seq_length, slot_size, 2]
            seq_masks: tensor, [seq_length]
            attni:
            attentioni:
            attention_maski:
            g_val:
        Return
        """

        targets = tf.split(0, self.args.seq_length, targets)
        masks = tf.split(0, self.args.seq_length, masks)
        seq_masks = tf.split(0, self.args.seq_length, seq_masks)
        attentioni = tf.split(0, self.args.seq_length, attentioni)
        attention_maski = tf.split(0, self.args.seq_length, attention_maski)

        targets = [tf.reshape(item, [self.args.target_vec_size]) for item in targets]
        masks = [tf.reshape(item, [self.args.slot_size, 2]) for item in masks]
        seq_masks = [tf.reshape(item, [1]) for item in seq_masks]
        attentioni = [tf.reshape(item, [self.args.slot_size, self.args.sentence_length]) for item in attentioni]
        attention_maski = [tf.reshape(item, [self.args.slot_size, 2]) for item in attention_maski]

        _loss1 = 0.0
        _loss2 = 0.0
        _loss3 = 0.0
        _loss4 = 0.0
        _loss5 = 0.0

        prob = []
        slot_prob = []
        mask_result = []
        for i in range(self.args.seq_length):
            l1, l2, l3, l4, l5, _p, _sp, msk = self.loss(
                    outputs[i], v_memorys[i], e_memorys[i], targets[i], masks[i], attni[i],
                    attentioni[i], attention_maski[i], g_val[i])
            _loss1 += l1 * seq_masks[i]
            _loss2 += l2 * seq_masks[i]
            _loss3 += l3 * seq_masks[i]
            _loss4 += l4 * seq_masks[i]
            _loss5 += l5 * seq_masks[i]
            prob.append(_p)
            slot_prob.append(_sp)
            mask_result.append(msk)

        return _loss1, _loss2, _loss3, _loss4, _loss5, prob, slot_prob, mask_result

    def loss(self, output, v_memory, e_memory, target, mask, attnij, attentionij, attention_maskij, g_val):
        """
        cpt loss for a cell
        Args:
            output: tensor(rnn_size)
            delete k_memory: tensor(slot_size, slot_size)
            v_memory: tensor(slot_size, embedding_size)
            target: tensor(target_size)
            mask: tensor(slot_size, 2)
            attnij: list of tensors [slot_size, Tensor(sentence_length)]
            attentionij: tensor [slot_size, sentence_length]
            attention_maskij: tensor[slot_size, 2]
            g_val: tensor[slot_size, 2]
        """
        # _target = [da_type_size, slot1_size, slot2_size, ..., slotn_size]
        _target = tf.split_v(target, self.args.split_sizes, 0)
        da = _target[0]  # da_type vector
        values = _target[1:]  # slot_value vectors

        le = self.args.e_memory_length * self.args.e_memory_size
        lv = self.args.slot_size * self.args.v_memory_size

        # 1st, da_type loss and prob
        _loss1 = 0.0
        # outputx = tf.reshape(output, [1, self.args.rnn_size])
        # vc = tf.reshape(v_memory, [1, self.args.slot_size * self.args.v_memory_size])
        # ov = tf.concat(1, [outputx, vc])
        # vc = tf.reshape(e_memory, [1, self.args.e_memory_length * self.args.e_memory_size])
        ve = tf.reshape(e_memory, [1, le])
        # vv = tf.reshape(v_memory, [1, lv])
        # vc = tf.concat(1, [ve, vv])
        logits = tf.matmul(ve, self.weight_da) + self.bias_da
        _loss1 += tf.nn.softmax_cross_entropy_with_logits(tf.squeeze(logits), tf.squeeze(da))
        p = tf.nn.softmax(logits)

        # 2nd, slot_value loss and slot_prob
        _loss2 = 0.0
        sp = []
        for i, item in enumerate(self.args.slots):
            tgt = values[i]
            # kv_concat = tf.concat(0, [self.init_k_memory[i], v_memory[i]])
            logits = tf.matmul(tf.reshape(v_memory[i], [1, self.args.v_memory_size]), self.weight_slot[i]) \
                + self.bias_slot[i]
            # TODO: temporary attention mask
            _loss2 += attention_maskij[i][0] * tf.nn.softmax_cross_entropy_with_logits(tf.squeeze(logits), tf.squeeze(tgt))
            sp.append(tf.nn.softmax(logits))

        # 3rd, attention loss
        _loss3 = 0.0
        attentionij = tf.split(0, self.args.slot_size, attentionij)
        attentionij = [tf.squeeze(item) for item in attentionij]
        for i in range(self.args.slot_size):
            _loss3 += attention_maskij[i][0] * self.cross_entropy(attentionij[i], attnij[i])

        # 4th, mask loss and msk
        _loss4 = 0.0
        msk = []
        mask = tf.split(0, self.args.slot_size, mask)
        for i, item in enumerate(self.args.slots):
            logits = tf.matmul(ve, self.weight_mask[i]) + self.bias_mask[i]
            _loss4 += tf.nn.softmax_cross_entropy_with_logits(tf.squeeze(logits), tf.squeeze(mask[i]))
            msk.append(tf.nn.softmax(tf.squeeze(logits)))

        # 5th, g loss
        _loss5 = 0.0
        atm = tf.reshape(attention_maskij, [self.args.slot_size, 2])
        g_val = tf.reshape(g_val, [self.args.slot_size, 2])
        for i, _ in enumerate(self.args.slots):
            # _loss5 += tf.nn.softmax_cross_entropy_with_logits(g_val[i], atm[i])
            _loss5 += self.cross_entropy(atm[i], g_val[i])
        return _loss1, _loss2, _loss3, _loss4, _loss5,  p, sp, msk

    def cross_entropy(self, p, q):
        """
        get the corss entropy loss of 2 distributions p and q
        Args:
            p: tensor, real target
            q: tensor, calculated target
        Return:
            loss: cross entropy of p and q
        """
        p = tf.squeeze(p)
        q = tf.squeeze(q)
        with tf.variable_scope('cross_entropy'):
            # TODO: check if tf.clip have effect on grad propagation
            loss = - tf.reduce_sum(p * tf.log(tf.clip_by_value(q, 1e-10, 1.0)))
            loss /= self.args.sentence_length
            return loss

    def get_output(self, output_):
        """
        get the real output of task from output embedding
        Args:
            output_: vector, output embedding from rnn model
        Returns:
            output: output prediction of task
        """
        args = self.args
        print('output size: [{}]'.format(output_.get_shape()))

        w_embedding = tf.get_variable("w_embedding", [args.rnn_size, args.target_vec_size])
        b_embedding = tf.get_variable("b_embedding", [args.target_vec_size])

        output = tf.einsum('ij,jk->ik', output_, w_embedding) + b_embedding

        return output

def rnn_decoder(input_q, input_a, input_q_bov, input_a_bov, x_sentence_length, y_sentence_length, initial_state, cell, scope=None):
    """RNN decoder for the sequence-to-sequence model.
    Args:
        # these are definitions of seq2seq.rnn_decoder, we still use that
        input_q: A list of 2D Tensors [batch_size x input_size].
        input_a: A list of 2D Tensors [batch_size x input_size].
        x_sentence_length: a list of 2D Tensors [batch_size]
        y_sentence_length: same to x_sentence_length
        initial_state: 2D Tensor with shape [batch_size x cell.state_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

    Returns:
        # these are definition of seq2seq.rnn_decoder where are not used in our model
        # A tuple of the form (outputs, state), where:
        #     outputs: A list of the same length as decoder_inputs of 2D Tensors with
        #         shape [batch_size x output_size] containing generated outputs.
        #     state: The state of each cell at the final time-step.
        #         It is a 2D Tensor of shape [batch_size x cell.state_size].
        #         (Note that in some cases, like basic RNN cell or GRU cell, outputs and
        #          states can be the same. They are different for LSTM cells though.)
        outputs: real outputs of model task
        attention: list of tensors [seq_length, batch_size, slot_size, Tensor(sentence_length)]
        state: the state of each cell at the final time-step
    """
    with vs.variable_scope(scope or 'rnn_decoder'):
        state = initial_state
        outputs = []
        k_memorys = []
        v_memorys = []
        attn = []
        tmp_result = []
        g_result = []
        e_memorys = []
        # print('shape [{}] [{}]'.format(input_q[0].get_shape(), input_a[0].get_shape()))
        # decoder_inputs = tf.concat(2, [input_q, input_a], name='decoder_inputs')
        # decoder_inputs = [tf.concat(1, [input_q[i], input_a[i]]) for i in range(len(input_q))]
        # print('shape [{}]'.format(decoder_inputs[0].get_shape()))

        for i in range(len(input_q)):
            if i > 0:
                vs.get_variable_scope().reuse_variables()
            a = input_a[i]
            q = input_q[i]
            qbov = input_q_bov[i]
            abov = input_a_bov[i]
            x_length = x_sentence_length[i]
            y_length = y_sentence_length[i]
            state = cell([q, a, qbov, abov, x_length, y_length], state, i)
            # state, value_memory
            outputs.append(state[0])
            k_memorys.append(state[1])
            v_memorys.append(state[2])
            attn.append(state[3])
            e_memorys.append(state[4])
            tmp_result.append(state[-1])
            g_result.append(cell.g_batch)
            state = state[0:len(state)-1]
    return outputs, k_memorys, v_memorys, attn, state, tmp_result, g_result, e_memorys
