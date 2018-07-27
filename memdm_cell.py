import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
import numpy as np


class MEMDMCell(rnn_cell.RNNCell):
    """MEMDM model cell
    mainly based on RNN structure
    new module:
        k-v memory
        external memory

    """

    def __init__(self, args, activation=tanh):
        """Initialize the basic LSTM cell.
        Args:
            args: Config, parameters
            activation: activation function
        """
        self._num_units = args.rnn_size
        # size is embedding size
        self.v_memory_size = args.v_memory_size
        self.k_memory_size = args.k_memory_size
        self.e_memory_size = args.e_memory_size
        # length is embedding number
        self.e_memory_length = args.e_memory_length
        self.batch_size = args.batch_size
        self.embedding_size = args.embedding_size
        self.embedding_size_0 = args.embedding_size_0
        self.slot_size = args.slot_size
        self.sentence_length = args.sentence_length
        self.g_batch = []
        self.coefficient = args.cofficient

    @property
    def state_size(self):
        # return (rnn_cell.LSTMStateTuple(self._num_units, self._num_units)
                # if self._state_is_tuple else 2 * self._num_units)
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    # TODO: check carefully
    def __call__(self, inputs, state_variables, isStart, scope=None):
        """
        memory cell for 1 data point, no batch size considered anymore
        Args:
            inputs = [x, y, x_length, y_length]
                x: shape[batch_size, ]
            state = [state, k_memory, v_memory, e_memory, wr(for reading e_memory)]
            isStart: whether is the first cell
            all the above tensors are of [batch_size, None] shape
        Returns:
            state 2D
            k_memory 3D
            v_memory 3D
            e_memory 3D
            wr
        """
        with vs.variable_scope(scope or type(self).__name__):  # "MEMDMCell"
            # only k memory: [slot_size, embedding_size]
            state0, k_memory, v_memory0, _, e_memory0, wr0 = state_variables
            # q0, a0: Tensor [batch_size, sentence_length, embedding_size]
            # x and y_sentence_length: Tensor [batch_size, None(sentence length)]
            q0, a0, qb, ab, x_sentence_length, y_sentence_length = inputs

            km_list = tf.split(0, self.slot_size, k_memory)
            km_list = [tf.squeeze(item, [0]) for item in km_list]

            st_list = tf.split(0, self.batch_size, state0)
            vm_list = tf.split(0, self.batch_size, v_memory0)
            vm_list = [tf.squeeze(item, [0]) for item in vm_list]
            em_list = tf.split(0, self.batch_size, e_memory0)
            wr_list = tf.split(0, self.batch_size, wr0)

            q_list = tf.split(0, self.batch_size, q0)
            q_list = [tf.squeeze(item, [0]) for item in q_list]

            attention = []
            tmp_result = []

            # TODO: change a method to read from v_memory,
            # now it is mean of all values

            # 0st: STEP: read from v_ memory
            v_read = tf.reduce_mean(v_memory0, 1)

            xsl = tf.reshape(x_sentence_length, [self.batch_size, self.sentence_length, 1])
            ysl = tf.reshape(y_sentence_length, [self.batch_size, self.sentence_length, 1])
            qb = qb * xsl
            ab = ab * ysl
            q0_ = tf.reduce_mean(qb, 1)
            # a0_ = tf.reduce_mean(ab, 1)

            ques_list = tf.split(0, self.batch_size, q0_)

            # 1st STEP: read from e_memory
            with tf.variable_scope('read_external_memory') as scope:

                wg = tf.get_variable('weight_ewg', [self._num_units, 1])
                v = tf.get_variable('weight_evr', [self.e_memory_size + self._num_units, 1])

                e_read = []
                read_weight = []
                for i in range(self.batch_size):
                    state = tf.squeeze(ques_list[i])
                    e_memory = tf.squeeze(em_list[i])
                    wr = tf.squeeze(wr_list[i])

                    gr = sigmoid(tf.matmul(tf.reshape(state, [1, self._num_units]), wg))
                    gr = tf.reshape(gr, [1])

                    state_list = tf.pack([state for _ in range(self.e_memory_length)])
                    list1 = tf.concat(1, [e_memory, state_list])
                    # _wr, [e_memory_length, 1]
                    _wr = tf.nn.softmax(tf.matmul(list1, v))
                    _wr = tf.reshape(wr, [self.e_memory_length, 1]) * gr + _wr * (1 - gr)
                    e_read.append(tf.reduce_sum(tf.mul(e_memory, _wr), 0))
                    read_weight.append(tf.squeeze(_wr))
                e_read = tf.pack(e_read)
                read_weight = tf.pack(read_weight)

            # 2nd STEP: using q, a, rv, re to update state
            # now both shape sentence_length * [batch_size, embedding_size]
            # delete the padding of sentence
            # qb [batch_size, sentence_length, embedding_size_0]
            # x_sentence_length [batch_size, sentence_length]
            # qb = qb * x_sentence_length
            # ab = ab * y_sentence_length


            # _state = linear([q0_, a0_, v_read, e_read], self._num_units, True)
            _state = linear([q0_, v_read, e_read, state0], self._num_units, True)

            # 3rd STEP: write v_memory
            with tf.variable_scope('write_value_memory') as scope:

                wattn, biasattn = list(), list()
                for i in range(self.slot_size):
                    wattn.append(tf.get_variable('weight_attention' + str(i),
                                                 [self.k_memory_size + self.v_memory_size, 1]))
                    biasattn.append(tf.get_variable('bias_attention' + str(i), [1]))

                wg = []
                bg = []
                for i in range(self.slot_size):
                    # wg.append(tf.get_variable('weight_vwg' + str(i),
                    #           [self.k_memory_size + self.v_memory_size, self.embedding_size_0]))
                    # bg.append(tf.get_variable('weight_vbg' + str(i), [self.embedding_size_0]))
                    # vg.append(tf.get_variable('weight_vvg' + str(i), [self.embedding_size_0 / 2, 1]))
                    wg.append(tf.get_variable('weight_vwg' + str(i),
                              [self.k_memory_size + self.v_memory_size, 2]))
                    bg.append(tf.get_variable('weight_vbg' + str(i), [2]))

                _v_memory = []
                g_batch = []
                for i in range(self.batch_size):
                    # attention batch
                    attention_batch = []
                    tmp_batch = []
                    # v_memory shape [slot_size, v_memory_size]
                    v_memory = vm_list[i]
                    # question shape [sentence_length, embedding_size]
                    question = q_list[i]
                    v_mem = []
                    v_memory = tf.split(0, self.slot_size, v_memory)
                    sentence_length = x_sentence_length[i]
                    g_slot = []
                    for j in range(self.slot_size):
                        # km [embedding_size], vm [1, v_memory_size]
                        vm = tf.reshape(v_memory[j], [self.embedding_size])
                        km = km_list[j]
                        kms = tf.pack([km for _ in range(self.sentence_length)])

                        ke = tf.concat(1, [kms, question])

                        e_ke = tf.matmul(ke, wattn[j]) + biasattn[j]
                        e_ke = tf.exp(e_ke)
                        e_ke = e_ke * tf.reshape(sentence_length, [self.sentence_length, 1])
                        softmax_sum = tf.reduce_sum(e_ke)
                        e_ke /= softmax_sum
                        # now e_ke is already a softmax result
                        tmp_batch.append(tf.squeeze(e_ke))

                        alpha = e_ke
                        attention_batch.append(e_ke)
                        new_value = tf.reduce_mean(question * tf.reshape(alpha, [self.sentence_length, 1]), 0)
                        # update gate
                        kv = tf.concat(1, [tf.reshape(km, [1, self.k_memory_size]),
                                           tf.reshape(new_value, [1, self.v_memory_size])])
                        # g = sigmoid(tf.matmul(sigmoid(tf.matmul(kv, wg[j]) + bg[j]), vg[j]))
                        # g = tf.reshape(g, [1])
                        g_logits = tf.reshape(tf.matmul(kv, wg[j]) + bg[j], [2])
                        g = tf.nn.softmax(self.coefficient * g_logits)
                        # add g to output
                        g_slot.append(g)
                        v = (1 - g[0]) * vm + g[0] * new_value
                        v_mem.append(v)
                    g_batch.append(g_slot)
                    v_mem = tf.pack(v_mem)
                    _v_memory.append(v_mem)
                    # attention_batch: list, slot_size * Tensor[sentence_length]
                    attention.append(attention_batch)
                    tmp_result.append(tmp_batch)
                self.g_batch = g_batch
                # _v_memory = v_memory0


            # 4th STEP: write e_memory
            with tf.variable_scope('write_external_memory'):
                we = tf.get_variable('weight_ewe', [self._num_units, self.e_memory_size])
                wa = tf.get_variable('weight_ewa', [self._num_units, self.e_memory_size])

                read_weight_list = tf.split(0, self.batch_size, read_weight)
                _e_memory = []
                for i in range(self.batch_size):
                    state = tf.squeeze(ques_list[i])
                    e_memory = tf.squeeze(em_list[i])
                    _wr = read_weight_list[i]

                    state2d = tf.reshape(state, [1, self._num_units])
                    # both [1, memory_size]
                    ue = sigmoid(tf.matmul(state2d, we))
                    ua = sigmoid(tf.matmul(state2d, wa))
                    # _wr [e_memory_length, 1]
                    _wr = tf.reshape(_wr, [self.e_memory_length, 1])
                    e_memory1 = e_memory * (1 - tf.matmul(_wr, ue))
                    e_memory1 = e_memory1 + tf.matmul(_wr, ua)
                    _e_memory.append(e_memory1)
                _e_memory = tf.pack(_e_memory)


                # batch version
                # [batch_size, memory_size]
                # ue = tf.math_ops.sigmoid(tf.matmul(state, we))
                # ua = tf.math_ops.sigmoid(tf.matmul(state, wa))
                #
                # ue = tf.concat(1, [ue for _ in range(self.e_memory_length)])
                # ue = tf.reshape(ue, [self.batch_size, self.e_memory_length, self.e_memory_size])
                #
                # ua = tf.concat(1, [ua for _ in range(self.e_memory_length)])
                # ua = tf.reshape(ua, [self.batch_size, self.e_memory_length, self.e_memory_size])
                #
                # __wr = tf.reshape(_wr, [self.batch_size, self.e_memory_length, 1])
                # __wr = tf.concat(2, [__wr for _ in range(self.e_memory_size)])
                #
                # _e_memory = e_memory * (1 - ue * __wr)
                # _e_memory = _e_memory + __wr * ua

            # return new state, k_memory, v_memory, e_memory
            return _state, k_memory, _v_memory, attention, _e_memory, read_weight, tmp_result


def get_initial_states(args):
    """
    get initial state of MEMDM_cell
    Args:
        args: Config
        init_k_memory: k_memory with shape(slot_size, embedding_size)
    Return:
        list of states [
            state, k_memory, v_memory, e_memory, wr
        ]
    """
    state = tf.zeros([args.batch_size, args.rnn_size])
    v_memory = tf.zeros([args.batch_size, args.slot_size, args.v_memory_size])
    e_memory = tf.zeros([args.batch_size, args.e_memory_length, args.e_memory_size])

    wr = tf.zeros([args.batch_size, args.e_memory_length])

    return [state, v_memory, None, e_memory, wr]


def update_init_state(state, kmem):
    length = len(state)
    ste = [0 for _ in range(length + 1)]
    ste[0] = state[0]
    for i in range(length+1):
        if i >= 2:
            ste[i] = state[i-1]
    ste[1] = kmem
    return ste


def linear(inputs, output_size, use_bias=False, scope=None):
    """
    linear function, operate like Y = W * X + B
    Args:
        inputs: list of tensors, each element of shape [batch_size, embedding_size]
                the tensors will be concat to get preprocessed result
        output_size: the output size of linear operate
                in rnn like model, it is rnn_size
        use_bias: boolean, if we need to use bias
        scope: name scope
    Returns:
        outputs: tensor, the result of linear
                shape [batch_size, output_size]
    """
    # get input size of inputs
    shape = [t.get_shape().as_list() for t in inputs]
    input_size = 0
    for s in shape:
        if not len(s) == 2:
            raise ValueError('input rank is not 2 but {}.'.format(len(s)))
        input_size += s[1]

    # do matmul operation
    with vs.variable_scope(scope or 'MEMDMCell_linear'):

        # get matrix and bias
        matrix = tf.get_variable('matrix', [input_size, output_size], tf.float32)
        bias = tf.get_variable('bias', [output_size], tf.float32)

        concat_inputs = tf.concat(1, inputs, 'concat_inputs')

        outputs = tf.matmul(concat_inputs, matrix)

        if not use_bias:
            return outputs

        outputs = tf.add(outputs, bias)

        return outputs
