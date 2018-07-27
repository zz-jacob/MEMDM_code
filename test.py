import tensorflow as tf
from config import Config
from memdm import Model
from data_utils import DataLoader
import numpy as np
import datetime
import time


def main():
    config = Config()
    config.data_dir = 'data/dialog-bAbI-tasks/'
    config.target_file = 'tst'
    config.save_path = config.save_path + str(config.task_id) + '/'
    config.is_shuffle = False
    sample(config)
    return


def sample(config):
    """
    :return:
    """
    # config.batch_size = 1
    loader = DataLoader(config)
    print('\nconfig vocab_size: {}, batch_num: {}, seq_length: {}'.format(
        config.vocab_size, config.batch_num, config.seq_length))
    config.create_properties()

    print('\nstart building models ...')
    start = time.time()
    model = Model(config)
    print('... finished!')
    print('building model time: {}'.format(time.time() - start))

    print('\ncreating batches ...')
    loader.create_batches()

    print('... finished!')

    # [0] is right number, [1] is total number
    # for da
    dt_rate = [0.0, 0.0]
    dt_rate_session = [0.0, 0.0]

    # for slot_value
    sv_rate = [0.0, 0.0]
    sv_rate_session = [0.0, 0.0]

    # for mask
    mask_rate = [0.0, 0.0]
    mask_rate_session = [0.0, 0.0]

    # for total
    rate = [0.0, 0.0]
    rate_session = [0.0, 0.0]

    test_dir = config.test_path + str(config.task_id) + '/'

    _time = str(datetime.datetime.now())
    _time = _time.replace(':', '_')
    _time = _time.replace(' ', '_')
    _time = _time.replace('.', '_')

    test_file = test_dir + _time + '.csv'
    test_file = open(test_file, 'w+')

    _config = tf.ConfigProto()
    _config.gpu_options.allow_growth = True
    if not config.device == '/cpu:0':
        _config.allow_soft_placement = True

    with tf.Session(config=_config) as sess:

        loss_mask = [1.0, 0.0, 0.0, 0.0, 0.0]

        saver = tf.train.Saver()

        sess.run(tf.assign(model.embedding, config.embeddings))
        sess.run(tf.assign(model.init_k_memory, config.slot_embedding))
        sess.run(tf.assign(model.value_feature0, loader.value_feature))

        for batch_index in range(config.batch_num):

            ckpt = tf.train.get_checkpoint_state(config.save_path)
            if ckpt and ckpt.model_checkpoint_path and batch_index == 0:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('\nmodel restored from {}'.format(ckpt.model_checkpoint_path))

            index, x, y, t, slot_mask, seq_lengths_mask, x_sen_len, y_sen_len, attn, attn_mask, \
            raw_x, raw_y, _, _ = loader.next_batch()
            print('\nbatch {} / {}'.format(batch_index, config.batch_num))
            feed = {model.input_x: x, model.input_y: y,
                    model.targets: t, model.masks: slot_mask, model.seq_masks: seq_lengths_mask,
                    model.x_sentence_length: x_sen_len, model.y_sentence_length: y_sen_len,
                    model.attention: attn, model.attention_mask: attn_mask,
                    model.loss_mask: loss_mask}

            [prob, slot_prob, mask_result,  attention_result] = sess.run(
                [model.prob, model.slot_prob, model.mask_result, model.attention_result], feed)

            # calculate result and print result
            for i in range(config.batch_size):
                dt_session_right = True
                sv_session_right = True
                mask_session_right = True
                test_file.write('session {}/{}\n'.format(batch_index * config.batch_size + i,
                                                         config.batch_num * config.batch_size))
                for j in range(config.seq_length):
                    sentence_right = True
                    if raw_y[i][j] == 0.0 or raw_x[i][j] == 0.0:
                        assert raw_y[i][j] == 0.0 or raw_x[i][j] == 0.0
                        break

                    # write x sentence
                    test_file.write('post: {},'.format(j))
                    test_file.write(str(raw_x[i][j]).strip().replace(' ', ',') + '\n')

                    # write slot attention result
                    for m in range(config.slot_size):
                        slot = config.slots[m]
                        test_file.write('[{}],'.format(slot))
                        att_ = attention_result[i][j][m]
                        for n in range(config.sentence_length):
                            test_file.write('{:.3f},'.format(att_[n][0]))
                        test_file.write('\n')

                    # write y sentence
                    test_file.write('response: {},'.format(j))
                    test_file.write(str(raw_y[i][j]).strip().replace(' ', ',') + '\n')

                    # split target
                    tars = np.split(t[i][j], config.numpy_split_sizes)
                    da_tar, sv_tar = tars[0], tars[1:config.slot_size + 1]

                    # write da_type information
                    da_index = np.argmax(da_tar)
                    da_index_prob = np.argmax(prob[i][j][0])
                    test_file.write(config.da_types[da_index] + ',' + config.da_types[da_index_prob] + '\n')

                    # calculate datype acc
                    if da_index_prob != da_index:
                        dt_session_right = False
                        sentence_right = False
                    else:
                        dt_rate[0] += 1.0
                    dt_rate[1] += 1.0

                    # mask and slot_value
                    for m in range(config.slot_size):
                        slot = config.slots[m]
                        test_file.write('[{}],'.format(slot))

                        # write mask information
                        test_file.write('{:.3f},'.format(slot_mask[i][j][m][0]))
                        test_file.write('{:.3f},'.format(mask_result[i][j][m][0]))

                        # calculate mask acc
                        smm = np.argmax(slot_mask[i][j][m])
                        mrm = np.argmax(mask_result[i][j][m])
                        if mrm != smm:
                            mask_session_right = False
                            sentence_right = False
                        else:
                            mask_rate[0] += 1.0
                        mask_rate[1] += 1.0

                        # write slot value
                        value_index = np.argmax(sv_tar[m])
                        value_index_prob = np.argmax(slot_prob[i][j][m][0])

                        test_file.write(config.slot_value[slot][value_index] + ',')
                        test_file.write(config.slot_value[slot][value_index_prob] + '\n')

                        # calculate slotvalue acc
                        if slot_mask[i][j][m][0] > 0.9:
                            if value_index != value_index_prob:
                                sv_session_right = False
                                sentence_right = False
                            else:
                                sv_rate[0] += 1.0
                            sv_rate[1] += 1.0

                    # calculate sentence acc
                    if sentence_right:
                        rate[0] += 1.0
                    rate[1] += 1.0

                if dt_session_right:
                    dt_rate_session[0] += 1.0
                dt_rate_session[1] += 1.0

                if sv_session_right:
                    sv_rate_session[0] += 1.0
                sv_rate_session[1] += 1.0

                if mask_session_right:
                    mask_rate_session[0] += 1.0
                mask_rate_session[1] += 1.0

                if dt_session_right and sv_session_right and mask_session_right:
                    rate_session[0] += 1.0
                rate_session[1] += 1.0

    test_file.close()

    test_file = test_dir + _time + '_final.csv'

    with open(test_file, 'w+') as fout:
        fout.write(', sentence, session\n')
        fout.write('da type, {}, {}\n'.format(dt_rate[0] / dt_rate[1], dt_rate_session[0] / dt_rate_session[1]))
        fout.write('slot value, {}, {}\n'.format(sv_rate[0] / sv_rate[1], sv_rate_session[0] / sv_rate_session[1]))
        fout.write('mask, {}, {}\n'.format(mask_rate[0] / mask_rate[1], mask_rate_session[0] / mask_rate_session[1]))
        fout.write('full, {}, {}\n'.format(rate[0] / rate[1], rate_session[0] / rate_session[1]))
        fout.close()
    print('final result writt to {}'.format(test_file))
    return

if __name__ == '__main__':
    main()
