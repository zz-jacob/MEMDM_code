import numpy as np
import tensorflow as tf
from config import Config
from data_utils import DataLoader
from memdm import Model
import time
import datetime
import os
import shutil


def main():
    """ config some args
    """
    print('start training ...')
    config = Config()
    config.save_path = config.save_path + str(config.task_id) + '/'
    config.data_dir = 'data/dialog-bAbI-tasks/'
    config.target_file = 'trn'
    loader = DataLoader(config)
    print('\nconfig vocab_size: {}, batch_num: {}, seq_length: {}'.format(
            config.vocab_size, config.batch_num, config.seq_length))
    config.create_properties()
    train(config, loader)


def train(args, loader):
    """
    train functions
    Args:
        args: the super-params of
        loader: DataLoader, a instance of DataLoader
    Returns:
        NON
    """

    print('\nstart building models ...')
    start = time.time()
    model = Model(args)
    print('... finished!')
    print('building model time: {:.3f}'.format(time.time() - start))

    _config = tf.ConfigProto()
    _config.gpu_options.allow_growth = True
    if not args.device == '/cpu:0':
        _config.allow_soft_placement = True

    start = time.time()
    with tf.Session(config=_config) as sess:

        # setup log file name with system time
        start_time = str(datetime.datetime.now())
        start_time = start_time.replace(':', '_')
        start_time = start_time.replace(' ', '_')
        start_time = start_time.replace('.', '_')
        logfile = open(args.log_path + start_time, 'w+')

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # setup tensorflow.save
        saver = tf.train.Saver()

        # setup summary
        merged = tf.merge_all_summaries()
        sum_writer = tf.train.SummaryWriter(args.summary_path + 'train.sum', sess.graph)

        # stt = True
        # eq, ea = [], []
        fwrite = open('train{}_log.txt'.format(args.task_id), 'w+')
        # loss_mask = [0.0, 0.0, 1.0, 0.0, 0.0]
        loss_mask = [1.0, 0.0, 0.0, 1.0, 0.0]
        print('\n start session time: {:.3f}'.format(time.time() - start))

        sess.run(tf.assign(model.embedding, args.embeddings))
        sess.run(tf.assign(model.init_k_memory, args.slot_embedding))
        sess.run(tf.assign(model.value_feature0, loader.value_feature))

        ckpt = tf.train.get_checkpoint_state(args.save_path)
        if ckpt and ckpt.model_checkpoint_path and args.restore:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('\nmodel restored from {}'.format(ckpt.model_checkpoint_path))

        for e in range(args.epoch_num):  # run epoch
            if e > 7:
                loss_mask = [1.0, 0.0, 2.0, 1.0, 0.0]
            if e > 9:
                loss_mask = [1.0, 0.0, 1.0, 1.0, 2.0]
            if e > 11:
                loss_mask = [1.0, 2.0, 1.0, 1.0, 1.0]

            print('\ncreating batches for epoch[{}] ...'.format(e))
            loader.is_shuffle = True
            loader.create_batches()
            print('... finished!')
            # early stop
            is_break = False

            # assign learning rate
            sess.run(tf.assign(model.lr, args.learning_rate*(args.decay_rate ** e)))

            for b in range(args.batch_num):  # run batch
                start = time.time()
                index, x, y, t, m, seq_lengths_mask, x_sen_len, y_sen_len, attn, attn_mask, \
                    raw_x, raw_y, _, _ = loader.next_batch()
                out0 = 'task{} run batch {}/{} in epoch {}'.format(args.task_id, b, args.batch_num, e)
                print(out0)

                feed = {model.input_x: x, model.input_y: y,
                        model.targets: t, model.masks: m, model.seq_masks: seq_lengths_mask,
                        model.x_sentence_length: x_sen_len, model.y_sentence_length: y_sen_len,
                        model.attention: attn, model.attention_mask: attn_mask,
                        model.loss_mask: loss_mask}

                # if stt:
                # if True:
                #     eq = args.embeddings
                #     ea = args.embeddings
                #     stt = False
                # assign embeddings
                # sess.run(tf.assign(model.embedding_q, eq))
                # sess.run(tf.assign(model.embedding_a, ea))

                fetch = [merged, model.g_val, model.tmp_val, model.attention_result, model.cost, model.cost1, model.cost2,
                         model.cost3, model.cost4, model.cost5, model.final_state, model.embedding_q,
                         model.embedding_a, model.mask_result, model.train_op]
                # if e <= 9:
                #     fetch.append(model.train_op)
                # else:
                #     fetch.append(model.train_op2)

                # session run
                mgd, g_val, tmp_val, attention_result, train_loss, loss1, loss2, loss3, loss4, loss5, state, eq, ea, mask_result, _ = sess.run(
                    fetch, feed)

                end = time.time()

                # if loss3 < 0.01:
                #     loss_mask[1] = 1.0
                #     loss_mask[3] = 1.0
                #     loss_mask[4] = 1.0

                # check tmp_values (attention result output to config.attnlog_path)
                filename = args.attnlog_path + 'attn_' + str(e) + '_' + str(b) + '.txt'
                with open(filename, 'w+') as fout:
                    for i in range(args.batch_size):
                        for j in range(args.seq_length):
                            # sentence
                            fout.write(str(raw_x[i][j]) + '\n')
                            # attention result
                            for m in range(args.slot_size):
                                fout.write('[' + args.slots[m] + ']\t')
                                fout.write('mask[{:.3f}]'.format(mask_result[i][j][m][0]))
                                fout.write('g:[{}]\t'.format(str(g_val[i][j][m][0])))
                                for k in range(args.sentence_length):
                                    fout.write(str(tmp_val[i][j][m][k]) + '\t')
                                # fout.write(str(tmp_val[i][j][m]) + ' ')
                                fout.write('\n')
                    fout.close()

                # write to summary
                sum_writer.add_summary(mgd, e * args.epoch_num + b)

                end2 = time.time()

                out1 = "\ttrain_loss = {:.3f}, time/batch = {}, write time = {}".format(
                        train_loss, end - start, end2 - end)
                out2 = "\tloss1: {:.3f}; loss2: {:.3f}; loss3: {:.3f}; loss4: {:.3f}; loss5: {:.3f}".format(
                        loss1, loss2, loss3, loss4, loss5)
                print(out1)
                print(out2)

                fwrite.write(out0 + '\n')
                fwrite.write(out1 + '\n')
                fwrite.write(out2 + '\n')

                if (e * args.batch_num + b) % args.save_step == 0:
                    checkpoint_path = os.path.join(args.save_path, 'model.ckpt')
                    save_path = saver.save(sess, checkpoint_path,
                                           write_meta_graph=False, global_step=e * args.epoch_num + b)
                    print('\t (SAVE) model saved to {}'.format(save_path))

            if is_break:
                break
        fwrite.close()
        logfile.close()


def clean_dir(save_path, task_id):
    """
    clean save/task_id dir
    Args:
        save_path: str
        task_id: str, id
    Return:
        None
    """
    shutil.rmtree(save_path + str(task_id))
    os.mkdir(save_path + str(task_id))


if __name__ == '__main__':
    main()
