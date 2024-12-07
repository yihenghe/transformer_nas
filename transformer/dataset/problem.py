
def prepare(problem_set, data_dir, max_length, batch_size, device, opt, test_flag=False):
    if problem_set not in ['wmt32k', 'lm1b']:
        raise Exception("only ['wmt32k', 'lm1b'] problem set supported.")

    setattr(opt, 'has_inputs', True)

    if problem_set == 'wmt32k':
        from dataset import translation
        prepare = translation.prepare_bpe if opt.use_bpe else translation.prepare
        train_iter, dev_iter, opt = \
            prepare(max_length, batch_size, device, opt, data_dir, test_flag)
    elif problem_set == 'lm1b':
        from dataset import lm
        train_iter, dev_iter, opt = \
            lm.prepare(max_length, batch_size, device, opt, data_dir, test_flag)

    return train_iter, dev_iter, opt.src_vocab_size, opt.trg_vocab_size, opt
