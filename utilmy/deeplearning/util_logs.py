#### utilities for deep learning training




def tensorboard_log(pars_dict:dict=None,  writer=None,  verbose=True):
    """
    #### Usage 1 
    logdir = 'logs/params'

    from tensorboardX import SummaryWriter
    # from tensorboard import SummaryWriter
    tb_writer = SummaryWriter(logdir)
    tensorboard_log(cc, writer= tb_writer)

    %reload_ext tensorboard
    %tensorboard --logdir logs/params/
    """
    import collections
    def dict_flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(dict_flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


    flatten_box = dict_flatten(pars_dict)
    if verbose:
        print(flatten_box)


    for k, v in flatten_box.items():            
        if isinstance(v, (int, float)):
            writer.add_scalar(str(k), v, 0)
        else :
            writer.add_text(str(k), str(v), 0)

    writer.close()
    return writer












