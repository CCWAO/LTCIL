import numpy as np
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader


def get_data(config, data_set, set_name, cpertask_cumsum, class_order):

    graph_out = {}
    set_batch = Batch.from_data_list(data_set)
    print('label', set_batch.y.size())

    for tt in range(config.num_tasks):
        graph_out[tt] = {}
        graph_out[tt]['name'] = 'task-' + str(tt)
        graph_out[tt][set_name] = []

    for g in data_set:
        this_label = class_order.index(g.y)
        this_task = (this_label >= cpertask_cumsum).sum()
        # new_label = this_label - init_class[this_task]
        new_label = this_label
        old_label = g.y
        g.old_label = old_label
        g.y = new_label
        graph_out[this_task][set_name].append(g)

    return graph_out


def get_dataloader(config, train_set, test_set, valid_set, class_order, shuffle_classes=0):

    if class_order is None:
        class_order = list(range(config.num_classes))
    else:
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if config.nc_first_task is None:
        cpertask = np.array([config.num_classes // config.num_tasks] * config.num_tasks)
        for i in range(config.num_classes % config.num_tasks):
            cpertask[i] += 1  # compute classes per task and num_tasks
    else:
        assert config.nc_first_task < config.num_classes, "first task wants more classes than exist"
        remaining_classes = config.num_classes - config.nc_first_task
        assert remaining_classes >= (config.num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([config.nc_first_task] + [remaining_classes // (config.num_tasks - 1)] * (config.num_tasks - 1))
        for i in range(remaining_classes % (config.num_tasks - 1)):
            cpertask[i + 1] += 1

    assert config.num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    print(cpertask_cumsum)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))
    print(init_class)


    data_train = get_data(config, train_set, set_name='trn', cpertask_cumsum=cpertask_cumsum, class_order=class_order)
    data_test = get_data(config, test_set, set_name='tst', cpertask_cumsum=cpertask_cumsum, class_order=class_order)
    data_valid = get_data(config, valid_set, set_name='val', cpertask_cumsum=cpertask_cumsum, class_order=class_order)

    taskcla = []
    for tt in range(config.num_tasks):
        train_set_batch = Batch.from_data_list(data_train[tt]['trn'])
        ncla = len(np.unique(train_set_batch.y))
        taskcla.append((tt, cpertask[tt]))
        assert ncla == cpertask[tt], "something went wrong splitting classes"

    trn_load, val_load, tst_load = [], [], []
    for tt in range(config.num_tasks):
        trn_load.append(DataLoader(data_train[tt]['trn'], batch_size=config.batch_size, shuffle=True))
        val_load.append(DataLoader(data_test[tt]['tst'], batch_size=config.batch_size, shuffle=False))
        tst_load.append(DataLoader(data_valid[tt]['val'], batch_size=config.batch_size, shuffle=False))

    return trn_load, val_load, tst_load, taskcla




