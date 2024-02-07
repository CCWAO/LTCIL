import importlib
import time
import os
from data_process.task_data_prepare import *

import utils
from loggers.exp_logger import MultiLogger
from functools import reduce
import argparse
from last_layer_analysis import last_layer_analysis, cf_plot

import torch

argv = None
tstart = time.time()
# Arguments
parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

# miscellaneous args
parser.add_argument('--gpu', type=int, default=0, help='GPU (default=%(default)s)')
parser.add_argument('--results-path', type=str, default='/home/chendongyue/data/CIL_graph/results',
                    help='Results path (default=%(default)s)')
parser.add_argument('--exp-name', default=None, type=str, help='Experiment name (default=%(default)s)')
parser.add_argument('--seed', type=int, default=12, help='4234,8627,3469,5712,9781')
parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                    help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
parser.add_argument('--save-models', action='store_true', help='Save trained models (default=%(default)s)')
parser.add_argument('--last-layer-analysis', action='store_true', help='Plot last layer analysis (default=%(default)s)')
parser.add_argument('--no-cudnn-deterministic', action='store_true',
                    help='Disable CUDNN deterministic (default=%(default)s)')
# dataset args
parser.add_argument('--datasets', type=str, default='swat',
                    choices=('ps', 'swat'))  # PTC_MR #NCI1, PROTEINS, COLLAB, and RDT-B.
parser.add_argument('--num-workers', default=0, type=int, required=False,
                    help='Number of subprocesses to use for dataloader (default=%(default)s)')
parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                    help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
parser.add_argument('--batch-size', default=40, type=int, required=False,
                    help='Number of samples per batch to load (default=%(default)s)')
parser.add_argument('--num-tasks', default=5, type=int, required=False,
                    help='Number of tasks per dataset (default=%(default)s)')
parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                    help='Number of classes of the first task (default=%(default)s)')
parser.add_argument('--use-valid-only', action='store_true',
                    help='Use validation split instead of test (default=%(default)s)')
parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                    help='Stop training after specified task (default=%(default)s)')
                    
                    
# model args
parser.add_argument('--model_name', default='GCNBB',
                    help='model name')
parser.add_argument('--num_features', default=5, type=int,
                    help='node feature size')
parser.add_argument('--num_classes', default=37, type=int,
                    help='total number of class')
#parser.add_argument('--in_channels', type=int, default=1)
#parser.add_argument('--var_weight', type=float, default=0.01)
parser.add_argument('--drop_rate', type=float, default=0.2)
parser.add_argument('--nhid', type=int, default=256)
#parser.add_argument('--num_layers', type=int, default=2)
# parser.add_argument('--tempreture', type=float, default=0.8)
parser.add_argument('--keep_existing_head', action='store_true',
                    help='Disable removing classifier last layer (default=%(default)s)')
parser.add_argument('--num_edge_preseve', type=int, default=3)
#parser.add_argument('--bias_scaler', default=0.9, type=float, help='Minimum learning rate (default=%(default)s)')
        

# training args
parser.add_argument('--approach', default='debias_new', type=str,
                    help='Learning approach used (default=%(default)s)', metavar="APPROACH")
parser.add_argument('--whether_loss_weight', default=False, type=bool,
                    help='Learning approach used (default=%(default)s)', metavar="APPROACH")
parser.add_argument('--nepochs', default=100, type=int, required=False,
                    help='Number of epochs per training session (default=%(default)s)')
parser.add_argument('--lr', default=0.001, type=float, required=False,
                    help='Starting learning rate (default=%(default)s)')
parser.add_argument('--lr-min', default=1e-5, type=float, required=False,
                    help='Minimum learning rate (default=%(default)s)')
parser.add_argument('--lr-factor', default=2, type=float, required=False,
                    help='Learning rate decreasing factor (default=%(default)s)')
parser.add_argument('--lr-patience', default=20, type=int, required=False,
                    help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
parser.add_argument('--clipping', default=10, type=float, required=False,
                    help='Clip gradient norm (default=%(default)s)')
parser.add_argument('--momentum', default=0.0, type=float, required=False,
                    help='Momentum factor (default=%(default)s)')
parser.add_argument('--weight-decay', default=0.0001, type=float, required=False,
                    help='Weight decay (L2 penalty) (default=%(default)s)')
parser.add_argument('--warmup-nepochs', default=0, type=int, required=False,
                    help='Number of warm-up epochs (default=%(default)s)')
parser.add_argument('--warmup-lr-factor', default=0.6, type=float, required=False,
                    help='Warm-up learning rate factor (default=%(default)s)')
parser.add_argument('--multi-softmax', action='store_true',
                    help='Apply separate softmax for each task (default=%(default)s)')
parser.add_argument('--fix-bn', action='store_true',
                    help='Fix batch normalization after first task (default=%(default)s)')
parser.add_argument('--eval-on-train', action='store_true',
                    help='Show train loss and accuracy (default=%(default)s)')
parser.add_argument('--class_order_mode', default='no', type=str, help='class order mode',
                    choices=('m2l', 'l2m', 'random1', 'random2', 'random3', 'no'))
parser.add_argument('--loss_normalize', default='softmax', type=str, help='how to normalize the loss',
                    choices=('softmax', 'sparsemax'))
#parser.add_argument('--num-exemplars-per-class', default=20, type=int, required=False,
#                    help='Growing memory, number of exemplars per class (default=%(default)s)')
# gridsearch args
parser.add_argument('--gridsearch-tasks', default=-1, type=int,
                    help='Number of tasks to apply GridSearch (-1: all tasks) (default=%(default)s)')

# Args -- Incremental Learning Framework
args, extra_args = parser.parse_known_args(argv)
args.results_path = os.path.expanduser(args.results_path)
base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                   lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                   wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                   wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train)

print('=' * 108)
print('Arguments =')
for arg in np.sort(list(vars(args).keys())):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 108)

device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
utils.seed_everything(seed=args.seed)
init_net = getattr(importlib.import_module(name='models'), args.model_name)
init_model = init_net(args)

init_net_b = getattr(importlib.import_module(name='models'), args.model_name)
init_model_b = init_net_b(args)

# ICL apporch initialized
from approach.debias_new import CIL_debias_new

ICL_appr = CIL_debias_new
appr_args, extra_args = ICL_appr.extra_parser(extra_args)
print('Approach arguments =')
for arg in np.sort(list(vars(appr_args).keys())):
    print('\t' + arg + ':', getattr(appr_args, arg))
print('=' * 108)

# Exemplars Management
from data_process.Exemplar_dataset import ExemplarsSet

ICL_ExemplarsDataset = ICL_appr.exemplars_dataset_class()
if ICL_ExemplarsDataset:
    assert issubclass(ICL_ExemplarsDataset, ExemplarsSet)
    appr_exemplars_dataset_args, extrda_args = ICL_ExemplarsDataset.extra_parser(extra_args)
    print('Exemplars dataset arguments =')
    for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
        print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
    print('=' * 108)
else:
    appr_exemplars_dataset_args = argparse.Namespace()

assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))

# Args -- GridSearch
if args.gridsearch_tasks > 0:
    from gridsearch import GridSearch
    gs_args, extra_args = GridSearch.extra_parser(extra_args)
    Appr_finetuning = CIL_debias_new
    # assert issubclass(Appr_finetuning, Inc_Learning_Appr)
    GridSearch_ExemplarsDataset = ICL_appr.exemplars_dataset_class()
    print('GridSearch arguments =')
    for arg in np.sort(list(vars(gs_args).keys())):
        print('\t' + arg + ':', getattr(gs_args, arg))
    print('=' * 108)

assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))


# logger
full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
full_exp_name += '_' + args.approach + '/' + args.model_name + '_' + args.class_order_mode
if args.exp_name is not None:
    full_exp_name += '_' + args.exp_name + '/' + args.model_name+ '_' + args.class_order_mode
logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)
logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))

# Load raw data and dataloader transform

train_set = torch.load('/swat_fc_train.pt')
test_set = torch.load('/swat_fc_test.pt')
valid_set = torch.load('/swat_fc_test.pt')


print('num_train', len(train_set))
print('num_test', len(test_set))


class_order = utils.get_class_order(args.class_order_mode)
print('class_order', class_order)

trn_loader, val_loader, tst_loader, taskcla = get_dataloader(args, train_set, test_set, valid_set,
                                                             class_order=class_order, shuffle_classes=0)

# Apply arguments for loaders
if args.use_valid_only:
    tst_loader = val_loader
max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

# Network and Approach instances
from network_cil import NET_ICL

net = NET_ICL(init_model, remove_existing_head=not args.keep_existing_head)
net_b = NET_ICL(init_model_b, remove_existing_head=not args.keep_existing_head)
appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
if ICL_ExemplarsDataset:
    appr_kwargs['exemplars'] = ExemplarsSet(**appr_exemplars_dataset_args.__dict__)

# ICL instance
ICL = ICL_appr(net, net_b, device, **appr_kwargs)



# GridSearch
if args.gridsearch_tasks > 0:
    ft_kwargs = {**base_kwargs, **dict(logger=logger,
                                       exemplars=GridSearch_ExemplarsDataset)}
    appr_ft = Appr_finetuning(net, net, device, **ft_kwargs)
    gridsearch = GridSearch(appr_ft, args.seed, gs_args.gridsearch_config, gs_args.gridsearch_acc_drop_thr,
                            gs_args.gridsearch_hparam_decay, gs_args.gridsearch_max_num_searches)

# Loop tasks
print(taskcla)
acc_taw = np.zeros((max_task, max_task))
acc_tag = np.zeros((max_task, max_task))
acc_taw_macro = np.zeros((max_task, max_task))
acc_tag_macro = np.zeros((max_task, max_task))

forg_taw = np.zeros((max_task, max_task))
forg_tag = np.zeros((max_task, max_task))
forg_taw_macro = np.zeros((max_task, max_task))
forg_tag_macro = np.zeros((max_task, max_task))

pred_debias_taw = []
pred_debias_tag = []
yy_debias = []

output_of_all= []
yy_of_all = []

for t, (_, ncla) in enumerate(taskcla):
    # Early stop tasks if flag
    print(t)
    if t >= max_task:
        continue

    print('*' * 108)
    print('Task {:2d}'.format(t))
    print('*' * 108)

    net.add_head(taskcla[t][1])
    net.to(device)
    
    net_b.add_head(taskcla[t][1])
    net_b.to(device)
    
    # Train
    print('main task id', t)
    ICL.train(t, trn_loader[t], val_loader[t])
    print('-' * 108)

    # Test
    output_of_task = []
    yy_of_task = []
    for u in range(t + 1):
        test_loss, acc_taw[t, u], acc_tag[t, u], acc_taw_macro[t,u], acc_tag_macro[t,u], _, xx_c, xx_y = ICL.eval(u, tst_loader[u])
        output_of_task.append(xx_c)
        yy_of_task.append(xx_y)

        if u < t:
            forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
            forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            forg_taw_macro[t, u] = acc_taw_macro[:t, u].max(0) - acc_taw_macro[t, u]
            forg_tag_macro[t, u] = acc_tag_macro[:t, u].max(0) - acc_tag_macro[t, u]
        print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:.4f}%, forg={:.4f}%'
              '| TAg acc={:.4f}%, forg={:.4f}% | TAw macro ={:.4f}%, forg_macro={:.4f}% | TAg macro={:.4f}%, forg macro={:.4f}% <<<'.format(
            u, test_loss,
            100 * acc_taw[t, u], 100 * forg_taw[t, u],
            100 * acc_tag[t, u], 100 * forg_tag[t, u],
            100 * acc_taw_macro[t, u], 100 * forg_taw_macro[t, u],
            100 * acc_tag_macro[t, u], 100 * forg_tag_macro[t, u]))
        
    output_of_all.append(output_of_task)
    yy_of_all.append(yy_of_task)


    logger.log_scalar(task=t, iter=u, name='loss', group='test', value=test_loss)
    logger.log_scalar(task=t, iter=u, name='acc_taw', group='test', value=100 * acc_taw[t, u])
    logger.log_scalar(task=t, iter=u, name='acc_tag', group='test', value=100 * acc_tag[t, u])
    logger.log_scalar(task=t, iter=u, name='forg_taw', group='test', value=100 * forg_taw[t, u])
    logger.log_scalar(task=t, iter=u, name='forg_tag', group='test', value=100 * forg_tag[t, u])
    logger.log_scalar(task=t, iter=u, name='acc_taw_macro', group='test', value=100 * acc_taw_macro[t, u])
    logger.log_scalar(task=t, iter=u, name='acc_tag_macro', group='test', value=100 * acc_tag_macro[t, u])
    logger.log_scalar(task=t, iter=u, name='forg_taw_macro', group='test', value=100 * forg_taw_macro[t, u])
    logger.log_scalar(task=t, iter=u, name='forg_tag_macro', group='test', value=100 * forg_tag_macro[t, u])

    # Save
    print('Save at ' + os.path.join(args.results_path, full_exp_name))
    logger.log_result(acc_taw, name="acc_taw", step=t)
    logger.log_result(acc_tag, name="acc_tag", step=t)
    logger.log_result(forg_taw, name="forg_taw", step=t)
    logger.log_result(forg_tag, name="forg_tag", step=t)
    logger.log_result(acc_taw_macro, name="acc_taw_macro", step=t)
    logger.log_result(acc_tag_macro, name="acc_tag_macro", step=t)
    logger.log_result(forg_taw_macro, name="forg_taw_macro", step=t)
    logger.log_result(forg_tag_macro, name="forg_tag_macro", step=t)

    #    logger.save_model(net, task=t)
    torch.save(net.state_dict(), os.path.join(logger.exp_path, "models", "task{}.pkl".format(t)))
    
    logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1), name="avg_accs_taw", step=t)
    logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1), name="avg_accs_tag", step=t)
    aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
    logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), name="wavg_accs_taw", step=t)
    logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), name="wavg_accs_tag", step=t)
    
    logger.log_result(acc_taw_macro.sum(1) / np.tril(np.ones(acc_taw_macro.shape[0])).sum(1), name="avg_taw_macro", step=t)
    logger.log_result(acc_tag_macro.sum(1) / np.tril(np.ones(acc_tag_macro.shape[0])).sum(1), name="avg_tag_macro", step=t)
    
    logger.log_result((acc_taw_macro * aux).sum(1) / aux.sum(1), name="wavg_taw_macro", step=t)
    logger.log_result((acc_tag_macro * aux).sum(1) / aux.sum(1), name="wavg_tag_macro", step=t)

    weights, biases = last_layer_analysis(net.heads_c, t, taskcla, y_lim=False)
    logger.log_figure(name='weights', iter=t, figure=weights)
    logger.log_figure(name='bias', iter=t, figure=biases)

    # Output sorted weights and biases
    weights, biases = last_layer_analysis(net.heads_c, t, taskcla, y_lim=False, sort_weights=True)
    logger.log_figure(name='weights_sort', iter=t, figure=weights)
    logger.log_figure(name='bias_sort', iter=t, figure=biases)
        
#confusionmatrix

f_cf = cf_plot(tst_loader, net, args.num_classes, device)
logger.log_figure(name='cf', iter=t, figure=f_cf)

# Print Summary
torch.save(output_of_all, os.path.join(logger.exp_path, "models", "output_of_all.pt"))
torch.save(yy_of_all, os.path.join(logger.exp_path, "models", "yy_of_all.pt"))
utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag, acc_taw_macro, acc_tag_macro, forg_taw_macro, forg_tag_macro)
print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
print(logger.exp_path)
print('Done!')

