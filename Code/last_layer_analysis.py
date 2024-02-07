import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score

matplotlib.use('Agg')

def last_layer_analysis(heads, task, taskcla, y_lim=False, sort_weights=False):
    """Plot last layer weight and bias analysis"""
    print('Plotting last layer analysis...')
    num_classes = sum([x for (_, x) in taskcla])
    weights, biases, indexes = [], [], []
    class_id = 0
    with torch.no_grad():
        for t in range(task + 1):
            n_classes_t = taskcla[t][1]
            indexes.append(np.arange(class_id, class_id + n_classes_t))
            if type(heads) == torch.nn.Linear:  # Single head
                biases.append(heads.bias[class_id: class_id + n_classes_t].detach().cpu().numpy())
                weights.append((heads.weight[class_id: class_id + n_classes_t] ** 2).sum(1).sqrt().detach().cpu().numpy())
            else:  # Multi-head
                weights.append((heads[t].weight ** 2).sum(1).sqrt().detach().cpu().numpy())
                if type(heads[t]) == torch.nn.Linear:
                    biases.append(heads[t].bias.detach().cpu().numpy())
                else:
                    biases.append(np.zeros(weights[-1].shape))  # For LUCIR
            class_id += n_classes_t

    # Figure weights
    f_weights = plt.figure(dpi=300)
    ax = f_weights.subplots(nrows=1, ncols=1)
    for i, (x, y) in enumerate(zip(indexes, weights), 0):
        if sort_weights:
            ax.bar(x, sorted(y, reverse=True), label="Task {}".format(i))
        else:
            ax.bar(x, y, label="Task {}".format(i))
    ax.set_xlabel("Classes", fontsize=11, fontfamily='serif')
    ax.set_ylabel("Weights L2-norm", fontsize=11, fontfamily='serif')
    if num_classes is not None:
        ax.set_xlim(0, num_classes)
    if y_lim:
        ax.set_ylim(0, 5)
    ax.legend(loc='upper left', fontsize='11') #, fontfamily='serif')

    # Figure biases
    f_biases = plt.figure(dpi=300)
    ax = f_biases.subplots(nrows=1, ncols=1)
    for i, (x, y) in enumerate(zip(indexes, biases), 0):
        if sort_weights:
            ax.bar(x, sorted(y, reverse=True), label="Task {}".format(i))
        else:
            ax.bar(x, y, label="Task {}".format(i))
    ax.set_xlabel("Classes", fontsize=11, fontfamily='serif')
    ax.set_ylabel("Bias values", fontsize=11, fontfamily='serif')
    if num_classes is not None:
        ax.set_xlim(0, num_classes)
    if y_lim:
        ax.set_ylim(-1.0, 1.0)
    ax.legend(loc='upper left', fontsize='11') #, fontfamily='serif')

    return f_weights, f_biases


def cf_plot(test_loader, net, num_classes, device):
    
    test_set = []
    for i in range(len(test_loader)):
        test_set = test_set + test_loader[i].dataset

    test_all_loader = DataLoader(test_set, batch_size=100, shuffle=True)
    preddd = torch.tensor([0])
    yy = torch.tensor([0])
    for idx, data in enumerate(test_all_loader):
        data = data.to(device)
        outputs, _ = net(data)
        pred = torch.cat(outputs, dim=1).argmax(1)
        pred = pred.view(-1)
        preddd = torch.cat([preddd, pred.cpu()])
        yy = torch.cat([yy, data.y.cpu()])

    preddd = preddd[1::]
    yy = yy[1::]
    micro_f1 = f1_score(preddd, yy, average='micro')
    macro_f1 = f1_score(preddd, yy, average='macro')
    print('micro f1', micro_f1)
    print('macro f1', macro_f1)
    
    print('preddd',preddd.size())
    C2 = confusion_matrix(yy + 1, preddd + 1)
    ss = []
    for i in range(num_classes):
        a = (yy == i).sum()
        ss.append(a)
    ss = np.array(ss)
    C2 = torch.tensor(C2)
    C2 = C2 / ss
    C2 = np.around(C2, 2)

    f_cf = plt.figure(dpi=300)
    ax = f_cf.subplots(nrows=1, ncols=1)
    ax.set_xlabel('Prediction', fontsize=11, fontfamily='serif')
    ax.set_ylabel('Real label', fontsize=11, fontfamily='serif')
    ax.imshow(C2, interpolation='nearest', cmap=plt.cm.Blues)
#    ax.colorbar(fraction=0.045)

    return f_cf
