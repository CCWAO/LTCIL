from torch_geometric.data import Dataset
import random
import numpy as np
from torch_geometric.data import Batch
from argparse import ArgumentParser
import importlib

class ExemplarsSet(Dataset):
    def __init__(self, transform=None, pre_transform=None, pre_filter=None,
                 num_exemplars=0, num_exemplars_per_class=0):
        super().__init__(transform, pre_transform, pre_filter)
        self.max_num_exemplars_per_class = num_exemplars_per_class
        self.max_num_exemplars = num_exemplars
        self.exemplars_dataset = []
        assert (num_exemplars_per_class == 0) or (num_exemplars == 0), 'Cannot use both limits at once!'

    def len(self):
        return len(self.exemplars_dataset)

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser("Exemplars Management Parameters")
        _group = parser.add_mutually_exclusive_group()
        _group.add_argument('--num-exemplars', default=0, type=int, required=False,
                            help='Fixed memory, total number of exemplars (default=%(default)s)')
        _group.add_argument('--num-exemplars-per-class', default=10, type=int, required=False,
                            help='Growing memory, number of exemplars per class (default=%(default)s)')
        # parser.add_argument('--exemplar-selection', default='random', type=str,
        #                     choices=['herding', 'random', 'entropy', 'distance'],
        #                     required=False, help='Exemplar selection strategy (default=%(default)s)')
        return parser.parse_known_args(args)

    def collect_exemplars(self, model, trn_loader, exemplars_per_class):
#        print('exempalars_per_class', exemplars_per_class)

        data = trn_loader.dataset
        num_cls = sum(model.task_cls)
        graph_batch = Batch.from_data_list(trn_loader.dataset)
        labels = graph_batch.y
        selected_indices = []
        for curr_cls in range(num_cls):
            # get all indices from current class -- check if there are exemplars from previous task in the loader
            cls_ind = np.where(labels == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            selected_indices.append(random.sample(list(cls_ind), exemplars_per_class))

        selected_indices = np.array(selected_indices)
        selected_indices = selected_indices.reshape(-1)
        self.exemplars_dataset = [data[idx] for idx in selected_indices]

        return self.exemplars_dataset

