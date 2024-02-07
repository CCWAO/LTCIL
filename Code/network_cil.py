import torch
import torch.nn as nn
from copy import deepcopy

class NET_ICL(torch.nn.Module):
    def __init__(self, model, remove_existing_head=False):
        head_var_c = model.head_var_c
        head_var_b = model.head_var_b
        assert type(head_var_c) == str
        assert not remove_existing_head or hasattr(model, head_var_c), \
            "Given model does not have a variable called {}".format(head_var_c)
        assert not remove_existing_head or type(getattr(model, head_var_c)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var_c)
        super(NET_ICL, self).__init__()
        self.model = model
        last_layer_c = getattr(self.model, head_var_c)
        last_layer_b = getattr(self.model, head_var_b)
        # print('NET_ICL remove_exsiting_head', remove_existing_head)
        if remove_existing_head:
            if type(last_layer_c) == nn.Sequential:
                self.out_size = last_layer_c[-1].in_features
                # strips off last linear layer of classifier
                del last_layer_c[-1]
                del last_layer_b[-1]
            elif type(last_layer_c) == nn.Linear:
                self.out_size = last_layer_c.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var_c, nn.Sequential())
                setattr(self.model, head_var_b, nn.Sequential())
        else:
            self.out_size = last_layer_c.out_features

        self.heads_c = nn.ModuleList()
        self.heads_b = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads_c.append(nn.Linear(self.out_size, num_outputs))
        self.heads_b.append(nn.Linear(self.out_size, num_outputs))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads_c])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        x_c, x_b = self.model(x)
        assert (len(self.heads_c) > 0), "Cannot access any head"
        y_c = []
        for head in self.heads_c:
            y_c.append(head(x_c))
        y_b = []
        for head in self.heads_b:
            y_b.append(head(x_b))
        if return_features:
            return y_c, y_b, x_c, x_b
        else:
            return y_c, y_b

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass