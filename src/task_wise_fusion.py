# import logging
# import types
# import warnings
# from copy import deepcopy
# from typing import Any, Callable, Dict, Iterator, List

# import torch
# from torch import Tensor, nn
# from torch.func import functional_call

# from .ties_merging_utils import check_parameterNamesMatch
# from .type import StateDict
# from .utils import timeit_context

# log = logging.getLogger(__name__)

# __all__ = ["get_task_wise_weights", "fuse_weights", "TaskWiseMergedModel"]


# class NonLinearWeightGenerator(nn.Module):
#     def __init__(self, num_models: int, hidden_dim: int = 64):
#         """
#         Nonlinear weight generator that produces task-wise weights.
        
#         Args:
#             num_models (int): Number of tasks/models to fuse
#             hidden_dim (int): Hidden dimension of the neural network
#         """
#         super().__init__()
#         self.num_models = num_models
        
#         # Shared feature extractor
#         self.shared_net = nn.Sequential(
#             nn.Linear(num_models, hidden_dim),
#             nn.ReLU(),
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU()
#         )
        
#         # Task-specific weight generators
#         self.weight_generators = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, 1),
#                 nn.Sigmoid()  # Ensure weights are in [0,1]
#             ) for _ in range(num_models)
#         ])
        
#     def forward(self):
#         # Input is just a placeholder since we're generating weights
#         base_input = torch.eye(self.num_models, device=self.shared_net[0].weight.device)
        
#         # Shared features
#         shared_features = self.shared_net(base_input)
        
#         # Generate task-specific weights
#         weights = torch.cat([
#             generator(shared_features[i].unsqueeze(0)) 
#             for i, generator in enumerate(self.weight_generators)
#         ], dim=0)
        
#         return weights.squeeze(-1)


# def get_task_wise_weights(num_models: int, init_values: float = None, nonlinear: bool = False):
#     """
#     This function generates weights for each model, either linear or nonlinear.
    
#     Args:
#         num_models (int): The number of models.
#         init_values (float, optional): The initial value for each weight. Defaults to None.
#         nonlinear (bool): Whether to use nonlinear weight generator. Defaults to False.
        
#     Returns:
#         Tensor or NonLinearWeightGenerator: Weights for each model.
#     """
#     assert num_models >= 1, f"num_models must be >= 1, got {num_models}"
    
#     if nonlinear:
#         return NonLinearWeightGenerator(num_models)
#     else:
#         if init_values is None:
#             init_values = 1.0 / num_models
#         return torch.full((num_models,), init_values, dtype=torch.float32)


# def _fuse_weights(task_wise_weight: Tensor, tensors: List[Tensor]):
#     """
#     This function fuses the weights of the models.
    
#     Args:
#         task_wise_weight (Tensor): The weights for each model.
#         tensors (List[Tensor]): The list of tensors to be fused.
        
#     Returns:
#         Tensor: The fused weights.
#     """
#     if isinstance(task_wise_weight, NonLinearWeightGenerator):
#         weights = task_wise_weight()
#         device = weights.device
#     else:
#         weights = task_wise_weight
#         device = weights.device
        
#     return sum(weights[i] * w.to(device) for i, w in enumerate(tensors))


# def fuse_weights(task_wise_weight: Tensor, state_dicts: List[StateDict]) -> StateDict:
#     """
#     This function fuses the weights of the models and returns a state dictionary.
    
#     Args:
#         task_wise_weight (Tensor or NonLinearWeightGenerator): The weights for each model.
#         state_dicts (List[StateDict]): The list of state dictionaries. on cpu.
        
#     Returns:
#         StateDict: The fused state dictionary.
#     """
#     num_models = len(state_dicts)
    
#     if isinstance(task_wise_weight, NonLinearWeightGenerator):
#         assert num_models == task_wise_weight.num_models, (
#             f"num_models must be equal to the number of state_dicts, "
#             f"got {num_models} and {task_wise_weight.num_models}"
#         )
#     else:
#         assert task_wise_weight.dim() == 1, (
#             f"task_wise_weight must be a 1D tensor, got {task_wise_weight.dim()}"
#         )
#         assert num_models == task_wise_weight.size(0), (
#             f"num_models must be equal to the number of state_dicts, "
#             f"got {num_models} and {task_wise_weight.size(0)}"
#         )
    
#     return {k: _fuse_weights(task_wise_weight, [sd[k] for sd in state_dicts]) 
#             for k in state_dicts[0].keys()}


# class TaskWiseMergedModel(nn.Module):
#     def __init__(
#         self,
#         pretrained_model: nn.Module,
#         task_wise_weight: Tensor or NonLinearWeightGenerator,
#         task_vectors: List[StateDict],
#         clamp_weights: bool = True,
#     ):
#         """
#         Initialize the TaskWiseMergedModel.
        
#         Args:
#             pretrained_model (nn.Module): The base pretrained model.
#             task_wise_weight (Tensor or NonLinearWeightGenerator): The weights for each task.
#             task_vectors (List[StateDict]): List of task vectors to merge.
#             clamp_weights (bool): Whether to clamp weights between 0 and 1.
#         """
#         super().__init__()
#         self._model = (pretrained_model,)  # self._model should be on cpu

#         if isinstance(task_wise_weight, NonLinearWeightGenerator):
#             self.task_wise_weight = task_wise_weight
#             # For nonlinear weights, we need to track parameters differently
#             self._weight_parameters = list(task_wise_weight.parameters())
#         else:
#             self.task_wise_weight = nn.Parameter(task_wise_weight, requires_grad=True)
#             self._weight_parameters = [self.task_wise_weight]
            
#         self.task_vectors = task_vectors  # should be on cpu
#         self.pretrained_state_dict: StateDict = self.model.state_dict(keep_vars=False)
#         check_parameterNamesMatch(self.task_vectors)
#         self.clamp_weights = clamp_weights
#         self.merged_state_dict = None

#     @property
#     def model(self):
#         return self._model[0]

#     def merge_weights(self):
#         """Merge the weights using the current task-wise weights."""
#         if isinstance(self.task_wise_weight, NonLinearWeightGenerator):
#             weights = self.task_wise_weight()
#             if self.clamp_weights:
#                 weights = weights.clamp(0, 1)
#         else:
#             weights = self.task_wise_weight
#             if self.clamp_weights:
#                 weights = weights.clamp(0, 1)
                
#         device = weights.device
#         task_vector = fuse_weights(weights, self.task_vectors)
        
#         self.merged_state_dict = {
#             k: self.pretrained_state_dict[k].to(device, non_blocking=True) 
#             for k in self.pretrained_state_dict.keys()
#         }
        
#         for k in task_vector.keys():
#             self.merged_state_dict[k] += task_vector[k]

#     def forward(self, *args, **kwargs):
#         if self.merged_state_dict is None:
#             self.merge_weights()
#         return functional_call(
#             self.model,
#             self.merged_state_dict,
#             args=args,
#             kwargs=kwargs,
#             tie_weights=False,
#         )

#     def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
#         """Override parameters to include weight generator parameters."""
#         for param in self._weight_parameters:
#             yield param
            
#         for param in self.model.parameters(recurse):
#             yield param

#     def __getattr__(self, name: str) -> Any:
#         try:
#             return super().__getattr__(name)
#         except AttributeError:
#             attr = getattr(self.model, name)
#             if isinstance(attr, Callable):
#                 warnings.warn(f"forwarding `{name}` to the underlying model", UserWarning)
#             return attr

#     def __setattr__(self, name: str, value: Any) -> None:
#         try:
#             super().__setattr__(name, value)
#         except AttributeError:
#             setattr(self.model, name, value)


# import torch
# import torch.nn as nn
# from torch import Tensor
# from typing import List, Dict, Any
# from torch.func import functional_call
# import warnings
# from typing import List, Dict, Any
# from torch import Tensor, nn
# from torch.func import functional_call

# from .ties_merging_utils import check_parameterNamesMatch
# from .type import StateDict
# from .utils import timeit_context


# __all__ = ["get_task_wise_weights", "fuse_weights", "NonlinearTaskWiseMergedModel"]




# class NonLinearMappingNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_tasks):
#         super().__init__()
#         self.num_tasks = num_tasks
#         self.shared_mapping = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim),
#         )
        
#         self.task_specific_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.GELU(),
#                 nn.Linear(hidden_dim, output_dim)
#             ) for _ in range(num_tasks)
#         ])
        
#         self.weight_predictor = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, num_tasks),
#             nn.Softmax(dim=-1)
#         )
        
#         # # 初始化权重为0.2
#         # with torch.no_grad():
#         #     self.weight_predictor[-2].bias.fill_(0.2)

#     def forward(self, x, task_id=None):
#         shared_features = self.shared_mapping(x)
#         if task_id is not None:
#             return self.task_specific_heads[task_id](shared_features)
        
#         task_weights = self.weight_predictor(x)
#         return sum(head(shared_features) * task_weights[:, i:i+1] 
#                for i, head in enumerate(self.task_specific_heads))

# class TaskWiseMergedModel(nn.Module):
#     def __init__(
#         self,
#         pretrained_model: nn.Module,
#         task_vectors: List[StateDict],
#         mapping_network: NonLinearMappingNetwork,
#         clamp_weights: bool = True,
#     ):
#         super().__init__()
#         self._model = (pretrained_model,)
#         self.mapping_network = mapping_network
#         self.task_vectors = task_vectors
#         self.pretrained_state_dict = self.model.state_dict(keep_vars=False)
#         check_parameterNamesMatch(self.task_vectors)
#         self.clamp_weights = clamp_weights
#         self.merged_state_dict = None

#     @property
#     def model(self):
#         return self._model[0]

#     def merge_weights(self, features: Tensor = None):
#         """使用动态权重或固定权重进行融合"""
#         if features is not None:
#             # 动态权重模式
#             with torch.no_grad():
#                 task_weights = self.mapping_network.weight_predictor(features).mean(dim=0)
#         else:
#             # 固定权重模式 (初始化为0.2)
#             task_weights = torch.full((len(self.task_vectors),), 0.2, 
#                                     device=self.mapping_network.weight_predictor[-2].bias.device)
        
#         if self.clamp_weights:
#             task_weights = task_weights.clamp(0, 1)
        
#         device = task_weights.device
#         task_vector = {k: sum(task_weights[i] * vec[k].to(device) 
#                            for i, vec in enumerate(self.task_vectors))
#                      for k in self.task_vectors[0].keys()}
        
#         self.merged_state_dict = {
#             k: self.pretrained_state_dict[k].to(device, non_blocking=True) + task_vector[k]
#             for k in self.pretrained_state_dict.keys()
#         }

#     def forward(self, *args, **kwargs):
#         # 获取输入特征
#         input_tensor = kwargs.get('images', args[0] if args else None)
#         if input_tensor is not None:
#             with torch.no_grad():
#                 features = self.model(input_tensor)
#         else:
#             features = None
        
#         # 合并权重
#         if self.merged_state_dict is None or features is not None:
#             self.merge_weights(features)
            
#         return functional_call(
#             self.model,
#             self.merged_state_dict,
#             args=args,
#             kwargs=kwargs,
#             tie_weights=False,
#         )

#     def __getattr__(self, name: str) -> Any:
#         try:
#             return super().__getattr__(name)
#         except AttributeError:
#             attr = getattr(self.model, name)
#             if isinstance(attr, Callable):
#                 warnings.warn(f"forwarding `{name}` to the underlying model", UserWarning)
#             return attr

#     def __setattr__(self, name: str, value: Any) -> None:
#         try:
#             super().__setattr__(name, value)
#         except AttributeError:
#             setattr(self.model, name, value)





"""
```python
# Get the task-wise weights
task_wise_weights = get_task_wise_weights(num_models)

# Define the task vectors (in this case, we'll use the state_dict of the pretrained model)
task_vectors = ...

# Initialize the TaskWiseMergedModel
merged_model = TaskWiseMergedModel(pretrained_model, task_wise_weights, task_vectors)

# Now you can use the merged_model like a regular PyTorch model
outputs = merged_model(inputs)
```
"""
import logging
import types
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .ties_merging_utils import check_parameterNamesMatch
from .type import StateDict
from .utils import timeit_context

log = logging.getLogger(__name__)

__all__ = ["get_task_wise_weights", "fuse_weights", "TaskWiseMergedModel"]


def get_task_wise_weights(num_models: int, init_values: float = None):
    """
    This function generates a tensor of weights for each model.

    Args:
        num_models (int): The number of models.
        init_values (float, optional): The initial value for each weight. Defaults to None.

    Returns:
        Tensor: A tensor of weights for each model.
    """
    assert num_models >= 1, f"num_models must be >= 1, got {num_models}"
    if init_values is None:
        init_values = 1.0 / num_models
    return torch.full((num_models,), init_values, dtype=torch.float32)


def _fuse_weights(task_wise_weight: Tensor, tensors: List[Tensor]):
    """
    This function fuses the weights of the models.

    Args:
        task_wise_weight (Tensor): The weights for each model.
        tensors (List[Tensor]): The list of tensors to be fused.

    Returns:
        Tensor: The fused weights.
    """
    device = task_wise_weight.device
    return sum(task_wise_weight[i] * w.to(device) for i, w in enumerate(tensors))


def fuse_weights(task_wise_weight: Tensor, state_dicts: List[StateDict]) -> StateDict:
    """
    This function fuses the weights of the models and returns a state dictionary.

    Args:
        task_wise_weight (Tensor): The weights for each model. on cuda or cpu.
        state_dicts (List[StateDict]): The list of state dictionaries. on cpu.

    Returns:
        StateDict: The fused state dictionary.
    """
    num_models = len(state_dicts)
    assert task_wise_weight.dim() == 1, f"task_wise_weight must be a 1D tensor, got {task_wise_weight.dim()}"
    assert num_models == task_wise_weight.size(
        0
    ), f"num_models must be equal to the number of state_dicts, got {num_models} and {task_wise_weight.size(0)}"
    return {k: _fuse_weights(task_wise_weight, [sd[k] for sd in state_dicts]) for k in state_dicts[0].keys()}


class TaskWiseMergedModel(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module,
        task_wise_weight: Tensor,
        task_vectors: List[StateDict],
        clamp_weights: bool = True,
    ):
        super().__init__()
        self._model = (pretrained_model,)  # self._model should be on cpu

        self.task_wise_weight = nn.Parameter(task_wise_weight, requires_grad=True)
        self.task_vectors = task_vectors  # should be on cpu
        self.pretrained_state_dict: StateDict = self.model.state_dict(keep_vars=False)
        check_parameterNamesMatch(self.task_vectors)
        self.clamp_weights = clamp_weights
        self.merged_state_dict = None

    @property
    def model(self):
        return self._model[0]

    def merge_weights(self):
        if self.clamp_weights:
            task_wise_weight = self.task_wise_weight.clamp(0, 1)
        else:
            task_wise_weight = self.task_wise_weight
        device = task_wise_weight.device
        task_vector = fuse_weights(task_wise_weight, self.task_vectors)
        self.merged_state_dict = {k: self.pretrained_state_dict[k].to(device, non_blocking=True) for k in self.pretrained_state_dict.keys()}
        for k in task_vector.keys():
            self.merged_state_dict[k] += task_vector[k]

    def forward(self, *args, **kwargs):
        if self.merged_state_dict is None:
            self.merge_weights()
        return functional_call(
            self.model,
            self.merged_state_dict,
            args=args,
            kwargs=kwargs,
            tie_weights=False,
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            attr = getattr(self.model, name)
            if isinstance(attr, Callable):
                warnings.warn(f"forwarding `{name}` to the underlying model", UserWarning)
            return attr

    def __setattr__(self, name: str, value: Any) -> None:
        try:
            super().__setattr__(name, value)
        except AttributeError:
            setattr(self.model, name, value)