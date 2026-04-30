"""Abstract base class shared by all trajectory tokenizers."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class BaseTokenizer(nn.Module, ABC):
    """
    Abstract base class for trajectory tokenizers.

    A tokenizer maps a continuous future trajectory to a sequence of
    discrete token indices (``encode``) and reconstructs the trajectory
    from those tokens (``decode``). Concrete subclasses (e.g. bin-based,
    k-means, VQ-VAE) only need to implement the abstract methods below.

    Args:
        num_future_points: Length of the (dense) future trajectory.
        num_skills:        Total number of discrete tokens produced per
                           sample. For tokenizers that encode ``dx`` and
                           ``dy`` separately this is ``2 * num_sample_points``.
    """

    def __init__(self, num_future_points: int, num_skills: int):
        nn.Module.__init__(self)
        self.num_future_points = int(num_future_points)
        self.num_skills = int(num_skills)

    # ------------------------------------------------------------------ API
    @abstractmethod
    def encode(
        self, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode a continuous trajectory into discrete token indices."""
        raise NotImplementedError

    @abstractmethod
    def decode(
        self, ego_lcf: torch.Tensor, token_indices: torch.Tensor
    ) -> torch.Tensor:
        """Decode token indices back into a continuous trajectory."""
        raise NotImplementedError

    @abstractmethod
    def forward_train(self, planning_ann_info: Dict) -> Dict:
        """Training-time forward pass; returns a dict of losses/metrics."""
        raise NotImplementedError

    # ----------------------------------------------------------- properties
    @property
    def vocab_size(self) -> int:
        """Total size of the discrete vocabulary."""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        # By convention ``forward`` performs an encode-then-decode round
        # trip; subclasses may override if a different behaviour is wanted.
        return self.decode(*args, **kwargs)
