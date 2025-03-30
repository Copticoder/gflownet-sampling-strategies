from abc import ABC
from gfn.states import DiscreteStates
import torch
from typing import Optional, Tuple
class ReverseDepthDiscreteStates(DiscreteStates):
    def __init__(
        self,
        tensor: torch.Tensor,
        forward_masks: Optional[torch.Tensor] = None,
        backward_masks: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(tensor, forward_masks, backward_masks)
        self.reverse_depths = torch.zeros(
                (*self.batch_shape, self.__class__.n_actions),
                dtype=torch.bool,
                device=self.__class__.device,
            )      
    
    def __repr__(self):
        return f"DiscreteStates(tensor={self.tensor}, forward_masks={self.forward_masks}, backward_masks={self.backward_masks}, reverse_depths={self.reverse_depths})"
    
    