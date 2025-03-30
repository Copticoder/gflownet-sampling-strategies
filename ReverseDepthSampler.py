from typing import Any, Tuple
import torch
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from gfn.samplers import Sampler
from gfn.states import States
from gfn.env import Env
from gfn.actions import Actions
from gfn.utils.handlers import (
    has_conditioning_exception_handler,
    no_conditioning_exception_handler,
)

class ReverseDepthSampler(Sampler):
    def __init__(self, estimator: DiscretePolicyEstimator, n_iterations: int, reverse_depth_boolean: bool,growth_parameter: int = 0.1, reverse_depth_estimator: ScalarEstimator | None = None):
        super().__init__(estimator)
        # n_iterations used for growing triangle
        self.n_iterations = n_iterations
        self.iteration_counter = 0
        self.reverse_depth_counter = 0
        self.reverse_depth_estimator = reverse_depth_estimator
        self.growth_parameter = growth_parameter
        # used to enable/disable topological sampling
        self.reverse_depth_boolean = reverse_depth_boolean
        # initialize the hash table for the reverse_depth of each state 
        states_dict = {}

    def sample_actions(
        self,
        env: Env,
        states: States,
        conditioning: torch.Tensor | None = None,
        save_estimator_outputs: bool = False,
        save_logprobs: bool = True,
        **policy_kwargs: Any,
    ) -> Tuple[Actions, torch.Tensor | None, torch.Tensor | None]:
        """Samples actions from the given states.

        Args:
            env: The environment to sample actions from.
            states: A batch of states.
            conditioning: An optional tensor of conditioning information.
            save_estimator_outputs: If True, the estimator outputs will be returned.
            save_logprobs: If True, calculates and saves the log probabilities of sampled
                actions.
            policy_kwargs: keyword arguments to be passed to the
                `to_probability_distribution` method of the estimator. For example, for
                DiscretePolicyEstimators, the kwargs can contain the `temperature`
                parameter, `epsilon`, and `sf_bias`. In the continuous case these
                kwargs will be user defined. This can be used to, for example, sample
                off-policy.

        When sampling off policy, ensure to `save_estimator_outputs` and not
            `calculate logprobs`. Log probabilities are instead calculated during the
            computation of `PF` as part of the `GFlowNet` class, and the estimator
            outputs are required for estimating the logprobs of these off policy
            actions.

        Returns:
            A tuple of tensors containing:
             - An Actions object containing the sampled actions.
             - An optional tensor of shape `batch_shape` containing the log probabilities of
                the sampled actions under the probability distribution of the given
                states.
             - An optional tensor of shape `batch_shape` containing the estimator outputs
        """
        # TODO: Should estimators instead ignore None for the conditioning vector?
        if conditioning is not None:
            with has_conditioning_exception_handler("estimator", self.estimator):
                estimator_output = self.estimator(states, conditioning)
        else:
            with no_conditioning_exception_handler("estimator", self.estimator):
                estimator_output = self.estimator(states)
                reverse_depth_output = torch.round(self.reverse_depth_estimator(states))
        
        if self.reverse_depth_boolean and self.reverse_depth_counter <= (env.ndim*(env.height-1)):
            # allow only the exit action for states that are at the reverse depth
            non_exit_condition = (reverse_depth_output != torch.full((states.tensor.size(dim=0),),self.reverse_depth_counter))
            states.forward_masks[non_exit_condition,-1] = False
            # enable exit action for anti-diagonal states
            exit_condition = ~non_exit_condition
            states.forward_masks[exit_condition,-1] = True
            # disable all actions for states that are at the same level as the triangle counter
            states.forward_masks[exit_condition,:-1] = False
            
        dist = self.estimator.to_probability_distribution(
            states, estimator_output, **policy_kwargs
        )
        
        with torch.no_grad():
            actions = dist.sample()

        if save_logprobs:
            log_probs = dist.log_prob(actions)
            if torch.any(torch.isinf(log_probs)):
                raise RuntimeError("Log probabilities are inf. This should not happen.")
        else:
            log_probs = None

        actions = env.actions_from_tensor(actions)
        if not save_estimator_outputs:
            estimator_output = None

        assert log_probs is None or log_probs.shape == actions.batch_shape
        # assert estimator_output is None or estimator_output.shape == actions.batch_shape  TODO: check expected shape
        return actions, log_probs, estimator_output
    
    def initial_depth_estimation(self, env: Env, states: States) -> torch.Tensor:
        """
        This is the initial depth estimation function that is used at the end of the first training iteration. The goal is to provide an initial depth to each state in the environment. The logic is as follows: For every state in the trajectories, subtract its depth from the maximum depth of this trajcetory, reverse depth.
        """
        
    def sample_trajectories(self, *args, **kwargs):
        # get env from args
        Trajectories = super().sample_trajectories(*args, **kwargs)
        env = args[0]
        self.iteration_counter += 1
        # increment the triangle counter every time the iteration counter is divisible by 
        if self.iteration_counter > (self.growth_parameter*self.n_iterations * self.reverse_depth_counter)/((env.ndim*env.height)-1):
                self.reverse_depth_counter += 1
        return Trajectories