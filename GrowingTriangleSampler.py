from typing import Any, Tuple
import torch
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.states import States
from gfn.env import Env
from gfn.actions import Actions
from gfn.utils.handlers import (
    has_conditioning_exception_handler,
    no_conditioning_exception_handler,
)

class GrowingTriangleSampler(Sampler):
    def __init__(self, estimator: DiscretePolicyEstimator, n_iterations: int, growth_parameter: int = 0.1):
        super().__init__(estimator)
        # n_iterations used for growing triangle
        self.n_iterations = n_iterations
        self.iteration_counter = 0
        self.triangle_counter = 1
        self.growth_parameter = growth_parameter
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
        # disable exit action for non-anti-diagonal states  
        non_exit_condition = (states.tensor < env.height-self.triangle_counter).any(dim=-1)
        states.forward_masks[non_exit_condition,-1] = False
        # allow exit action only for anti-diagonal states
        exit_condition = (states.tensor >= env.height-self.triangle_counter).any(dim=-1)
        states.forward_masks[exit_condition,:] = False
        states.forward_masks[exit_condition,-1] = True

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

    def sample_trajectories(self, *args, **kwargs):
        # get env from args
        env = args[0]
        self.iteration_counter += 1
        # increment the triangle counter every time the iteration counter is divisible by 
        if self.iteration_counter % (((self.n_iterations*0.01) // ((2*env.height)-1))) == 0:
            if self.triangle_counter == env.height-1:
                self.triangle_counter = 1
            else:
                self.triangle_counter = min(self.triangle_counter + 1, env.height-1)
        Trajectories = super().sample_trajectories(*args, **kwargs)
        return Trajectories