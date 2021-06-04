import gin
import numpy as np

from alpacka import data
from alpacka import math
from alpacka.agents import core
from alpacka.agents import mcts
from alpacka.agents.tree_search import DeadEnd
from alpacka.utils import space as space_utils


class NewLeafRater:
    def __init__(
        self, agent, use_policy=False, boltzmann_temperature=None
    ):
        self._agent = agent
        assert not (use_policy and boltzmann_temperature)
        assert use_policy
        self._use_policy = use_policy
        self._boltzmann_temperature = boltzmann_temperature

    def __call__(self, observation, model=None):
        if self._use_policy:
            (qualities, prior, states, child_infos) = self.qualities_and_prior(
                observation, model
            )
        else:
            assert False
            qualities, states, child_infos = self.qualities(observation, model)
            prior = mcts.uniform_prior(len(qualities))

        if self._boltzmann_temperature is not None:
            prior = math.softmax(
                np.array(qualities) / self._boltzmann_temperature
            )

        return list(zip(qualities, prior, states, child_infos))

    def qualities(self, observation, model):
        raise NotImplementedError

    def qualities_and_prior(self, observation, model):
        raise NotImplementedError

    def network_signature(self, observation_space, action_space):
        raise NotImplementedError


@gin.configurable
class RolloutNewLeafRater(NewLeafRater):
    def __init__(
        self,
        agent,
        boltzmann_temperature=None,
        rollout_agent_class=core.RandomAgent,
        rollout_time_limit=100,
    ):
        super().__init__(
            agent,
            use_policy=False,
            boltzmann_temperature=boltzmann_temperature,
        )
        self._discount = agent.discount
        self._rollout_agent = rollout_agent_class()
        self._time_limit = rollout_time_limit

    def qualities(self, observation, model):
        init_state = model.clone_state()

        child_qualities = []
        for init_action in space_utils.element_iter(model.action_space):
            (observation, init_reward, done) = yield from model.step(init_action)
            yield from self._rollout_agent.reset(model, observation)
            value = 0
            total_discount = 1
            time = 0
            while not done and time < self._time_limit:
                (action, _) = yield from self._rollout_agent.act(observation)
                (observation, reward, done) = yield from model.step(action)
                value += total_discount * reward
                total_discount *= self._discount
                time += 1
            child_qualities.append(init_reward + self._discount * value)
            model.restore_state(init_state)
        return child_qualities

    def network_signature(self, observation_space, action_space):
        return self._rollout_agent.network_signature(
            observation_space, action_space
        )


@gin.configurable
class ValueNetworkNewLeafRater(NewLeafRater):

    def __init__(
        self, agent, value_function, goal_builder,
        use_policy=False, boltzmann_temperature=None
    ):
        assert use_policy, 'Not supported'
        super().__init__(
            agent,
            use_policy=use_policy,
            boltzmann_temperature=boltzmann_temperature,
        )
        self._discount = agent.discount
        self._value_function = value_function
        self._goal_builder = goal_builder

    def qualities(self, observation, model):
        raise ValueError()

    def qualities_and_prior(self, observation, model):
        states_infos_and_prior = self._goal_builder.build_goals(observation)
        if not states_infos_and_prior:
            raise DeadEnd
        (states, child_infos, prior) = zip(*states_infos_and_prior)

        values = self._value_function(states)
        qualities = [

            info.reward + self._discount * value * (1 - info.done)
            for value, info in zip(values, child_infos)
        ]
        return (qualities, prior, states, child_infos)

    def network_signature(self, observation_space, action_space):
        n_actions = space_utils.max_size(action_space)
        if self._use_policy:
            return data.NetworkSignature(
                input=space_utils.signature(observation_space),
                output=(
                    data.TensorSignature(shape=(1,)),
                    data.TensorSignature(shape=(n_actions,))
                ),
            )
        else:

            return data.NetworkSignature(
                input=space_utils.signature(observation_space),
                output=data.TensorSignature(shape=(1,)),
            )


@gin.configurable
class QualityNetworkNewLeafRater(NewLeafRater):
    def qualities_and_prior(self, observation, model):
        del model
        (qualities, prior) = yield np.expand_dims(observation, axis=0)
        qualities = np.squeeze(qualities, axis=0)
        prior = np.squeeze(prior, axis=0)
        return (qualities, prior)

    def qualities(self, observation, model):
        del model
        qualities = yield np.expand_dims(observation, axis=0)
        return np.squeeze(qualities, axis=0)

    def network_signature(self, observation_space, action_space):
        n_actions = space_utils.max_size(action_space)
        action_vector_sig = data.TensorSignature(shape=(n_actions,))
        if self._use_policy:
            output_sig = (action_vector_sig,) * 2
        else:
            output_sig = action_vector_sig


        return data.NetworkSignature(
            input=space_utils.signature(observation_space),
            output=output_sig,
        )


class Node(mcts.Node):
    def __init__(self, init_quality, prior_probability, state):
        super().__init__(prior_probability=prior_probability, state=state)
        if init_quality is None:
            self._quality_sum = 0
            self._quality_count = 0
        else:
            self._quality_sum = init_quality
            self._quality_count = 1

    def visit(self, reward, value, discount):
        if reward is None:
            return

        quality = reward + discount * value
        self._quality_sum += quality
        self._quality_count += 1

    def quality(self, discount):
        del discount
        return self._quality_sum / self._quality_count

    @property
    def count(self):
        return self._quality_count


class StochasticMCTSAgent(mcts.MCTSAgent):

    def __init__(
        self, value_function, goal_builder, **kwargs,
    ):
        super().__init__(**kwargs)
        self._new_leaf_rater = ValueNetworkNewLeafRater(
            agent=self,
            use_policy=True,
            value_function=value_function,
            goal_builder=goal_builder,
        )

    def _init_root_node(self, state):
        return Node(init_quality=None, prior_probability=None, state=state)

    def _init_child_nodes(self, leaf, observation):
        del leaf
        child_qualities_probs_states_and_infos = self._new_leaf_rater(observation)
        return ([
            Node(quality, prob, state=state)
            for (quality, prob, state, _) in child_qualities_probs_states_and_infos
        ], [
            child_info for (_, _, _, child_info) in child_qualities_probs_states_and_infos
        ])

    def network_signature(self, observation_space, action_space):
        return self._new_leaf_rater.network_signature(
            observation_space, action_space
        )

