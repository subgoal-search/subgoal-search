
import collections

import numpy as np

from alpacka.agents import base
from alpacka.agents import models
from alpacka.data import ops
from alpacka.utils import metric as metric_utils


ChildInfo = collections.namedtuple('ChildInfo', [
    'action_list',
    'reward',
    'done',
])


class GoalBuilder:
    def build_goals(self, state):
        raise NotImplementedError


TreeStats = collections.namedtuple('TreeStats', [
    'subgoal_sum', 'nodes', 'leaves'
])


class Node:
    prior_probability = None

    def __init__(self, state):
        self.state = state
        self.children = []
        self.children_infos = []
        self.was_expanded = False

    def visit(self, reward, value, discount):
        raise NotImplementedError

    def quality(self, discount):
        raise NotImplementedError

    @property
    def count(self):
        raise NotImplementedError

    def value(self, discount):
        return (
            sum(
                child.quality(discount) * child.count for child in self.children
            ) / sum(child.count for child in self.children)
        )

    @property
    def is_leaf(self):
        return not self.children


def find_solving_actions(node):
    if node.is_leaf:
        return None

    for child, child_info in zip(node.children, node.children_infos):
        cur_action_state = (child_info.action_list, child.state)
        if child_info.done:
            return [cur_action_state]
        maybe_actions_states = find_solving_actions(child)
        if maybe_actions_states is not None:
            return [cur_action_state] + maybe_actions_states


def compute_tree_stats_helper(node):
    if node.is_leaf:
        return TreeStats(subgoal_sum=0, nodes=1, leaves=1)

    result = TreeStats(subgoal_sum=len(node.children), nodes=1, leaves=0)
    for child in node.children:
        stats = compute_tree_stats_helper(child)
        result = TreeStats(
            subgoal_sum=result.subgoal_sum + stats.subgoal_sum,
            nodes=result.nodes + stats.nodes,
            leaves=result.leaves + stats.leaves,
        )
    return result


def compute_tree_stats(node):
    stats = compute_tree_stats_helper(node)
    return {
        'nodes': stats.nodes,
        'leaves': stats.leaves,
        'inner_nodes': stats.nodes - stats.leaves,
        'inner_nodes_degree_mean': stats.subgoal_sum / max((stats.nodes - stats.leaves), 1),
    }


class DeadEnd(Exception):
    pass

class TreeSearchAgent(base.OnlineAgent):
    def __init__(
        self,
        n_passes=10,
        discount=0.99,
        depth_limit=float('+inf'),
        n_leaves_to_expand=1,
        model_class=models.PerfectModel,
        keep_tree_between_steps=True,
        callback_classes=None,
        **kwargs
    ):
        if not callback_classes:
            callback_classes = []
        if not model_class.is_perfect:
            callback_classes = [ImperfectModelCallback] + callback_classes
        super().__init__(callback_classes=callback_classes, **kwargs)

        self.n_passes = n_passes
        self._discount = discount
        self._depth_limit = depth_limit
        self._n_leaves_to_expand = n_leaves_to_expand
        self._model_class = model_class
        self._model = None
        self._root = None
        self._root_state = None
        self._keep_tree_between_steps = keep_tree_between_steps

        self._dead_end_hits = None

    @property
    def discount(self):
        return self._discount

    @property
    def model(self):
        return self._model

    def reset_tree(self, state):
        self._root = self._init_root_node(state)

    def _init_root_node(self, state):
        raise NotImplementedError

    def _init_child_nodes(self, leaf, observation):
        raise NotImplementedError
        yield  

    def network_signature(self, observation_space, action_space):
        raise NotImplementedError


    def _before_pass(self):
        pass
    def _before_model_step(self, node):
        pass

    def _before_real_step(self, node):
        pass

    def _on_new_root(self, root):
        pass

    def _make_filter_fn(self, exploratory):
        del exploratory
        return lambda _: True

    @property
    def _zero_quality(self):
        return 0

    @property
    def _dead_end_quality(self):
        return 0

    def _choose_child(self, node, exploratory, strict_filter=True):
        filter_fn = self._make_filter_fn(exploratory)
        actions = [
            action for (action, child) in enumerate(node.children)
            if filter_fn(child)
        ]
        if not actions:
            if strict_filter:
                raise DeadEnd

            actions = list(range(len(node.children)))

        action = self._choose_action(
            node, actions, exploratory=exploratory
        )
        assert action in actions, (
            'Invalid implementation of _choose_action: action '
            '{} disallowed.'.format(action)
        )
        return (node.children[action], action)

    def _choose_action(self, node, actions, exploratory):
        raise NotImplementedError

    def _traverse(self, root, observation, path):
        assert not path

        path.append((None, root))
        node = root
        done = False
        n_leaves_left = self._n_leaves_to_expand
        quality = self._zero_quality
        while not node.is_leaf and not done and len(path) < self._depth_limit:
            agent_info = self._compute_node_info(node)

            self._before_model_step(node)
            parent = node
            (node, action) = self._choose_child(
                node, exploratory=True, strict_filter=True
            )
            (observation, reward, done) = (
                parent.children[action].state,
                parent.children_infos[action].reward,
                parent.children_infos[action].done,
            )

            for callback in self._callbacks:
                callback.on_model_step(
                    agent_info, action, observation, reward, done
                )

            path.append((reward, node))

            if node.is_leaf and n_leaves_left > 0:
                if not done:
                    quality = self._expand_leaf(node, observation)
                n_leaves_left -= 1

        if node.is_leaf and not done and n_leaves_left > 0:
            quality = self._expand_leaf(node, observation)

        return (path, quality)

    def _expand_leaf(self, leaf, observation):
        if leaf.was_expanded:
            self._dead_end_hits += 1
            raise DeadEnd
        leaf.was_expanded = True

        (leaf.children, leaf.children_infos) = self._init_child_nodes(leaf, observation)
        for node in leaf.children:
            quality = node.quality(self._discount)
            prob = node.prior_probability
            prob_ok = prob is None or np.isscalar(prob)
            assert np.isscalar(quality) and prob_ok, (
                'Invalid shape of node quality or prior probability - expected '
                'scalars, got {} and {}. Check if your network architecture is '
                'appropriate for the observation shape.'.format(
                    quality.shape, prob.shape if prob is not None else None
                )
            )

        if leaf is self._root:
            self._on_new_root(leaf)

        (child, _) = self._choose_child(
            leaf, exploratory=True, strict_filter=True
        )
        return child.quality(self._discount)

    def _backpropagate(self, quality, path):
        for (reward, node) in reversed(path):
            node.visit(reward, value=quality, discount=self._discount)
            if reward is None:
                break
            quality = reward + self._discount * quality

    def _run_pass(self, root, observation):
        for callback in self._callbacks:
            callback.on_pass_begin()

        path = []
        try:
            (path, quality) = self._traverse(root, observation, path)
        except DeadEnd:
            quality = self._dead_end_quality
        self._backpropagate(quality, path)

        for callback in self._callbacks:
            callback.on_pass_end()

    def reset(self, observation):
        self._root = self._init_root_node(observation)

    def act(self, observation):
        self._root_state = observation

        if not self._keep_tree_between_steps:
            self._root = self._init_root_node(self._root_state)

        self._dead_end_hits = 0
        for _ in range(self.n_passes):
            self._before_pass()
            self._run_pass(self._root, observation)
        if len(self._root.children) == 0:
            return None

        agent_info = {
            '_node': self._root,
            'dead_end_hits': self._dead_end_hits,
        }
        agent_info.update(self._compute_node_info(self._root))
        agent_info.update(self._compute_tree_metrics(self._root))

        self._before_real_step(self._root)
        (new_root, action) = self._choose_child(
            self._root, exploratory=False, strict_filter=False
        )
        multi_step_info = self._root.children_infos[action]

        self._root = new_root
        if not self._root.is_leaf:
            self._on_new_root(self._root)

        return action, new_root.state, multi_step_info, agent_info

    def postprocess_transitions(self, transitions):
        for transition in transitions:
            transition.agent_info.update(
                self._compute_node_info(transition.agent_info.pop('_node'))
            )
        return transitions

    def _compute_node_info(self, node):
        value = node.value(self._discount)
        qualities = np.array(
            [child.quality(self._discount) for child in node.children]
        )
        action_counts = np.array([child.count for child in node.children])

        action_histogram_smooth = action_counts / (np.sum(action_counts) + 1e-6)

        action_histogram = (action_counts - 1) / (
            np.sum(action_counts - 1) + 1e-6
        )
        return {
            'value': value,
            'qualities': qualities,
            'action_histogram_smooth': action_histogram_smooth,
            'action_histogram': action_histogram,
        }

    def _compute_tree_metrics(self, root):
        leaf_depths = []
        path = [root]

        children_visited = [0]

        def go_to_parent():
            path.pop()
            children_visited.pop()


        while path:
            node = path[-1]
            if node.is_leaf:
                leaf_depths.append(len(path) - 1)
                go_to_parent()
            elif children_visited[-1] == len(node.children):

                go_to_parent()
            else:

                path.append(node.children[children_visited[-1]])
                children_visited[-1] += 1
                children_visited.append(0)

        return {
            'depth_mean': sum(leaf_depths) / len(leaf_depths),
            'depth_max': max(leaf_depths),
        }

    def _compute_model_prediction(self, action):
        prediction = yield from self._model.predict_steps(
            [action], include_state=False
        )
        keys = ['predicted_observation', 'predicted_reward', 'predicted_done']
        return {
            key: value
            for key, [value] in zip(keys, prediction)
        }

    @staticmethod
    def compute_metrics(episodes):
        def episode_info(key):
            for episode in episodes:
                yield from episode.transition_batch.agent_info[key]

        def entropy(probs):
            def plogp(p):

                return p * np.log(p) if p else 0
            return -np.sum([plogp(p) for p in probs])

        return {
            'depth_mean': np.mean(list(episode_info('depth_mean'))),
            'depth_max': max(episode_info('depth_max')),
            'entropy_mean': np.mean(
                list(map(entropy, episode_info('action_histogram')))
            ),
            'entropy_smooth_mean': np.mean(
                list(map(entropy, episode_info('action_histogram_smooth')))
            ),
            **metric_utils.compute_scalar_statistics(
                list(episode_info('value')),
                prefix='value',
                with_min_and_max=True
            ),
        }


class ImperfectModelCallback(base.AgentCallback):
    def __init__(self, agent):
        super().__init__(agent)
        self._last_observation = None

    def on_episode_begin(self, env, observation, epoch):
        self._last_observation = observation

    def on_real_step(self, agent_info, action, observation, reward, done):
        if not ops.nested_array_equal(
                agent_info['predicted_observation'],
                observation
        ):
            self._agent.model.catch_up(observation)
            correct_state = self._agent.model.clone_state()
            self._agent.reset_tree(correct_state)

        self._agent.model.correct(
            self._last_observation, action, observation, reward, done,
            agent_info
        )

        self._last_observation = observation
