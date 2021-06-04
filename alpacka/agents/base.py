
from alpacka import data

from alpacka import metric_logging
from alpacka import utils


class Agent:
    def __init__(self, parameter_schedules=None):
        self._parameter_schedules = parameter_schedules or {}

    def solve(self, init_state, time_limit, epoch=None):  
        del init_state
        del time_limit
        for attr_name, schedule in self._parameter_schedules.items():
            param_value = schedule(epoch)
            utils.recursive_setattr(self, attr_name, param_value)
            metric_logging.log_scalar(
                'agent_param/' + attr_name, epoch, param_value
            )
        return

    def network_signature(self, observation_space, action_space):  
        del observation_space
        del action_space
        return None

    def close(self):
        pass


class OnlineAgent(Agent):

    def __init__(self, callback_classes=(), **kwargs):
        super().__init__(**kwargs)

        self._action_space = None
        self._epoch = None
        self._callbacks = [
            callback_class(self) for callback_class in callback_classes
        ]

    def reset(self, observation):  
        del observation
        return

    def act(self, observation):
        raise NotImplementedError

    def postprocess_transitions(self, transitions):
        return transitions

    @staticmethod
    def compute_metrics(episodes):
        del episodes
        return {}

    def solve(self, init_state, time_limit, epoch=None):
        super().solve(init_state, time_limit, epoch=epoch)

        self._epoch = epoch

        observation = init_state
        self.reset(observation)

        for callback in self._callbacks:
            callback.on_episode_begin(None, observation, epoch)

        transitions = []
        done = False
        steps_completed = 0
        while not done and steps_completed < time_limit:
            step_info = self.act(observation)
            if step_info is None:
                break
            steps_completed += 1
            (action, new_subgoal, multi_step_info, agent_info) = step_info
            next_observation = new_subgoal
            reward = multi_step_info.reward
            done = multi_step_info.done

            for callback in self._callbacks:
                callback.on_real_step(
                    agent_info, action, next_observation, reward, done
                )

            transitions.append(data.Transition(
                observation=observation,
                action_list=multi_step_info.action_list,
                reward=reward,
                done=done,
                next_observation=next_observation,
                agent_info=agent_info,
            ))
            observation = next_observation

        for callback in self._callbacks:
            callback.on_episode_end()

        transitions = self.postprocess_transitions(transitions)

        return_ = sum(transition.reward for transition in transitions)
        return data.Episode(
            transitions=transitions,
            return_=return_,
            solved=done,
        )


class AgentCallback:
    def __init__(self, agent):
        self._agent = agent

    def on_episode_begin(self, env, observation, epoch):
        pass

    def on_episode_end(self):
        pass

    def on_real_step(self, agent_info, action, observation, reward, done):
        pass

    def on_pass_begin(self):
        pass

    def on_pass_end(self):
        pass

    def on_model_step(self, agent_info, action, observation, reward, done):
        pass
