
from alpacka import data


class EnvModel:
    def __init__(self, env):
        self._action_space = env.action_space
        self._observation_space = env.observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def step(self, action):
        raise NotImplementedError
        yield  

    def predict_steps(self, actions, include_state):
        raise NotImplementedError
        yield  

    def catch_up(self, observation):
        pass

    def correct(self, obs, action, next_obs, reward, done, agent_info):
        pass

    def clone_state(self):
        raise NotImplementedError

    def restore_state(self, state):
        raise NotImplementedError


class PerfectModel(EnvModel):
    is_perfect = True

    def __init__(self, env):
        super().__init__(env)
        self._env = env

    def step(self, action):
        return self._env.step(action)[:-1]
        yield  

    def predict_steps(self, actions, include_state):
        return step_into_successors(self._env, actions, include_state)
        yield  

    def clone_state(self):
        return self._env.clone_state()

    def restore_state(self, state):
        return self._env.restore_state(state)


def step_into_successors(env, actions, include_state):
    init_state = env.clone_state()

    def step_and_rewind(action):
        (observation, reward, done, _) = env.step(action)
        if include_state:
            state = env.clone_state()
        env.restore_state(init_state)
        info = (observation, reward, done)
        if include_state:
            info += (state,)
        return info

    (observations, rewards, dones, *maybe_states) = list(zip(*[
        step_and_rewind(action) for action in actions
    ]))
    return list(map(
        data.nested_stack, (observations, rewards, dones)
    )) + maybe_states

