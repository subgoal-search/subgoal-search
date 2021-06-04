

import gin

from alpacka.agents import core
from alpacka.agents import stochastic_mcts
from alpacka.agents.base import *




def configure_agent(agent_class):
    return gin.external_configurable(
        agent_class, module='alpacka.agents'
    )



ActorCriticAgent = configure_agent(core.ActorCriticAgent)
PolicyNetworkAgent = configure_agent(core.PolicyNetworkAgent)
RandomAgent = configure_agent(core.RandomAgent)



StochasticMCTSAgent = configure_agent(stochastic_mcts.StochasticMCTSAgent)








