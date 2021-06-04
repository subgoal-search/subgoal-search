import collections

import gin

from alpacka.data.ops import *


Transition = collections.namedtuple(
    'Transition',
    [
        'observation',
        'action_list',
        'reward',
        'done',
        'next_observation',
        'agent_info',
    ]
)


Episode = collections.namedtuple(
    'Episode',
    [
        'transitions',
        'return_',
        'solved',
    ]
)


TensorSignature = collections.namedtuple(
    'TensorSignature', ['shape', 'dtype']
)
TensorSignature.__new__.__defaults__ = (np.float32,)
register_leaf_type(TensorSignature)


NetworkSignature = collections.namedtuple(
    'NetworkSignature', ['input', 'output']
)


class NetworkRequest:
    pass


def request_type_id(request_type):
    try:
        return request_type.slug
    except AttributeError:
        raise ValueError(
            'Expected request type registered by '
            'alpacka.data.register_prediction_request() function.'
        )


def register_prediction_request(name, slug=None, module='alpacka.data'):
    request_type = collections.namedtuple(name, ['value'], module=module)
    request_type.slug = slug or name

    gin_type = gin.external_configurable(request_type, module=module)

    type_id = request_type_id(gin_type)
    if type_id in register_prediction_request.taken_ids:
        raise ValueError(f'Request id {type_id} is already taken.')
    register_prediction_request.taken_ids.add(type_id)

    return gin_type


register_prediction_request.taken_ids = set()


AgentRequest = register_prediction_request('AgentRequest', slug='agent')  

ModelRequest = register_prediction_request('ModelRequest', slug='model')  
