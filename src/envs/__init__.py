from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env



def env_fn(env, **kwargs) -> MultiAgentEnv:    #‘**kwargs’: 这是一个关键字参数列表，它可以接受任意数量的关键字参数。这些参数将在后续调用 env 时传递给它。函数的主要作用是调用传入的 env 对象（预期是一个类）并传递给它 **kwargs 中的所有关键字参数。
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)    #使用partial函数来“部分应用”env_fn函数。具体来说，它固定了env_fn的一个参数env的值为StarCraft2Env。这意味着当你稍后调用这个部分应用的函数时，你不需要再提供env参数，因为它已经被设置为StarCraft2Env了。



if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
