from .nq_learner import NQLearner
from .myalg_learner import MyAlgLearner

REGISTRY = {}

REGISTRY["nq_learner"] = NQLearner

REGISTRY["MyAlg_learner"] = MyAlgLearner
