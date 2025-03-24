REGISTRY = {}

from .parallel_hpn import ParallelHpn
REGISTRY["parallel_HPN"] = ParallelHpn

from .parallel_myalg import ParallelMyAlg
REGISTRY["parallel_MyAlg"] = ParallelMyAlg