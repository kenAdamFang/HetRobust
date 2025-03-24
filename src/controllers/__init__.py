REGISTRY = {}

from .hpn_controller import HPNMAC
from .basic_controller import BasicMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["hpn_mac"] = HPNMAC
