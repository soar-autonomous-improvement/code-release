from .continuous.bc import BCAgent
from .continuous.calql import CalQLAgent
from .continuous.cql import ContinuousCQLAgent
from .continuous.gc_bc import GCBCAgent
from .continuous.gc_ddpm_bc import GCDDPMBCAgent
from .continuous.gc_iql import GCIQLAgent
from .continuous.iql import IQLAgent
from .continuous.sac import SACAgent
from .continuous.stable_contrastive_rl import StableContrastiveRLAgent
from .continuous.vision_backbone_1 import VisionBackbone1
from .discrete.cql import DiscreteCQLAgent, QTransformerDiscretizedCQLAgent

agents = {
    "gc_bc": GCBCAgent,
    "gc_iql": GCIQLAgent,
    "gc_ddpm_bc": GCDDPMBCAgent,
    "bc": BCAgent,
    "iql": IQLAgent,
    "stable_contrastive_rl": StableContrastiveRLAgent,
    "cql": ContinuousCQLAgent,
    "calql": CalQLAgent,
    "discrete_cql": DiscreteCQLAgent,
    "q_transformer_discrete_cql": QTransformerDiscretizedCQLAgent,
    "sac": SACAgent,
    "vision_backbone_1": VisionBackbone1,
}
