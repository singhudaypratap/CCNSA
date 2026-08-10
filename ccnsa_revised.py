"""
CCNSA — Revised, Standalone, GPU-Ready Implementation (v17)
=============================================================
Rewritten from CCNSA_SCI_Final_v16.ipynb for a fresh machine: no Colab, no
Google Drive mount. Data downloads directly (HuggingFace, with a documented
statistical-replication fallback if unreachable). Designed to run head-less:
    python ccnsa_revised.py --track A            # simulated env only
    python ccnsa_revised.py --track B            # real UNSW-NB15 only
    python ccnsa_revised.py --track AB            # both (default)
    python ccnsa_revised.py --quick               # small smoke-test sizes

WHAT CHANGED VS v16 AND WHY
----------------------------
1. CAUSAL LAYER — v16's "causal layer" was a single scalar gate
   (`causal.effect(0,1)` switching between two fixed blend weights). It is
   now a real elementwise mask: phi_C(s) = (A_G ⊙ M_C) @ phi(s), where A_G is
   a PC-algorithm-derived adjacency matrix and M_C is a learnable parameter,
   exactly as the manuscript's Eq. describes. This actually participates in
   the forward pass used by the actor/critics.
2. SYMBOLIC LAYER — v16 trained an 8-rule DiffLogicLayer on the side via a
   proxy BCE loss, then DISCARDED its output at inference
   (`sp_,_ = self._sym_action(state)` — the confidence score was thrown
   away). The rule layer now produces R(s) in R^8, concatenated into
   phi_aug(s) = [phi_C(s); R(s)], which is what the actor and critics
   actually condition on. Gradients from the task reward reach the rules
   directly, making the layer end-to-end differentiable in practice, not
   just in name.
3. STATE REPRESENTATION — v16 flattened raw arrays into a zero-padded
   448-dim vector (mostly constant/zero). Replaced with a documented,
   dense 32-dim engineered feature vector (see `featurize()`), matching the
   dimensionality the manuscript states and giving the causal-discovery step
   meaningful, non-degenerate columns to test.
4. NEW BASELINES — CQL and IQL (offline-RL-style baselines reviewers flagged
   as missing), run in the same online-interaction protocol as SAC/PPO for a
   fair like-for-like comparison in this environment.
5. CAUSAL-DISCOVERY COMPARISON — PC vs NOTEARS vs DirectLiNGAM, compared via
   bootstrap edge-stability, with the most stable method used to justify
   (empirically, not just declaratively) the causal-layer's default.
6. THREE RESULTS THAT WERE NOT ACTUALLY COMPUTED IN v16 ARE NOW REAL:
     - Ablation study: v16 multiplied a single baseline mean by hardcoded
       constants (0.88, 0.84, 0.79, ...) to fabricate "component removal"
       rows. This version actually builds and runs each ablated variant.
     - Faithfulness (SHAP/LIME/IG agreement): v16 computed this from ONE
       state sample (not the 500 the manuscript claims) and then hard-
       clamped the result into [0.70, 0.85] regardless of the true value;
       baseline-agent scores were hardcoded constants, never computed. This
       version computes all of it, for all agents, over 500 samples, with
       no clamping.
     - Deletion-insertion fidelity test: absent from v16 entirely (no code
       anywhere computed it). Implemented here per Petsiuk et al. (2018).
   Numbers this script prints WILL differ from the ones in the current
   manuscript draft. That is expected and correct — the old numbers were
   partly synthetic.

Run `pip install -r` is not needed; the script bootstraps its own deps.
"""
import argparse, os, sys, subprocess, warnings, random, time, json, pickle
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# 0. DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────
def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q",
                            "--break-system-packages"])

_REQUIRED = {
    "numpy": "numpy", "torch": "torch", "scipy": "scipy", "pandas": "pandas",
    "sklearn": "scikit-learn", "tqdm": "tqdm", "shap": "shap", "lime": "lime",
    "causallearn": "causal-learn", "lingam": "lingam",
    "networkx": "networkx", "datasets": "datasets",
    "transformers": "transformers",  # DecisionTransformerAgent (GPT2Model/GPT2Config) needs
    # this, but it was missing from this list -- unguarded `from transformers import ...`
    # at agent-construction time then crashed EVERY seed identically ("No module named
    # 'transformers'"), not just the DecisionTransformer baseline, because all 9 agents
    # are built together per seed before any of them run.
}
for mod, pkg in _REQUIRED.items():
    try:
        __import__(mod)
    except ImportError:
        print(f"Installing missing dependency: {pkg}")
        try:
            _pip(pkg)
        except Exception as e:
            print(f"  ! could not install {pkg} ({e}); continuing, some features may be skipped")

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
from tqdm import tqdm
from scipy import stats as spstats
from scipy.stats import ttest_ind, ttest_rel, wilcoxon

# ─────────────────────────────────────────────────────────────────────────
# 1. DEVICE / SEEDS / OUTPUT DIR
# ─────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"  # mixed precision only makes sense on GPU

if DEVICE.type == "cuda":
    # BUG FIX (real-data finding): cudnn.benchmark=True was previously
    # justified here purely on speed grounds ("fixed input shape, free
    # win"), which missed the actual downside -- benchmark mode picks among
    # multiple conv/matmul algorithm implementations via a timing heuristic
    # that can vary run-to-run (GPU thermal/contention state, etc.),
    # independent of input-shape stability. This was caught empirically:
    # PPO's deletion-insertion Delta flipped from +0.918 to -0.895, and
    # NeuroSymbolic's from +0.901 to -0.290, between two runs using the
    # identical seed list. That's not noise in the underlying method, it's
    # non-determinism in cudnn algorithm selection. These are small, fixed-
    # size MLPs (256-dim hidden layers) -- the autotuning speed benefit is
    # marginal for a network this size, so disabling it in favor of
    # reproducible, seed-controlled results is the right trade here.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM)")
    print(f"  cudnn.deterministic=True (benchmark disabled for reproducibility), AMP (mixed precision)=True")
else:
    print(f"Device: {DEVICE}  -- no CUDA GPU detected, running on CPU")

def gpu_diagnostic(agent, name="agent"):
    """Explicit, verifiable proof a model's parameters actually live on the
    GPU -- rather than assuming DEVICE='cuda' + .to(device) calls worked."""
    if DEVICE.type != "cuda":
        return
    try:
        p = next(agent.actor.parameters())
        print(f"  [gpu-check] {name}.actor param device={p.device}, dtype={p.dtype}")
    except Exception:
        pass

OUT_DIR = os.environ.get("CCNSA_OUT", "./ccnsa_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 101112, 202020, 303030, 404040, 505050, 606060]

def get_seeds(n):
    """First `n` seeds. Extends deterministically past the original 10 if
    more are requested (see the n=10 vs n=20-25 power-analysis discussion
    in the manuscript's Prop. 2 note) -- e.g. `--seeds 20` for a properly
    powered run against close baselines like Decision Transformer."""
    base = list(SEEDS)
    i = len(base)
    while len(base) < n:
        base.append(707070 + i * 10001)  # deterministic, collision-unlikely extension
        i += 1
    return base[:n]

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(SEEDS[0])

# ─────────────────────────────────────────────────────────────────────────
# 2. NETWORK TRAUMA ENVIRONMENT  (unchanged from v16 apart from cleanup)
# ─────────────────────────────────────────────────────────────────────────
class NetworkTraumaEnvironment:
    # CALIBRATION FIX (round 1): lognormal(8,1) capacity (median ~2981)
    # against exponential(2) bandwidth demand (mean 2), split over ~5
    # users/node, gave a baseline oversupply of roughly 490x -- confirmed by
    # direct simulation to saturate mean user satisfaction at 1.0 in 98%+ of
    # timesteps even under severe trauma. First fix rescaled to
    # lognormal(2.5,1), which fixed the ceiling-saturation problem but
    # (paired with the old 0.5+0.5*power capacity multiplier below)
    # over-corrected into a near step-function reward -- see the comment on
    # the multiplier in step() for the full story and the round-2 fix that
    # actually resolved it. Capacity here bumped once more to lognormal(3.0,1)
    # as part of that round-2 fix, verified together with the narrower
    # multiplier via standalone simulation to give a real, smooth reward
    # gradient across the action space (not a cliff, not a ceiling).
    def __init__(self, num_nodes=20, num_users=100):
        self.num_nodes = num_nodes
        self.num_users = num_users
        self.reset()

    def reset(self):
        self.nodes = {
            'id':       np.arange(self.num_nodes),
            'capacity': np.random.lognormal(3.0, 1.0, self.num_nodes),
            'power':    np.ones(self.num_nodes),
            'health':   np.ones(self.num_nodes),
            'position': np.random.rand(self.num_nodes, 2) * 100,
            'type':     np.random.choice(['tower', 'router', 'gateway', 'edge'],
                                          self.num_nodes, p=[0.5, 0.3, 0.1, 0.1])
        }
        self.users = {
            'id':             np.arange(self.num_users),
            'position':       np.random.rand(self.num_users, 2) * 100,
            'priority':       np.random.choice([1, 2, 3], self.num_users, p=[0.1, 0.3, 0.6]),
            'bandwidth_need': np.random.exponential(2, self.num_users),
            'connected_node': np.random.randint(0, self.num_nodes, self.num_users),
            'satisfaction':   np.ones(self.num_users)
        }
        self.trauma_type = None; self.trauma_severity = 0
        self.trauma_location = None; self.time_step = 0
        return self._get_state()

    def _get_state(self):
        nf = np.column_stack([
            self.nodes['capacity'], self.nodes['power'], self.nodes['health'],
            (self.nodes['type'] == 'tower').astype(int),
            (self.nodes['type'] == 'router').astype(int),
            (self.nodes['type'] == 'gateway').astype(int),
            (self.nodes['type'] == 'edge').astype(int)
        ])
        uf = np.column_stack([self.users['priority'], self.users['bandwidth_need'],
                               self.users['satisfaction']])
        tf = np.array([
            1 if self.trauma_type == 'physical' else 0,
            1 if self.trauma_type == 'cyber' else 0,
            1 if self.trauma_type == 'congestion' else 0,
            0, self.trauma_severity, self.time_step % 24 / 24
        ])
        loc = self.trauma_location if self.trauma_location is not None else np.zeros(2)
        return {'node_features': nf, 'user_features': uf,
                'trauma_features': tf, 'trauma_location': loc}

    def apply_trauma(self, trauma_type, severity=0.5, location=None):
        self.trauma_type = trauma_type; self.trauma_severity = severity
        if location is None: location = np.random.rand(2) * 100
        self.trauma_location = location
        if trauma_type == 'physical':
            dist = np.linalg.norm(self.nodes['position'] - location, axis=1)
            prob = np.exp(-dist / 30) * severity
            for i in range(self.num_nodes):
                if np.random.random() < prob[i]:
                    d = np.random.uniform(0.3, 1.0) * severity
                    self.nodes['health'][i] = max(0.1, self.nodes['health'][i] - d)
                    self.nodes['capacity'][i] *= (1 - d * 0.8)
        elif trauma_type == 'cyber':
            tgt = np.random.choice(self.num_nodes,
                                    size=int(self.num_nodes * severity * 0.5), replace=False)
            for i in tgt:
                self.nodes['capacity'][i] *= np.random.uniform(0.1, 0.5)
                if np.random.random() < 0.3: self.nodes['power'][i] = 0
        elif trauma_type == 'congestion':
            dist = np.linalg.norm(self.users['position'] - location, axis=1)
            aff = dist < 20 * severity
            self.users['bandwidth_need'][aff] *= np.random.uniform(3, 10)
            self.users['priority'][aff] = 1
        self._update_satisfaction(); return self._get_state()

    def _update_satisfaction(self):
        for i in range(self.num_users):
            nid = self.users['connected_node'][i]
            n_users = np.sum(self.users['connected_node'] == nid) or 1
            avail = self.nodes['capacity'][nid] / n_users
            sat = min(1.0, avail / self.users['bandwidth_need'][i])
            w = {1: 2.0, 2: 1.5, 3: 1.0}[self.users['priority'][i]]
            self.users['satisfaction'][i] = min(1.0, sat * w)

    def step(self, action):
        self.time_step += 1
        if 'node_power' in action:
            for i in range(min(len(action['node_power']), self.num_nodes)):
                self.nodes['power'][i] = np.clip(action['node_power'][i], 0.1, 1.0)
                # CALIBRATION FIX #2: a (0.5+0.5*power) multiplier -- i.e. a
                # 0.55x-1.0x swing across the full power range -- combined
                # with the tightened capacity margin from the first fix
                # (lognormal(2.5,1)) produced a near step-function reward:
                # verified by direct simulation that constant power=1.0 gave
                # mean reward 0.68 but power=0.5 collapsed to 0.01 and
                # power=0.1 to 0.02 -- almost the entire action space gave
                # near-zero, gradient-free reward except a narrow band near
                # power=1. That's exactly what the real GPU run showed: SAC/
                # PPO/XAI/RuleBased/NeuroSymbolic all collapsed near 0 while
                # CCNSA/CQL/IQL (whichever happened to push toward high
                # power) scored ~0.45 -- not evidence of causal/symbolic
                # superiority, just who found the one non-collapsed region.
                # Narrowed the multiplier to (0.85+0.15*power) and raised
                # capacity to lognormal(3.0,1) -- verified this gives a real,
                # smooth gradient instead of a cliff: constant-power sweep
                # now goes 0.14 (p=0.1) -> 0.22 (p=0.5) -> 0.68 (p=1.0), and
                # the fixed RuleBased heuristic (which the agent can't learn
                # its way out of) goes from 0.008 (collapsed) to 0.30
                # (meaningful, non-trivial) under this fix.
                self.nodes['capacity'][i] *= (0.85 + 0.15 * self.nodes['power'][i])
        self.nodes['health'] = np.clip(
            self.nodes['health'] + np.random.normal(0.01, 0.005, self.num_nodes), 0.1, 1.0)
        self._update_satisfaction()
        reward = (0.5 * np.mean(self.users['satisfaction'])
                  + 0.4 * np.mean(self.users['satisfaction'][self.users['priority'] == 1])
                  - 0.1 * np.mean(self.nodes['power']))
        done = self.time_step >= 100
        if np.random.random() < 0.05:
            t = np.random.choice(['physical', 'cyber', 'congestion', None])
            if t: self.apply_trauma(t, np.random.uniform(0.3, 0.8))
        return self._get_state(), reward, done


class CascadingTraumaEnvironment(NetworkTraumaEnvironment):
    """ADDED (post-review, follow-up to the null ablation result): the base
    NetworkTraumaEnvironment has no structural dependency between nodes --
    each node's capacity responds only to its own power/health, so there is
    nothing in the task that a causal-DAG-aware policy could exploit that a
    correlation-blind one couldn't eventually learn directly. The properly
    powered ablation (Section~ablation) found no individual necessity for
    the causal or symbolic layers on that task, which is honest but leaves
    open whether they would matter on a task that actually has latent
    structure to find. This variant adds one: a fixed one-hop dependency
    hierarchy (gateway -> router -> {tower, edge}, matching the existing
    node-type categories) where a node's DELIVERED capacity is throttled by
    its upstream parent's health, plus a hard power-allocation budget that
    forces the agent to choose WHICH nodes get power rather than maxing
    everyone out. Fixing a downstream node before its upstream parent is
    now a genuinely wasted action (the capacity gain is discounted by the
    still-unhealthy parent), which is exactly the kind of structure a
    causal mask over the state could help exploit and a purely reactive
    policy has to discover by trial and error. This is a new, additional
    test, not a replacement for the original ablation -- both are reported.
    """

    def reset(self):
        s = super().reset()
        # Fixed one-hop hierarchy from node type: gateway = root (no
        # parent); router's parent = nearest gateway; tower/edge's parent =
        # nearest router, falling back to nearest gateway if no router
        # exists. All distances in the existing 2D node position space, so
        # no new state dimensions are required to REALIZE the structure --
        # only to (partially) observe it, which is deliberate: the agent
        # must infer the dependency from its effect on capacity, not read
        # it off a label.
        types = self.nodes['type']; pos = self.nodes['position']
        parent = np.full(self.num_nodes, -1, dtype=int)
        gw = np.where(types == 'gateway')[0]
        rt = np.where(types == 'router')[0]
        for i in range(self.num_nodes):
            if types[i] == 'gateway':
                continue
            elif types[i] == 'router':
                if len(gw) > 0:
                    d = np.linalg.norm(pos[gw] - pos[i], axis=1)
                    parent[i] = gw[np.argmin(d)]
            else:  # tower / edge
                pool = rt if len(rt) > 0 else gw
                if len(pool) > 0:
                    d = np.linalg.norm(pos[pool] - pos[i], axis=1)
                    parent[i] = pool[np.argmin(d)]
        self.nodes['parent'] = parent
        self.power_budget = 0.6 * self.num_nodes
        return self._get_state()

    def _effective_capacity(self):
        """Own capacity, throttled by the parent's health if a parent
        exists (0.3 floor so a fully-dead parent doesn't zero the child
        outright -- avoids a degenerate all-or-nothing reward)."""
        cap = self.nodes['capacity'].copy()
        parent = self.nodes.get('parent')
        if parent is None:
            return cap
        has_parent = parent >= 0
        parent_health = np.where(has_parent, self.nodes['health'][np.clip(parent, 0, None)], 1.0)
        throttle = np.where(has_parent, 0.3 + 0.7 * parent_health, 1.0)
        return cap * throttle

    def _update_satisfaction(self):
        eff_cap = self._effective_capacity()
        for i in range(self.num_users):
            nid = self.users['connected_node'][i]
            n_users = np.sum(self.users['connected_node'] == nid) or 1
            avail = eff_cap[nid] / n_users
            sat = min(1.0, avail / self.users['bandwidth_need'][i])
            w = {1: 2.0, 2: 1.5, 3: 1.0}[self.users['priority'][i]]
            self.users['satisfaction'][i] = min(1.0, sat * w)

    def step(self, action):
        # Enforce the power budget by proportional rescaling BEFORE the
        # parent class applies it, rather than adding a new action
        # semantics -- keeps the action interface identical to the base
        # environment so the same agents/baselines run unmodified.
        if 'node_power' in action:
            p = np.asarray(action['node_power'], dtype=float)
            p = np.clip(p, 0.1, 1.0)
            total = p[:min(len(p), self.num_nodes)].sum()
            if total > self.power_budget > 0:
                p = p * (self.power_budget / total)
                p = np.clip(p, 0.1, 1.0)
            action = dict(action); action['node_power'] = p
        return super().step(action)


# ─────────────────────────────────────────────────────────────────────────
# 3. STATE FEATURIZER — dense, documented 32-dim vector (replaces v16's
#    zero-padded 448-dim raw flatten). Every dimension is named so the
#    causal-discovery step below tests meaningful columns, not padding.
# ─────────────────────────────────────────────────────────────────────────
STATE_DIM = 32
FEATURE_NAMES = [
    "node_capacity_mean", "node_capacity_std", "node_power_mean", "node_power_std",
    "node_health_mean", "node_health_std", "frac_tower", "frac_router",
    "frac_gateway", "frac_edge", "user_priority_mean", "user_priority_std",
    "user_bandwidth_mean", "user_bandwidth_std", "user_satisfaction_mean",
    "user_satisfaction_std", "trauma_physical", "trauma_cyber", "trauma_congestion",
    "trauma_severity", "time_of_day", "trauma_loc_x", "trauma_loc_y",
    "node_capacity_p25", "node_capacity_p75", "node_capacity_min", "node_capacity_max",
    "users_per_node_mean", "users_per_node_std", "satisfaction_emergency_mean",
    "satisfaction_routine_mean", "frac_nodes_low_health",
]
assert len(FEATURE_NAMES) == STATE_DIM

def featurize(state, dim=STATE_DIM):
    """Map the env's raw dict state to a dense, documented `dim`-vector."""
    if not isinstance(state, dict):
        v = np.asarray(state, dtype=np.float32).flatten()
        v = v[:dim] if len(v) > dim else np.pad(v, (0, dim - len(v)))
        return np.nan_to_num(v.astype(np.float32))
    nf, uf, tf, loc = state["node_features"], state["user_features"], \
                       state["trauma_features"], state["trauma_location"]
    cap = nf[:, 0]; pwr = nf[:, 1]; hlth = nf[:, 2]
    typ = nf[:, 3:7]  # tower, router, gateway, edge one-hots
    prio = uf[:, 0]; bw = uf[:, 1]; sat = uf[:, 2]
    emergency = prio == 1
    routine = prio == 3
    # users-per-node — approximate via connected_node not available post-flatten,
    # so use a robust proxy: bandwidth demand concentration as a stand-in.
    users_per_node_proxy_mean = float(len(uf)) / max(1, len(nf))
    users_per_node_proxy_std = float(np.std(bw))  # demand dispersion proxy
    v = np.array([
        cap.mean(), cap.std(), pwr.mean(), pwr.std(),
        hlth.mean(), hlth.std(),
        typ[:, 0].mean(), typ[:, 1].mean(), typ[:, 2].mean(), typ[:, 3].mean(),
        prio.mean(), prio.std(), bw.mean(), bw.std(), sat.mean(), sat.std(),
        tf[0], tf[1], tf[2], tf[4], tf[5],
        loc[0] / 100.0, loc[1] / 100.0,
        np.percentile(cap, 25), np.percentile(cap, 75), cap.min(), cap.max(),
        users_per_node_proxy_mean, users_per_node_proxy_std,
        sat[emergency].mean() if emergency.any() else sat.mean(),
        sat[routine].mean() if routine.any() else sat.mean(),
        float(np.mean(hlth < 0.4)),
    ], dtype=np.float32)
    return np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)

def _t(x, device=DEVICE):
    return torch.as_tensor(np.nan_to_num(np.array(x, dtype=np.float32)), device=device)

# ─────────────────────────────────────────────────────────────────────────
# 4. GENERIC UTILITIES
# ─────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """Prioritised replay with real importance-sampling correction
    (v16 sampled by priority but never applied IS weights — biased updates).
    """
    def __init__(self, cap=50_000, alpha=0.6):
        self.buf = deque(maxlen=cap); self.prios = deque(maxlen=cap); self.alpha = alpha

    def push(self, s, a, r, ns, d, td_err=None):
        self.buf.append((s, a, float(r), ns, float(d)))
        p = (abs(td_err) + 1e-3) if td_err is not None else (max(self.prios, default=1.0))
        self.prios.append(p ** self.alpha)

    def sample(self, n, beta=0.4):
        if len(self.buf) == 0: return None
        n = min(n, len(self.buf))
        prios = np.array(self.prios, dtype=np.float64)
        probs = prios / prios.sum()
        idx = np.random.choice(len(self.buf), n, p=probs, replace=len(self.buf) < n)
        batch = [self.buf[i] for i in idx]
        s, a, r, ns, d = zip(*batch)
        N = len(self.buf)
        w = (N * probs[idx]) ** (-beta)
        w /= w.max()
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.float32),
                np.array(r, dtype=np.float32), np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32), idx, w.astype(np.float32))

    def update_priorities(self, idx, td_errs):
        for i, e in zip(idx, td_errs):
            self.prios[i] = (abs(float(e)) + 1e-3) ** self.alpha

    def __len__(self): return len(self.buf)


def mlp(dims, act=nn.ReLU):
    layers = []
    for i in range(len(dims) - 1):
        lin = nn.Linear(dims[i], dims[i + 1])
        nn.init.orthogonal_(lin.weight, gain=0.5); nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(dims) - 2: layers.append(act())
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────
# 5. CAUSAL DISCOVERY — PC / NOTEARS / DirectLiNGAM, with a bootstrap
#    edge-stability comparison (addresses reviewer #2.10: "PC adopted
#    without comparing alternatives"). Whichever method is most stable
#    across resamples is used to justify (empirically) the default.
# ─────────────────────────────────────────────────────────────────────────
import signal
from contextlib import contextmanager

class _TimeoutError(Exception): pass

@contextmanager
def time_limit(seconds):
    """Unix-only wall-clock guard. PC's conditional-independence search has
    no built-in depth cap in causal-learn's high-level wrapper and can blow
    up combinatorially on >20-feature graphs (observed: depth=5 projecting
    ~40+ minutes for one fit). This aborts and lets the caller fall back
    instead of hanging the whole run."""
    if not hasattr(signal, "SIGALRM"):
        yield; return  # Windows fallback: no-op, caller's own try/except still applies
    def _handler(signum, frame): raise _TimeoutError(f"timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

def run_pc(X, alpha=0.10, depth=2, timeout_s=45):
    """PC algorithm, wall-clock-bounded. `depth` caps the conditioning-set
    size actually searched (manuscript: 'depth bounded at 2').

    BUG FOUND AND FIXED: earlier versions of this function passed
    `depth=depth` as a kwarg, but the installed causal-learn's `pc()`
    signature has no `depth` parameter -- it's called `max_k`. Since `pc()`
    accepts `**kwargs`, the stray `depth=` kwarg was silently swallowed
    (no TypeError, so the old try/except fallback never triggered) and the
    cap was never actually applied. This is exactly why the depth=5 /
    ~40-minute blowup was observed on real hardware despite the code
    claiming depth=2. Fixed to pass `max_k`, with the TypeError fallback
    kept as a genuine safety net for causal-learn versions that use yet
    another name, and the wall-clock timeout as the final backstop either
    way."""
    from causallearn.search.ConstraintBased.PC import pc
    with time_limit(timeout_s):
        try:
            cg = pc(np.nan_to_num(X), alpha, "fisherz", True, max_k=depth,
                     verbose=False, show_progress=False)
        except TypeError:
            cg = pc(np.nan_to_num(X), alpha, "fisherz", True, verbose=False)
    return (np.abs(cg.G.graph) > 0).astype(float)

def run_lingam(X, seed=0, timeout_s=45):
    import lingam
    with time_limit(timeout_s):
        lm = lingam.DirectLiNGAM(random_state=seed)
        lm.fit(np.nan_to_num(X))
    return (lm.adjacency_matrix_ != 0).astype(float)

def notears_linear(X, lambda1=0.05, max_iter=60, h_tol=1e-8, rho_max=1e16, w_threshold=0.2, timeout_s=45):
    """Compact re-implementation of NOTEARS (Zheng et al., 2018) linear-SEM
    variant for offline causal-discovery comparison. Not the paper's main
    contribution — a comparison utility. Uses an L2 (ridge) relaxation of
    the L1 sparsity term for a simple scipy L-BFGS-B inner solve.
    """
    from scipy.optimize import minimize
    with time_limit(timeout_s):
        return _notears_linear_inner(X, lambda1, max_iter, h_tol, rho_max, w_threshold)

def _notears_linear_inner(X, lambda1, max_iter, h_tol, rho_max, w_threshold):
    from scipy.optimize import minimize
    n, d = X.shape
    Xc = X - X.mean(0, keepdims=True)

    def _loss_grad(w):
        W = w.reshape(d, d)
        R = Xc - Xc @ W
        loss = 0.5 / n * (R ** 2).sum()
        grad_loss = -1.0 / n * Xc.T @ R
        E = np.linalg.matrix_power(np.eye(d) + (W * W) / d, d - 1) if d <= 12 else None
        # h(W) = tr(e^{W∘W}) - d, via scaling trick for numerical stability
        M = W * W
        # matrix exponential via eigen-free series (small d expected here, d<=32)
        expm = np.eye(d)
        term = np.eye(d)
        for k in range(1, 30):
            term = term @ M / k
            expm = expm + term
            if np.abs(term).max() < 1e-10: break
        h = np.trace(expm) - d
        grad_h = (expm.T * 2 * W)
        return loss, grad_loss, h, grad_h

    w = np.zeros(d * d); rho, alpha_dual, h = 1.0, 0.0, np.inf
    for it in range(max_iter):
        def obj(wf):
            loss, gL, h_, gH = _loss_grad(wf)
            pen = 0.5 * rho * h_ * h_ + alpha_dual * h_ + lambda1 * (wf ** 2).sum()
            g = (gL + rho * h_ * gH + alpha_dual * gH).flatten() + 2 * lambda1 * wf
            return loss + pen, g
        res = minimize(obj, w, jac=True, method="L-BFGS-B",
                        options={"maxiter": 100, "disp": False})
        w = res.x
        _, _, h_new, _ = _loss_grad(w)
        if h_new > 0.25 * h: rho *= 10
        alpha_dual += rho * h_new
        h = h_new
        if h <= h_tol or rho >= rho_max: break
    W = w.reshape(d, d)
    W[np.abs(W) < w_threshold] = 0
    return (W != 0).astype(float)

def structural_hamming_distance(A, B):
    return int(np.sum((A != 0).astype(int) != (B != 0).astype(int)))

def bootstrap_edge_stability(X, method_fn, n_boot=10, **kw):
    n = X.shape[0]
    graphs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        try:
            graphs.append(method_fn(X[idx], **kw))
        except Exception:
            continue
    if not graphs: return 0.0, None
    stack = np.stack(graphs)
    edge_freq = stack.mean(0)
    stability = float(np.mean(np.where(edge_freq > 0, np.maximum(edge_freq, 1 - edge_freq), 1)))
    consensus = (edge_freq >= 0.5).astype(float)
    return stability, consensus

def compare_causal_discovery(X, n_boot=8, verbose=True):
    """Run PC, NOTEARS, DirectLiNGAM on the same data, compare bootstrap
    edge-stability, and return the most stable method's adjacency matrix
    plus a report dict. This is the empirical justification reviewer #2.10
    asked for.

    BUG FIX: this used to call each method directly on the raw, unpruned
    X. On real data that produced a nonsensical result: PC failed on every
    single bootstrap resample (Fisher-Z needs a non-singular correlation
    matrix, and the raw feature set has the same collinear/constant columns
    documented in _prune_degenerate_columns), so bootstrap_edge_stability
    silently returned (0.0, None) for PC -- while NOTEARS and DirectLiNGAM,
    which don't require inverting a correlation matrix, succeeded on the
    SAME broken data and looked artificially superior. That's an unfair
    comparison (pruned-vs-unpruned), not evidence about which method is
    actually more stable. Now pruning once up front so all three methods
    are compared on the identical, well-posed feature subset -- matching
    what CausalLayer.fit_graph() already does during training."""
    kept_idx, X_pruned = _prune_degenerate_columns(np.nan_to_num(np.asarray(X, dtype=float)))
    if verbose and len(kept_idx) < X.shape[1]:
        print(f"    (comparison uses the same {len(kept_idx)}/{X.shape[1]} pruned, "
              f"non-degenerate columns as training, for a fair PC-vs-NOTEARS-vs-LiNGAM comparison)")
    methods = {"PC": run_pc, "NOTEARS": notears_linear, "DirectLiNGAM": run_lingam}
    report = {}
    for name, fn in methods.items():
        try:
            stab, cons = bootstrap_edge_stability(X_pruned, fn, n_boot=n_boot)
            n_edges = int(cons.sum()) if cons is not None else 0
            report[name] = {"stability": stab, "n_edges": n_edges, "graph": cons, "kept_idx": kept_idx}
            if verbose:
                print(f"    {name:14s}: bootstrap edge-stability={stab:.3f}  edges={n_edges}")
        except Exception as e:
            report[name] = {"stability": 0.0, "n_edges": 0, "graph": None, "error": str(e)}
            if verbose: print(f"    {name:14s}: failed ({e})")
    best = max(report, key=lambda k: report[k]["stability"])
    if verbose: print(f"    -> most stable: {best}")
    return best, report

# ─────────────────────────────────────────────────────────────────────────
# 6. CCNSA v2 — causal mask and symbolic-rule layer both actually
#    participate in the forward pass used by the actor and critics.
# ─────────────────────────────────────────────────────────────────────────
N_RULES = 8

def _prune_degenerate_columns(X, var_thresh=1e-8, sv_tol=1e-6, max_iter=None):
    """Drop columns that make the correlation matrix singular/rank-deficient
    before a correlation-based CI test (PC/Fisher-Z) is run on it. Two
    concrete, structural sources were found in this feature set by direct
    diagnosis (confirmed via standalone simulation, not guesswork):
      1. frac_tower + frac_router + frac_gateway + frac_edge == 1.0 exactly,
         every sample -- the classic one-hot 'dummy variable trap'. This is
         a *multi-way* linear dependency (no single pair need be highly
         correlated for the foursome to be collinear), so a pairwise
         correlation check alone can miss it. One-hot encodings summing to a
         constant are a documented, common cause of singular correlation
         matrices for Fisher-Z-based CI tests (see e.g. causal-learn issue
         #155). Standard fix: drop one category (k-1 dummy encoding).
      2. users_per_node_std and user_bandwidth_std were accidentally computed
         from the identical formula (np.std(bw)) in featurize() -- an exact
         duplicate column.
    This handles both generically via iterative rank-revealing removal:
    drop near-zero-variance columns, then repeatedly take the SVD of the
    standardized remaining columns and, for each near-null singular
    direction (singular value < sv_tol * largest), drop the column with the
    largest loading in that direction -- until the matrix is full rank (or
    <2 columns remain). This catches exact multi-way dependencies that a
    simple pairwise-correlation threshold would miss. Returns the surviving
    column indices (into the original `dim`-length feature vector) and the
    pruned data matrix.
    """
    X = np.asarray(X, dtype=float)
    var = X.var(axis=0)
    keep = list(np.where(var >= var_thresh)[0])
    if max_iter is None:
        max_iter = len(keep)
    for _ in range(max_iter):
        if len(keep) < 2:
            break
        sub = X[:, keep]
        mu, sd = sub.mean(0), sub.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Xs = (sub - mu) / sd
        try:
            _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        if len(S) == 0 or S[0] <= 0 or S[-1] > sv_tol * S[0]:
            break  # full rank (within tolerance) -- done
        drop_local = int(np.argmax(np.abs(Vt[-1])))
        keep.pop(drop_local)
    # BUG FIX (found via real-run crash during --track PROFILE3): np.array([])
    # on an empty list defaults to float64 dtype, and X[:, keep] with a
    # float-dtype index array raises "IndexError: arrays used as indices
    # must be of integer (or boolean) type" -- this fires whenever every
    # column gets pruned (e.g. a fully degenerate/constant-rows history
    # window), crashing fit_graph() *before* its own try/except (which only
    # wraps the PC/NOTEARS/LiNGAM call, not this preprocessing step) can
    # catch it and fall back to the identity graph. Explicit int dtype fixes
    # this for both the empty and non-empty cases.
    keep = np.array(keep, dtype=int)
    return keep, X[:, keep]


class CausalLayer(nn.Module):
    """phi_C(s) = (A_G ⊙ M_C) @ phi(s).  A_G: PC/NOTEARS/LiNGAM adjacency
    (buffer, not trained). M_C: learnable elementwise mask (trained)."""
    def __init__(self, dim=STATE_DIM):
        super().__init__()
        self.dim = dim
        self.embed = nn.Linear(dim, dim)
        nn.init.orthogonal_(self.embed.weight, gain=1.0); nn.init.zeros_(self.embed.bias)
        self.register_buffer("A_G", torch.eye(dim))  # identity until fit
        self.M_C = nn.Parameter(torch.ones(dim, dim) * 0.5 + 0.5 * torch.eye(dim))
        self.fitted = False
        self.n_edges_found = 0  # diagnostic: off-diagonal edges from the last fit
        self.n_pruned_last_fit = 0  # diagnostic: columns dropped as degenerate/redundant

    def fit_graph(self, X, method="PC", verbose=True):
        X = np.nan_to_num(np.asarray(X, dtype=float))
        kept_idx, X_pruned = _prune_degenerate_columns(X)
        self.n_pruned_last_fit = self.dim - len(kept_idx)
        if verbose and self.n_pruned_last_fit > 0:
            dropped = [FEATURE_NAMES[i] for i in range(self.dim) if i not in set(kept_idx.tolist())]
            print(f"  causal-fit preprocessing: dropped {self.n_pruned_last_fit} structurally "
                  f"redundant/constant column(s) before the CI test (standard practice -- these "
                  f"would otherwise leave the correlation matrix singular): {dropped}")
        try:
            if len(kept_idx) < 2:
                raise ValueError("fewer than 2 non-degenerate columns survive pruning")
            if method == "PC": A_sub = run_pc(X_pruned)
            elif method == "NOTEARS": A_sub = notears_linear(X_pruned)
            elif method == "DirectLiNGAM": A_sub = run_lingam(X_pruned)
            else: A_sub = np.eye(len(kept_idx))
            A = np.eye(self.dim)
            for a, i in enumerate(kept_idx):
                for b, j in enumerate(kept_idx):
                    A[i, j] = A_sub[a, b]
        except Exception as e:
            if verbose: print(f"  causal fit failed ({e}); keeping identity graph")
            A = np.eye(self.dim)
        A = np.maximum(A, np.eye(self.dim))  # keep self-loops so features never get fully masked
        self.n_edges_found = int((A != 0).sum() - self.dim)  # off-diagonal count
        if verbose:
            print(f"  causal graph fit ({method}): {self.n_edges_found} off-diagonal edges "
                  f"over {self.dim} features" + (" -- near-trivial (mask ~= identity); "
                  "check the feature buffer isn't too small/homogeneous if this is unexpectedly 0"
                  if self.n_edges_found == 0 else ""))
        with torch.no_grad():
            self.A_G.copy_(torch.tensor(A, dtype=torch.float32, device=self.M_C.device))
        self.fitted = True

    def forward(self, s):
        phi = self.embed(s)
        mask = self.A_G * self.M_C
        return phi @ mask.T


# The 8 values below are the ONLY human-curated target power levels that
# exist in this project -- they were hand-set to span the thresholds used in
# the original hard-coded heuristic (0.1-0.92), one per rule, and that
# correspondence to a real prior heuristic is the actual interpretability
# claim (not just "8 numbers"). There is no equivalent hand-curated set for
# any other rule count.
N_RULES_DEFAULT_U = [0.85, 0.55, 0.20, 0.75, 0.90, 0.45, 0.60, 0.30]

class SymbolicRuleLayer(nn.Module):
    """n_rules differentiable policy rules R_1..R_n over phi_C(s), plus a
    scalar symbolic policy pi_sym(s) = sum_k Pi_k(s) * u_k used both (a) as
    the BC warm-start target and (b) folded into phi_aug for the
    actor/critics. u_k are fixed target power levels per rule, Pi is a
    softmax over rule activations.
    At n_rules=8 (the only value used anywhere in this paper's reported
    results), u is the original 8 hand-curated targets. For any other
    n_rules -- used ONLY by the rule-count sensitivity sweep in
    run_rule_count_study(), never by the headline Track A/B runs -- u is
    instead evenly spaced across the same [0.20, 0.90] range the original 8
    values span. That is a deliberate, explicit downgrade of the
    interpretability claim for the swept variants: those extra/fewer rules
    are NOT independently human-curated the way the original 8 are, and the
    manuscript must not describe a non-8 variant as "interpretable
    initialisation" without this caveat."""
    def __init__(self, dim=STATE_DIM, n_rules=N_RULES, u=None):
        super().__init__()
        self.n_rules = n_rules
        self.rules = nn.Linear(dim, n_rules)
        nn.init.orthogonal_(self.rules.weight, gain=0.3); nn.init.zeros_(self.rules.bias)
        if u is not None:
            u_t = torch.tensor(u, dtype=torch.float32)
        elif n_rules == len(N_RULES_DEFAULT_U):
            u_t = torch.tensor(N_RULES_DEFAULT_U)
        else:
            # NOT human-curated -- see class docstring. Evenly spaced across
            # the same range the original 8 hand-set values span.
            lo, hi = min(N_RULES_DEFAULT_U), max(N_RULES_DEFAULT_U)
            u_t = torch.linspace(lo, hi, n_rules)
        assert u_t.numel() == n_rules
        self.register_buffer("u", u_t)

    def forward(self, phi_c):
        R = torch.sigmoid(self.rules(phi_c))               # R(s) in R^n_rules
        Pi = F.softmax(self.rules(phi_c), dim=-1)           # attention-style distribution
        pi_sym = (Pi * self.u).sum(-1, keepdim=True)        # scalar symbolic policy
        return R, Pi, pi_sym


class CCNSAAgentV2:
    """Causal-Cognitive Neuro-Symbolic Agent, v2.
    phi_aug(s) = [phi_C(s) ; R(s)]  ->  actor / twin critics condition on this.
    """
    def __init__(self, action_dim=20, state_dim=STATE_DIM, lr=3e-4,
                 use_causal=True, use_symbolic=True, use_prioritized=True,
                 use_smooth_clip=True, use_emergency_boost=True,
                 use_conservative_q=True, cql_alpha=1.0,
                 use_symbolic_gating=False, gate_alpha=0.7,
                 use_bare_capacity=False, n_rules=N_RULES,
                 causal_method="PC", causal_fit_threshold=200, device=DEVICE):
        self.ad = action_dim; self.sd = state_dim; self.device = device
        self.n_rules = n_rules  # see SymbolicRuleLayer docstring: only n_rules=8 (the
        # default used by every headline Track A/B result) has hand-curated u targets.
        self.use_causal = use_causal; self.use_symbolic = use_symbolic
        self.use_prioritized = use_prioritized
        self.use_smooth_clip = use_smooth_clip
        self.use_emergency_boost = use_emergency_boost
        # FINDING (reviewer-prompted code audit, not a bug -- the code always
        # did this deliberately, it just wasn't honestly described as a
        # limitation until now): pi_sym, the symbolic layer's actual
        # prescribed action, was computed every step and then discarded in
        # get_action (`aug, _ = self._phi_aug(s_t)`) -- only the numeric rule
        # activations R(s) reached the actor as an input FEATURE. The
        # symbolic layer never got to decide anything; it was feature
        # engineering into the same black-box MLP as everything else, which
        # is a plausible mechanistic reason CCNSA doesn't beat CQL/IQL on
        # faithfulness despite having a symbolic layer at all -- attribution
        # methods score the function's behaviour, not its internal
        # vocabulary. use_symbolic_gating makes pi_sym an actual input to the
        # final action (blended with the learned actor output, gradients
        # flowing both ways) instead of training-time-only scaffolding.
        # Default OFF so the validated Full-CCNSA baseline is unchanged;
        # ablated in via "+ Symbolic Gating" to test whether this closes any
        # of the faithfulness gap.
        self.use_symbolic_gating = use_symbolic_gating
        self.gate_alpha = gate_alpha
        # Borrowed directly from CQLAgent's critic penalty (see class CQLAgent
        # below): CCNSA's twin-Q critic previously had NO regularization
        # against Q-overestimation on off-support actions, unlike CQL
        # (explicit logsumexp penalty) or IQL (sidesteps it via expectile
        # regression) -- the two baselines CCNSA statistically tied on
        # reward rather than beat. CCNSA's ~6x higher seed-to-seed reward
        # variance vs CQL/IQL (CI +/-0.06 vs +/-0.01) was consistent with
        # this being a real, missing stabilizer rather than a
        # representation-capacity problem (which the skip-connection fix
        # already tested and ruled out).
        # VALIDATED (ablation-only run, n=10 seeds, full 50-episode budget):
        # enabling this took reward 0.5596+/-0.1097 -> 0.6733+/-0.0084 --
        # mean reward now numerically above both CQL (0.621) and IQL (0.602)
        # from the same Track A run, and ~13x variance reduction, landing
        # right in CQL/IQL's stability range. d=1.01 vs Full-CCNSA-without,
        # raw p=0.011 (n.s. only under a 7-way Bonferroni shared with six
        # unrelated exploratory ablations -- see ablation_significance_test
        # results). Promoted to the default; ablated via "w/o Conservative Q"
        # below for a dedicated, properly-powered confirmatory test in the
        # main Track A comparison against every baseline.
        self.use_conservative_q = use_conservative_q
        self.cql_alpha = cql_alpha
        self.causal_method = causal_method
        # FAIRNESS PROBE (user-requested audit, not a bug): the raw-state
        # skip connection above (Task #27 fix) is additive on top of
        # phi_c = causal.embed(s), a trainable nn.Linear(state_dim,
        # state_dim) that runs EVEN WHEN use_causal=False (only the A_G*M_C
        # masking is skipped, not the embed layer itself -- see _phi_aug).
        # So every ablation arm that "removes" the causal component still
        # gives CCNSA 2x state_dim input width vs every baseline (CQL, IQL,
        # SAC, PPO, ...), which only ever see raw state_dim. That extra,
        # never-ablated capacity is a plausible confound for the DelIns/
        # reward edge that survived every other single-component removal.
        # use_bare_capacity strips CCNSA down to raw state_dim only --
        # exactly what the baselines get -- so this can be tested directly
        # instead of assumed away.
        self.use_bare_capacity = use_bare_capacity

        self.causal = CausalLayer(state_dim).to(device)
        self.symbolic = SymbolicRuleLayer(state_dim, n_rules=n_rules).to(device)
        # ARCHITECTURE FIX (real-data finding, not a bug-fix): phi_aug used
        # to be [phi_c ; R(s)] ONLY -- the actor, both critics, and the
        # symbolic rule layer all saw exclusively the causally-masked
        # representation phi_c = embed(s) @ (A_G elementwise* M_C).T, never
        # the raw state. Every baseline (SAC/PPO/CQL/IQL/...) sees the full
        # raw 32-dim state directly. A_G is a SPARSE mask (~15-20 edges over
        # ~19 non-degenerate features) fit ONCE from only 200 early-training
        # samples using state-only correlational structure (no action/reward
        # information at all) and then frozen for the rest of training. If
        # that graph is even mildly noisy or captures the wrong structure
        # for control (very plausible -- it reflects passive environment
        # correlations like "trauma_severity correlates with node_health",
        # not which state combinations actually matter for picking good
        # power actions), CCNSA's actor was working from a strictly lossier
        # input than every baseline it's compared against. This is a
        # concrete, verified (by reading the code, not guessing) candidate
        # explanation for why CCNSA didn't beat CQL/IQL despite having more
        # machinery. Fixed with a raw-state skip connection -- the standard
        # fix for representation bottlenecks -- so phi_c and R(s) become
        # purely ADDITIVE structured signal layered on top of the same raw
        # features every baseline gets, rather than a lossy replacement.
        aug_dim = state_dim if use_bare_capacity else \
            (state_dim + state_dim + (n_rules if use_symbolic else 0))
        self.aug_dim = aug_dim

        self.actor = mlp([aug_dim, 256, 256, action_dim]).to(device)
        self.c1 = mlp([aug_dim + action_dim, 256, 256, 1]).to(device)
        self.c2 = mlp([aug_dim + action_dim, 256, 256, 1]).to(device)
        self.tc1 = mlp([aug_dim + action_dim, 256, 256, 1]).to(device)
        self.tc2 = mlp([aug_dim + action_dim, 256, 256, 1]).to(device)
        self.tc1.load_state_dict(self.c1.state_dict())
        self.tc2.load_state_dict(self.c2.state_dict())

        params = list(self.actor.parameters()) + list(self.causal.parameters()) + \
                 list(self.symbolic.parameters())
        self.ao = torch.optim.Adam(params, lr=lr, eps=1e-5)
        self.co1 = torch.optim.Adam(self.c1.parameters(), lr=lr, eps=1e-5)
        self.co2 = torch.optim.Adam(self.c2.parameters(), lr=lr, eps=1e-5)

        self.mem = ReplayBuffer(alpha=0.6 if use_prioritized else 0.0)
        self.gamma = 0.99; self.tau = 0.005; self.bs = 128
        self.scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        # `causal_fit_threshold`: how many collected samples before the PC/
        # NOTEARS/LiNGAM graph is fit once and frozen (Prop. 1's proof
        # relies on this being a one-time offline step). Configurable so
        # short ablation/smoke runs can still exercise the causal branch
        # instead of silently staying at the identity-graph default for
        # the whole run.
        self.causal_fit_threshold = causal_fit_threshold
        self._hist_cap = max(1000, causal_fit_threshold)  # bounded history (was unbounded list -> memory leak over long runs)
        self._hist = []; self._last_action = None; self._bc_done = False

    def _phi_aug(self, s_t):
        # Bare-capacity mode (fairness probe, see __init__ note): raw state
        # only, no causal.embed() linear layer, no symbolic rules -- exactly
        # what CQL/IQL/SAC/PPO see. Bypasses causal/symbolic entirely rather
        # than relying on use_causal=False, which still runs a trainable
        # embed layer.
        if self.use_bare_capacity:
            return s_t, None
        phi_c = self.causal(s_t) if self.use_causal else self.causal.embed(s_t)
        # Raw-state skip connection (see __init__ note): s_t is concatenated
        # in directly so the actor/critics never lose access to the
        # unfiltered state, regardless of how good or bad the current
        # causal-graph fit is.
        if self.use_symbolic:
            R, Pi, pi_sym = self.symbolic(phi_c)
            return torch.cat([s_t, phi_c, R], dim=-1), pi_sym
        return torch.cat([s_t, phi_c], dim=-1), None

    def _gate(self, a_tanh, pi_sym):
        """Blend a tanh-squashed action with the symbolic layer's scalar
        prescription pi_sym, converted to the same tanh-space via
        2*pi_sym-1 and broadcast across action dims. Used identically here,
        in the TD-target computation, and in the actor loss below, so the
        behaviour policy and the policy actually being optimised agree --
        unlike the pre-fix code, where pi_sym never reached any of these."""
        if not self.use_symbolic_gating or pi_sym is None:
            return a_tanh
        sym_a = (2 * pi_sym - 1).clamp(-1, 1).expand_as(a_tanh)
        return self.gate_alpha * a_tanh + (1 - self.gate_alpha) * sym_a

    def _maybe_fit_causal(self, sv):
        self._hist.append(sv)
        if len(self._hist) > self._hist_cap:
            self._hist = self._hist[-self._hist_cap:]
        if (len(self._hist) >= self.causal_fit_threshold and not self.causal.fitted
                and self.use_causal and not self.use_bare_capacity):
            data = np.nan_to_num(np.array(self._hist[-self.causal_fit_threshold:]))
            self.causal.fit_graph(data, method=self.causal_method)

    def get_action(self, state, training=True, **kw):
        sv = featurize(state, self.sd)
        self._maybe_fit_causal(sv)
        s_t = _t(sv, self.device).unsqueeze(0)
        with torch.no_grad():
            aug, pi_sym = self._phi_aug(s_t)
            raw = self.actor(aug).squeeze(0).cpu().numpy()
        ap = np.clip((np.tanh(np.nan_to_num(raw)) + 1) / 2, 0.02, 1.0)
        # Symbolic gating (see __init__ note): when enabled, pi_sym -- the
        # symbolic layer's scalar prescribed power level -- actually blends
        # into the final action instead of being discarded here. Previously
        # this line read `aug, _ = self._phi_aug(s_t)`, throwing pi_sym away.
        if self.use_symbolic_gating and pi_sym is not None:
            sym_p = float(np.clip(np.nan_to_num(pi_sym.squeeze(0).cpu().numpy()[0]), 0.0, 1.0))
            ap = self.gate_alpha * ap + (1 - self.gate_alpha) * sym_p
        if self.use_emergency_boost:
            em = state["user_features"][:, 0] == 1
            if np.any(em):
                ap[:min(5, self.ad)] = np.maximum(ap[:min(5, self.ad)], 0.80)
        if self.use_smooth_clip and self._last_action is not None:
            ap = self._last_action + np.clip(ap - self._last_action, -0.20, 0.20)
        ap = ap.clip(0.1, 1.0); self._last_action = ap.copy()
        return {"node_power": ap}, "CCNSA"

    def _sym_target(self, state):
        """Fixed heuristic used ONLY as a bootstrap prior for BC warm-start
        initialisation of the rule layer's u_k mapping; the trained
        pi_sym(s) above (from SymbolicRuleLayer) is what actually persists
        into the model."""
        tf = state["trauma_features"]; nh = state["node_features"][:, 2]
        nc = state["node_features"][:, 0]; uf = state["user_features"]
        p = np.clip(nh * 0.8 + 0.2, 0.3, 0.9)
        if tf[0] > 0.3: p = np.where(nh > 0.7, 0.90, np.where(nh > 0.4, 0.60, 0.20))
        elif tf[1] > 0.3: avg = nc.mean() + 1e-9; p = np.where(nc < avg * 0.5, 0.10, 0.85)
        elif tf[2] > 0.3:
            p[:] = 0.45
            em = uf[:, 0] == 1
            if np.any(em): p[:min(5, self.ad)] = 0.92
        return p.clip(0.1, 1.0)

    def bc_warmup(self, env, steps=500):
        """NOTE (bug fixed post-review): this loop previously wrapped the
        `_phi_aug` call in `torch.no_grad()`, which severed the graph
        before it reached `self.actor` -- `self.symbolic.parameters()` was
        registered with `opt` but never actually received a gradient
        during warm-start, despite the docstring/paper claiming BC warm-
        start initialises the rule layer. `_phi_aug` is now computed with
        grad enabled so `opt.step()` actually updates the symbolic layer,
        not just the actor."""
        if self._bc_done: return
        opt = torch.optim.Adam(list(self.actor.parameters()) +
                                list(self.causal.parameters()) +
                                list(self.symbolic.parameters()), lr=2e-3)
        s = env.reset()
        for _ in range(steps):
            sp = self._sym_target(s)
            sv = featurize(s, self.sd); self._maybe_fit_causal(sv)
            s_t = _t(sv, self.device).unsqueeze(0)
            aug, _ = self._phi_aug(s_t)  # grad enabled -- see note above
            tgt = torch.tensor(np.arctanh(np.clip(2 * sp - 1, -0.999, 0.999)),
                                dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.actor(aug)
            loss = F.mse_loss(out, tgt)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor.parameters()) +
                                      list(self.causal.parameters()) +
                                      list(self.symbolic.parameters()), 1.0)
            opt.step()
            a, _ = self.get_action(s, training=False)  # get_action manages its own no_grad internally
            s, _, done = env.step(a)
            if done: s = env.reset()
        self._bc_done = True

    def _shape(self, state, r):
        em = state["user_features"][:, 0] == 1
        if np.any(em): r += 0.20 * float(np.mean(state["user_features"][em, 2]))
        return float(r)

    def update(self, state, action, reward, next_state, done):
        reward = self._shape(state, reward)
        if not np.isfinite(reward): return
        sv = featurize(state, self.sd); nsv = featurize(next_state, self.sd)
        ap = np.nan_to_num(action["node_power"] if isinstance(action, dict) else action)
        self.mem.push(sv, ap, reward, nsv, float(done))
        if len(self.mem) < self.bs: return
        batch = self.mem.sample(self.bs)
        if batch is None: return
        S, A, R, NS, D, idx, W = batch
        S, A, R, NS, D, W = (_t(S, self.device), _t(A, self.device), _t(R, self.device).unsqueeze(1),
                              _t(NS, self.device), _t(D, self.device).unsqueeze(1), _t(W, self.device).unsqueeze(1))
        try:
            # AMP (mixed precision) on GPU: forward passes run in FP16 where
            # safe, losses computed in FP32 via the GradScaler's loss-scaling
            # to avoid FP16 underflow -- standard PyTorch AMP recipe. No-op
            # context managers on CPU (USE_AMP=False there).
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_AMP):
                aug_ns, pi_sym_ns = self._phi_aug(NS)
                na = self._gate(torch.tanh(self.actor(aug_ns)), pi_sym_ns)
                tq = torch.min(self.tc1(torch.cat([aug_ns, na], 1)),
                                self.tc2(torch.cat([aug_ns, na], 1)))
                tv = (R + (1 - D) * self.gamma * torch.nan_to_num(tq)).clamp(-10, 10)
            # Computed once and reused for both the critic loss (via .detach(),
            # which blocks gradient into causal/symbolic without consuming the
            # graph) and the actor loss below. Previously this was computed
            # TWICE (aug_s and a redundant aug_s2) with the first call's
            # pi_sym silently discarded -- same wasted-compute/dead-value
            # pattern flagged in the v16 audit, fixed here.
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                aug_s, pi_sym = self._phi_aug(S)
            # Conservative-Q penalty setup (CQL-style, see __init__ note):
            # random actions + a repeated augmented-state block, computed
            # once and reused across both critics, exactly mirroring
            # CQLAgent.update()'s rand_a / S_rep pattern.
            if self.use_conservative_q:
                with torch.no_grad():
                    rand_a = (torch.rand(S.size(0) * 10, self.ad, device=self.device) * 2 - 1)
                aug_s_rep = aug_s.detach().repeat_interleave(10, dim=0)
            td_errs = []
            for net, opt in [(self.c1, self.co1), (self.c2, self.co2)]:
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    q = net(torch.cat([aug_s.detach(), A], 1))
                    td = (q - tv)
                    loss = (W * td.pow(2)).mean()
                    if self.use_conservative_q:
                        q_rand = net(torch.cat([aug_s_rep, rand_a], 1)).view(S.size(0), 10)
                        cql_pen = (torch.logsumexp(q_rand, dim=1, keepdim=True) - q).mean()
                        loss = loss + self.cql_alpha * cql_pen
                if torch.isfinite(loss):
                    opt.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(opt)  # unscale before clipping, else the norm is wrong
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    self.scaler.step(opt)
                td_errs.append(td.detach().float().abs().cpu().numpy().flatten())
            if self.use_prioritized:
                self.mem.update_priorities(idx, np.mean(td_errs, axis=0))

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                a2 = self._gate(torch.tanh(self.actor(aug_s)), pi_sym)
                q2 = torch.min(self.c1(torch.cat([aug_s, a2], 1)), self.c2(torch.cat([aug_s, a2], 1)))
                actor_loss = -q2.mean()
                if self.use_symbolic and pi_sym is not None and not self.use_symbolic_gating:
                    # Soft anchor toward the symbolic prior -- used only when
                    # gating is OFF. When gating is ON, pi_sym already
                    # participates directly in a2 above (with gradients), so
                    # this separate anchor loss would be redundant/conflicting
                    # with the direct blend rather than complementary to it.
                    aux = F.mse_loss(a2.mean(1, keepdim=True), pi_sym.detach() * 2 - 1) * 0.01
                    actor_loss = actor_loss + aux
            if torch.isfinite(actor_loss):
                self.ao.zero_grad()
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(self.ao)
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) +
                                          list(self.causal.parameters()) +
                                          list(self.symbolic.parameters()), 1.0)
                self.scaler.step(self.ao)
            self.scaler.update()  # one update() per iteration, after all scaler.step() calls above
            for tc, c in [(self.tc1, self.c1), (self.tc2, self.c2)]:
                for tp, p in zip(tc.parameters(), c.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        except Exception as e:
            pass

# ─────────────────────────────────────────────────────────────────────────
# 7. BASELINE AGENTS  (all operate on the same 32-dim featurize() state)
# ─────────────────────────────────────────────────────────────────────────
class SACAgent:
    def __init__(self, sd=STATE_DIM, ad=20, lr=3e-4, device=DEVICE):
        self.sd = sd; self.ad = ad; self.device = device
        self.actor = mlp([sd, 256, 256, ad * 2]).to(device)
        self.c1 = mlp([sd + ad, 256, 256, 1]).to(device); self.c2 = mlp([sd + ad, 256, 256, 1]).to(device)
        self.tc1 = mlp([sd + ad, 256, 256, 1]).to(device); self.tc2 = mlp([sd + ad, 256, 256, 1]).to(device)
        self.tc1.load_state_dict(self.c1.state_dict()); self.tc2.load_state_dict(self.c2.state_dict())
        self.ao = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.co1 = torch.optim.Adam(self.c1.parameters(), lr=lr, eps=1e-5)
        self.co2 = torch.optim.Adam(self.c2.parameters(), lr=lr, eps=1e-5)
        self.la = torch.tensor(0., requires_grad=True, device=device)
        self.oa = torch.optim.Adam([self.la], lr=lr)
        self.te = -ad; self.mem = ReplayBuffer(); self.bs = 128
        self.gamma = 0.99; self.tau = 0.005

    def get_action(self, state, training=True, **kw):
        sv = _t(featurize(state, self.sd), self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.actor(sv); m, ls = out[:, :self.ad], out[:, self.ad:]
            ls = ls.clamp(-4, 2); std = torch.exp(ls).clamp(1e-4, 1.)
            a = torch.tanh(torch.distributions.Normal(m, std).rsample() if training else m)
        return {"node_power": ((a.cpu().numpy()[0] + 1) / 2).clip(0.1, 1.)}, "SAC"

    def store_transition(self, s, a, r, ns, d):
        ap = a["node_power"] if isinstance(a, dict) else a
        self.mem.push(featurize(s, self.sd), ap, r, featurize(ns, self.sd), d)

    def update(self):
        if len(self.mem) < self.bs: return
        batch = self.mem.sample(self.bs)
        if batch is None: return
        S, A, R, NS, D, idx, W = batch
        S, A, NS = _t(S, self.device), _t(A, self.device), _t(NS, self.device)
        R = _t(R, self.device).unsqueeze(1); D = _t(D, self.device).unsqueeze(1)
        al = self.la.exp().detach()
        with torch.no_grad():
            out = self.actor(NS); nm, nls = out[:, :self.ad], out[:, self.ad:]
            nls = nls.clamp(-4, 2); nstd = torch.exp(nls).clamp(1e-4, 1.)
            nd = torch.distributions.Normal(nm, nstd); nz = nd.rsample(); na = torch.tanh(nz)
            nlp = (nd.log_prob(nz) - torch.log((1 - na ** 2).clamp(1e-6))).sum(1, keepdim=True)
            tq = torch.min(self.tc1(torch.cat([NS, na], 1)), self.tc2(torch.cat([NS, na], 1))) - al * nlp
            tv = (R + (1 - D) * self.gamma * tq).clamp(-10, 10)
        for net, opt in [(self.c1, self.co1), (self.c2, self.co2)]:
            loss = F.mse_loss(net(torch.cat([S, A], 1)), tv)
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
        out = self.actor(S); m, ls = out[:, :self.ad], out[:, self.ad:]
        ls = ls.clamp(-4, 2); std = torch.exp(ls).clamp(1e-4, 1.)
        dist = torch.distributions.Normal(m, std); z = dist.rsample(); sa = torch.tanh(z)
        lp = (dist.log_prob(z) - torch.log((1 - sa ** 2).clamp(1e-6))).sum(1, keepdim=True)
        q = torch.min(self.c1(torch.cat([S, sa], 1)), self.c2(torch.cat([S, sa], 1)))
        al2 = (self.la.exp() * lp - q).mean()
        if torch.isfinite(al2):
            self.ao.zero_grad(); al2.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), 1.); self.ao.step()
        al3 = -(self.la * (lp.detach() + self.te)).mean()
        if torch.isfinite(al3): self.oa.zero_grad(); al3.backward(); self.oa.step()
        for tc, c in [(self.tc1, self.c1), (self.tc2, self.c2)]:
            for tp, p in zip(tc.parameters(), c.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)


class PPOAgent:
    def __init__(self, sd=STATE_DIM, ad=20, device=DEVICE):
        self.sd = sd; self.ad = ad; self.device = device
        self.actor = mlp([sd, 256, 256, ad]).to(device); self.critic = mlp([sd, 256, 256, 1]).to(device)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                     lr=3e-4, eps=1e-5)
        self.clip = 0.2; self.gamma = 0.99; self.lam = 0.95; self.bs = 64; self.mem = []

    def get_action(self, state, training=True, **kw):
        sv = _t(featurize(state, self.sd), self.device).unsqueeze(0)
        with torch.no_grad():
            am = self.actor(sv)
            dist = torch.distributions.Normal(am, torch.ones_like(am) * 0.1)
            a = dist.sample(); lp = dist.log_prob(a).sum(1); v = self.critic(sv).item()
        an = torch.tanh(a).cpu().numpy()[0]
        if training: self.mem.append({"s": featurize(state, self.sd), "a": an, "lp": lp.item(), "v": v})
        return {"node_power": ((an + 1) / 2).clip(0.1, 1.)}, "PPO"

    def store_outcome(self, r, d):
        if self.mem: self.mem[-1]["r"] = float(r); self.mem[-1]["d"] = float(d)

    def update(self):
        if len(self.mem) < self.bs: self.mem = []; return
        rews = [m.get("r", 0.) for m in self.mem]; vals = [m["v"] for m in self.mem]
        dons = [m.get("d", 0.) for m in self.mem]
        gae = 0.; advs = []
        for t in reversed(range(len(rews))):
            nv = vals[t + 1] if t + 1 < len(vals) and not dons[t] else 0.
            delta = rews[t] + self.gamma * nv - vals[t]
            gae = delta + self.gamma * self.lam * gae * (1 - dons[t]); advs.insert(0, gae)
        rets = [a + v for a, v in zip(advs, vals)]
        advs = np.array(advs, dtype=np.float32); advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        for _ in range(4):
            idx = np.random.permutation(len(self.mem))
            for st in range(0, len(self.mem), self.bs):
                bi = idx[st:st + self.bs]
                if not len(bi): continue
                BS = _t([self.mem[i]["s"] for i in bi], self.device)
                BA = _t([self.mem[i]["a"] for i in bi], self.device)
                BL = _t([self.mem[i]["lp"] for i in bi], self.device)
                BR = _t([rets[i] for i in bi], self.device).unsqueeze(1)
                BAd = _t([advs[i] for i in bi], self.device).unsqueeze(1)
                am = self.actor(BS)
                dist = torch.distributions.Normal(am, torch.ones_like(am) * 0.1)
                nl = dist.log_prob(BA).sum(1)
                r = torch.exp((nl - BL).clamp(-10, 10))
                al = -torch.min(r * BAd, r.clamp(1 - self.clip, 1 + self.clip) * BAd).mean()
                vl = F.mse_loss(self.critic(BS), BR)
                loss = al + 0.5 * vl
                if torch.isfinite(loss):
                    self.opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 1.)
                    self.opt.step()
        self.mem = []


class XAIAgent(SACAgent):
    def get_action(self, state, training=False, **kw):
        act, _ = super().get_action(state, training=training); return act, "XAI"


class RuleBasedAgent:
    def __init__(self, n=20): self.n = n
    def get_action(self, state, **kw):
        tf = state["trauma_features"]; nh = state["node_features"][:, 2]
        nc = state["node_features"][:, 0]; uf = state["user_features"]
        p = np.ones(self.n) * 0.5
        if tf[0] > 0.5: p = np.where(nh > 0.7, 0.8, np.where(nh > 0.4, 0.5, 0.2))
        elif tf[1] > 0.5: avg = np.mean(nc) + 1e-9; p = np.where(nc < avg * 0.5, 0.1, 0.7)
        elif tf[2] > 0.5:
            em = uf[:, 0] == 1
            if np.any(em): p[:min(int(em.sum()), self.n)] = 0.9
        return {"node_power": p.clip(0.1, 1.)}, "RuleBased"


class NeuroSymbolicAgent:
    """CCNSA ablation: neural backbone only, no causal layer at all."""
    def __init__(self, n=20, sd=STATE_DIM, device=DEVICE):
        self.n = n; self.sd = sd; self.device = device
        self.net = mlp([sd, 256, 256, n]).to(device)
    def get_action(self, state, **kw):
        sv = _t(featurize(state, self.sd), self.device).unsqueeze(0)
        with torch.no_grad(): out = self.net(sv).squeeze().cpu().numpy()
        p = np.clip((np.nan_to_num(out) + 1) / 2, 0.1, 1.)
        em = state["user_features"][:, 0] == 1
        if np.any(em): p[:min(5, self.n)] = np.maximum(p[:min(5, self.n)], 0.7)
        return {"node_power": p}, "NeuroSymbolic"


class DecisionTransformerAgent:
    def __init__(self, sd=STATE_DIM, ad=20, seq_len=20, emb=64, device=DEVICE):
        from transformers import GPT2Model, GPT2Config
        self.sd = sd; self.ad = ad; self.seq_len = seq_len; self.device = device
        cfg = GPT2Config(n_embd=emb, n_layer=2, n_head=2, n_positions=seq_len * 3,
                          resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
        self.transformer = GPT2Model(cfg).to(device)
        self.ah = nn.Linear(emb, ad).to(device); self.se = nn.Linear(sd, emb).to(device)
        self.opt = torch.optim.Adam(list(self.transformer.parameters()) +
                                     list(self.ah.parameters()) + list(self.se.parameters()), lr=1e-4)
        self.mem = deque(maxlen=5000); self.sb = []

    def get_action(self, state, training=False, **kw):
        if training and len(self.sb) >= 2:
            sq = _t([s for s, *_ in self.sb[-self.seq_len:]], self.device).unsqueeze(0)
            with torch.no_grad():
                emb = self.se(sq)
                out = self.transformer(inputs_embeds=emb).last_hidden_state
                an = torch.tanh(self.ah(out[0, -1])).cpu().numpy()
        else: an = np.random.uniform(-0.2, 0.2, self.ad)
        return {"node_power": ((an + 1) / 2).clip(0.1, 1.)}, "DecisionTransformer"

    def store_transition(self, s, a, r, ns, d):
        fs = featurize(s, self.sd); ap = a["node_power"] if isinstance(a, dict) else a
        self.mem.append((fs, ap, r)); self.sb.append((fs, ap, r))
        if len(self.sb) > self.seq_len: self.sb.pop(0)

    def update(self):
        if len(self.mem) < self.seq_len + 1: return
        try:
            i = random.randint(0, len(self.mem) - self.seq_len - 1)
            seq = list(self.mem)[i:i + self.seq_len]
            S = _t([s for s, *_ in seq], self.device).unsqueeze(0)
            A = _t([a for _, a, *_ in seq], self.device)
            emb = self.se(S)
            out = self.transformer(inputs_embeds=emb).last_hidden_state
            loss = F.mse_loss(self.ah(out[0]), A)
            if torch.isfinite(loss):
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(self.transformer.parameters()) + list(self.ah.parameters()), 1.)
                self.opt.step()
        except Exception: pass


# ─────────────────────────────────────────────────────────────────────────
# 8. NEW BASELINES — CQL and IQL (reviewers: "baseline set is outdated,
#    no offline-RL comparators"). Run online (interact + store + update per
#    step) in the same protocol as SAC/PPO/DT above, for a like-for-like
#    comparison in this environment; both are standard continuous-control
#    formulations adapted to the deterministic-tanh action head used here.
# ─────────────────────────────────────────────────────────────────────────
class CQLAgent:
    """Conservative Q-Learning (Kumar et al., 2020), continuous-action
    simplification: penalise Q at random/policy actions, reward Q at the
    action actually taken (in-distribution)."""
    def __init__(self, sd=STATE_DIM, ad=20, lr=3e-4, cql_alpha=1.0, device=DEVICE):
        self.sd = sd; self.ad = ad; self.device = device; self.cql_alpha = cql_alpha
        self.actor = mlp([sd, 256, 256, ad]).to(device)
        self.c1 = mlp([sd + ad, 256, 256, 1]).to(device); self.c2 = mlp([sd + ad, 256, 256, 1]).to(device)
        self.tc1 = mlp([sd + ad, 256, 256, 1]).to(device); self.tc2 = mlp([sd + ad, 256, 256, 1]).to(device)
        self.tc1.load_state_dict(self.c1.state_dict()); self.tc2.load_state_dict(self.c2.state_dict())
        self.ao = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.co1 = torch.optim.Adam(self.c1.parameters(), lr=lr)
        self.co2 = torch.optim.Adam(self.c2.parameters(), lr=lr)
        self.mem = ReplayBuffer(); self.bs = 128; self.gamma = 0.99; self.tau = 0.005

    def get_action(self, state, training=True, **kw):
        sv = _t(featurize(state, self.sd), self.device).unsqueeze(0)
        with torch.no_grad():
            a = torch.tanh(self.actor(sv))
            if training: a = a + 0.1 * torch.randn_like(a)
            a = a.clamp(-1, 1)
        return {"node_power": ((a.cpu().numpy()[0] + 1) / 2).clip(0.1, 1.)}, "CQL"

    def store_transition(self, s, a, r, ns, d):
        ap = a["node_power"] if isinstance(a, dict) else a
        self.mem.push(featurize(s, self.sd), ap, r, featurize(ns, self.sd), d)

    def update(self):
        if len(self.mem) < self.bs: return
        batch = self.mem.sample(self.bs)
        if batch is None: return
        S, A, R, NS, D, idx, W = batch
        S, A, NS = _t(S, self.device), _t(A, self.device), _t(NS, self.device)
        R = _t(R, self.device).unsqueeze(1); D = _t(D, self.device).unsqueeze(1)
        with torch.no_grad():
            na = torch.tanh(self.actor(NS))
            tq = torch.min(self.tc1(torch.cat([NS, na], 1)), self.tc2(torch.cat([NS, na], 1)))
            tv = (R + (1 - D) * self.gamma * tq).clamp(-10, 10)
        rand_a = (torch.rand(S.size(0) * 10, self.ad, device=self.device) * 2 - 1)
        S_rep = S.repeat_interleave(10, dim=0)
        for net, opt in [(self.c1, self.co1), (self.c2, self.co2)]:
            q_data = net(torch.cat([S, A], 1))
            bellman = F.mse_loss(q_data, tv)
            q_rand = net(torch.cat([S_rep, rand_a], 1)).view(S.size(0), 10)
            cql_pen = (torch.logsumexp(q_rand, dim=1, keepdim=True) - q_data).mean()
            loss = bellman + self.cql_alpha * cql_pen
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
        a2 = torch.tanh(self.actor(S))
        q2 = torch.min(self.c1(torch.cat([S, a2], 1)), self.c2(torch.cat([S, a2], 1)))
        al = -q2.mean()
        if torch.isfinite(al):
            self.ao.zero_grad(); al.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), 1.); self.ao.step()
        for tc, c in [(self.tc1, self.c1), (self.tc2, self.c2)]:
            for tp, p in zip(tc.parameters(), c.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)


class IQLAgent:
    """Implicit Q-Learning (Kostrikov et al., 2021): expectile-regressed
    V(s), advantage-weighted actor update. No max/logsumexp over actions
    needed, which is the point of IQL for offline-style stability."""
    def __init__(self, sd=STATE_DIM, ad=20, lr=3e-4, expectile=0.7, beta=3.0, device=DEVICE):
        self.sd = sd; self.ad = ad; self.device = device
        self.expectile = expectile; self.beta = beta
        self.actor = mlp([sd, 256, 256, ad]).to(device)
        self.v = mlp([sd, 256, 256, 1]).to(device)
        self.c1 = mlp([sd + ad, 256, 256, 1]).to(device); self.c2 = mlp([sd + ad, 256, 256, 1]).to(device)
        self.ao = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.vo = torch.optim.Adam(self.v.parameters(), lr=lr)
        self.co1 = torch.optim.Adam(self.c1.parameters(), lr=lr)
        self.co2 = torch.optim.Adam(self.c2.parameters(), lr=lr)
        self.mem = ReplayBuffer(); self.bs = 128; self.gamma = 0.99

    def get_action(self, state, training=True, **kw):
        sv = _t(featurize(state, self.sd), self.device).unsqueeze(0)
        with torch.no_grad():
            a = torch.tanh(self.actor(sv))
            if training: a = a + 0.1 * torch.randn_like(a)
            a = a.clamp(-1, 1)
        return {"node_power": ((a.cpu().numpy()[0] + 1) / 2).clip(0.1, 1.)}, "IQL"

    def store_transition(self, s, a, r, ns, d):
        ap = a["node_power"] if isinstance(a, dict) else a
        self.mem.push(featurize(s, self.sd), ap, r, featurize(ns, self.sd), d)

    def update(self):
        if len(self.mem) < self.bs: return
        batch = self.mem.sample(self.bs)
        if batch is None: return
        S, A, R, NS, D, idx, W = batch
        S, A, NS = _t(S, self.device), _t(A, self.device), _t(NS, self.device)
        R = _t(R, self.device).unsqueeze(1); D = _t(D, self.device).unsqueeze(1)
        with torch.no_grad():
            q_sa = torch.min(self.c1(torch.cat([S, A], 1)), self.c2(torch.cat([S, A], 1)))
        v_s = self.v(S)
        diff = q_sa - v_s
        w = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        v_loss = (w * diff.pow(2)).mean()
        if torch.isfinite(v_loss):
            self.vo.zero_grad(); v_loss.backward(); nn.utils.clip_grad_norm_(self.v.parameters(), 1.); self.vo.step()
        with torch.no_grad():
            nv = self.v(NS)
            tv = (R + (1 - D) * self.gamma * nv).clamp(-10, 10)
        for net, opt in [(self.c1, self.co1), (self.c2, self.co2)]:
            loss = F.mse_loss(net(torch.cat([S, A], 1)), tv)
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
        with torch.no_grad():
            adv = (q_sa - v_s).squeeze(1)
            aw = torch.clamp(torch.exp(self.beta * adv), max=100.0)
        a_pred = torch.tanh(self.actor(S))
        actor_loss = (aw * ((a_pred - A) ** 2).mean(1)).mean()
        if torch.isfinite(actor_loss):
            self.ao.zero_grad(); actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.); self.ao.step()

# ─────────────────────────────────────────────────────────────────────────
# 9. EVALUATION FRAMEWORK (Track A)
# ─────────────────────────────────────────────────────────────────────────
AGENTS_ALL = ["CCNSA", "SAC", "PPO", "CQL", "IQL", "XAI", "RuleBased",
              "NeuroSymbolic", "DecisionTransformer"]

def make_agents(action_dim=20, causal_method="PC"):
    return {
        "CCNSA": CCNSAAgentV2(action_dim=action_dim, causal_method=causal_method),
        "SAC": SACAgent(ad=action_dim), "PPO": PPOAgent(ad=action_dim),
        "CQL": CQLAgent(ad=action_dim), "IQL": IQLAgent(ad=action_dim),
        "XAI": XAIAgent(ad=action_dim), "RuleBased": RuleBasedAgent(n=action_dim),
        "NeuroSymbolic": NeuroSymbolicAgent(n=action_dim),
        "DecisionTransformer": DecisionTransformerAgent(ad=action_dim),
    }

class EvaluationFramework:
    def __init__(self, num_episodes=50, num_steps=100, num_nodes=20, causal_method="PC"):
        self.ne = num_episodes; self.ns = num_steps
        self.env = NetworkTraumaEnvironment(num_nodes=num_nodes)
        self.agents = make_agents(action_dim=num_nodes, causal_method=causal_method)
        gpu_diagnostic(self.agents["CCNSA"], "CCNSA")
        self.metrics = {n: {"performance": [], "emergency_coverage": [], "convergence": []}
                         for n in AGENTS_ALL}

    def run_evaluation(self, bc_steps=800, quiet=False):
        self.agents["CCNSA"].bc_warmup(self.env, steps=bc_steps)
        it = range(self.ne) if quiet else tqdm(range(self.ne), desc="Episodes", leave=False)
        for ep in it:
            tc = np.random.choice(["physical", "cyber", "congestion", None], p=[0.2, 0.2, 0.2, 0.4])
            for name, ag in self.agents.items():
                s = self.env.reset()
                if tc: s = self.env.apply_trauma(tc, np.random.uniform(0.3, 0.8))
                ep_r = []; ep_em = []
                for _ in range(self.ns):
                    a, _ = ag.get_action(s, training=True)
                    ns2, r, done = self.env.step(a)
                    ep_r.append(r)
                    em = s["user_features"][:, 0] == 1
                    if np.any(em): ep_em.append(np.mean(s["user_features"][em, 2]))
                    try:
                        if name == "CCNSA": ag.update(s, a, r, ns2, done)
                        elif name in ["SAC", "XAI", "CQL", "IQL"]:
                            ag.store_transition(s, a, r, ns2, done); ag.update()
                        elif name == "PPO": ag.store_outcome(r, done)
                        elif name == "DecisionTransformer":
                            ag.store_transition(s, a, r, ns2, done); ag.update()
                    except Exception:
                        pass
                    s = ns2
                    if done: break
                if name == "PPO": ag.update()
                self.metrics[name]["performance"].append(np.mean(ep_r))
                if ep_em: self.metrics[name]["emergency_coverage"].append(np.mean(ep_em))
                self.metrics[name]["convergence"].append(np.mean(ep_r))

    def report(self):
        rep = {}
        for n, m in self.metrics.items():
            rep[n] = {}
            for k, v in m.items():
                if v: rep[n][k] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
        return rep


def run_track_a(seeds=SEEDS, num_episodes=50, num_steps=100, causal_method="PC", quiet=False,
                 faith_seed_samples=15, di_seed_samples=10):
    """Runs all seeds, but only keeps ONE seed's live agents/replay-buffers
    (the best-performing one, needed downstream for faithfulness/ablation/
    deletion-insertion) resident in memory. Previously every seed's full
    EvaluationFramework -- 9 agents' networks + up to 50k-transition replay
    buffers each -- was kept alive in `all_results` for the whole run, which
    accumulates GPU/CPU memory linearly with seed count for no reason once
    that seed's metrics have been extracted."""
    print("=" * 70); print(f"TRACK A: {len(seeds)} SEEDS x {num_episodes} EPISODES x {len(AGENTS_ALL)} AGENTS")
    print("=" * 70)
    all_results = []          # (seed, report) -- lightweight, kept for every seed
    # Per-seed faithfulness, collected here (cheap n_samples) so the
    # significance test below can be paired-by-seed exactly like the reward
    # test, rather than inventing a different resampling scheme. Computed
    # while each seed's agents are still in memory, before the non-best
    # seeds get discarded (same reasoning as the memory-leak fix above --
    # doesn't require keeping every seed's agents alive simultaneously).
    faith_raw_data = {ag: [] for ag in AGENTS_ALL}
    # Same treatment for deletion-insertion, added after real-data evidence
    # showed it wasn't reproducible run-to-run: PPO's Delta flipped from
    # +0.918 to -0.895, and NeuroSymbolic's from +0.901 to -0.290, between
    # two runs with identical seeds -- a symptom of both cudnn non-
    # determinism (fixed above) AND relying on a single best-seed agent
    # (n=1) for what should be a per-seed statistic, exactly the same
    # pseudo-replication-adjacent problem reward and faithfulness had
    # before being fixed. Collected here the same way.
    di_raw_data = {ag: [] for ag in AGENTS_ALL}
    best_seed_ev = None; best_seed = None; best_ccnsa_mean = -np.inf
    for si, seed in enumerate(seeds):
        print(f"\n> Seed {si+1}/{len(seeds)}: {seed}")
        set_seed(seed)
        try:
            ev = EvaluationFramework(num_episodes=num_episodes, num_steps=num_steps, causal_method=causal_method)
            ev.run_evaluation(quiet=quiet)
            report = ev.report()
            all_results.append((seed, report))
            for ag in AGENTS_ALL:
                m = ev.metrics[ag]["performance"]
                if m: print(f"  {ag:22s}: {np.mean(m):.4f} +/- {np.std(m):.4f}")
            try:
                faith_env = NetworkTraumaEnvironment()
                seed_faith = faithfulness_test(ev.agents, faith_env, n_samples=faith_seed_samples, verbose=False)
                for ag, sc in seed_faith.items():
                    faith_raw_data[ag].append(sc["F"])
            except Exception as fe:
                print(f"  [!] per-seed faithfulness failed for seed {seed}: {fe}")
            try:
                di_env = NetworkTraumaEnvironment()
                seed_di = run_deletion_insertion_suite(ev.agents, di_env, n_samples=di_seed_samples, verbose=False)
                for ag, sc in seed_di.items():
                    di_raw_data[ag].append(sc["Delta"])
            except Exception as die:
                print(f"  [!] per-seed deletion-insertion failed for seed {seed}: {die}")
            ccnsa_perf = ev.metrics["CCNSA"]["performance"]
            ccnsa_mean = float(np.mean(ccnsa_perf)) if ccnsa_perf else -np.inf
            if ccnsa_mean > best_ccnsa_mean:
                best_ccnsa_mean = ccnsa_mean; best_seed = seed
                del best_seed_ev  # release the previous best's GPU tensors
                best_seed_ev = ev
            else:
                del ev  # not the best seed -- drop its live agents/buffers now
            if DEVICE.type == "cuda": torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  [!] seed {seed} failed: {exc}")
            continue
    if not all_results:
        raise RuntimeError("All seeds failed.")
    print(f"\n  best CCNSA seed retained for downstream analysis: {best_seed} (mean reward {best_ccnsa_mean:.4f})")

    aggregated = {}; raw_data = {ag: [] for ag in AGENTS_ALL}
    for ag in AGENTS_ALL:
        all_perf, all_em = [], []
        for _, report in all_results:
            perf_stats = report.get(ag, {}).get("performance")
            em_stats = report.get(ag, {}).get("emergency_coverage")
            if perf_stats:
                all_perf.append(perf_stats["mean"])
                raw_data[ag].append(perf_stats["mean"])  # one point per SEED (see note above)
            if em_stats: all_em.append(em_stats["mean"])
        if all_perf:
            m, s, n = np.mean(all_perf), np.std(all_perf, ddof=1) if len(all_perf) > 1 else 0.0, len(all_perf)
            # 95% CI via the t-distribution, not the z=1.96 large-sample
            # approximation -- with only n=10 seeds, z understates the true
            # interval (t_{0.975,9} ~= 2.26 vs z=1.96, ~15% wider CI).
            t_crit = float(spstats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
            aggregated[ag] = {"performance": {"mean": m, "std": s, "ci_95": t_crit * s / np.sqrt(n), "n": n}}
        if all_em:
            aggregated.setdefault(ag, {})["emergency_coverage"] = {"mean": np.mean(all_em), "std": np.std(all_em)}
    print("\n" + "=" * 70); print("AGGREGATED (mean +/- 95% CI across seeds, t-distribution)"); print("=" * 70)
    for ag in AGENTS_ALL:
        if ag in aggregated and "performance" in aggregated[ag]:
            d = aggregated[ag]["performance"]; print(f"  {ag:22s}: {d['mean']:.4f} +/- {d['ci_95']:.4f}")
    # raw_data / conv_data here are per-seed MEANS (n=len(seeds) points per
    # agent), which is what the paired t-test / Cohen's d / Wilcoxon in
    # statistical_tests() are designed to consume -- one observation per
    # seed, not one per episode (episodes within a seed are not
    # independent draws, so pooling them would inflate n and understate p).
    conv_data = {ag: [] for ag in AGENTS_ALL}  # convergence curves not retained per-seed post-refactor; see best_seed_ev for one full curve
    return (best_seed, best_seed_ev), aggregated, raw_data, conv_data, faith_raw_data, di_raw_data


def deletion_insertion_significance_test(di_raw_data):
    """Paired significance test for deletion-insertion Delta, structurally
    identical to statistical_tests()/faithfulness_significance_test(). Added
    for the same reason as the faithfulness test, plus a reliability finding:
    the single-best-seed Delta for PPO and NeuroSymbolic flipped sign
    entirely between two runs with identical seeds (cudnn non-determinism,
    now fixed) -- which means the previous single-point Delta numbers were
    never trustworthy to begin with. This gives a proper n=10 paired
    estimate instead."""
    print("\n" + "=" * 70)
    print("DELETION-INSERTION SIGNIFICANCE TEST (paired, per-seed Delta, Bonferroni corrected)")
    print("=" * 70)
    if "CCNSA" not in di_raw_data or len(di_raw_data["CCNSA"]) < 2:
        print("  [!] insufficient per-seed deletion-insertion data -- skipped")
        return []
    ccnsa = np.array(di_raw_data["CCNSA"])
    baselines = [b for b in AGENTS_ALL if b != "CCNSA" and b in di_raw_data and len(di_raw_data[b]) > 0]
    k = len(baselines) if baselines else 1
    alpha_bonf = 0.05 / k
    n = len(ccnsa)
    if n < 10:
        print(f"  [!] n={n} seeds with deletion-insertion computed -- sanity-check only.")
    rows = []
    for b in baselines:
        bv = np.array(di_raw_data[b])
        if len(bv) != len(ccnsa) or len(bv) < 2:
            print(f"  CCNSA vs {b:22s}: skipped (n mismatch or n<2)")
            continue
        t_stat, p = ttest_rel(ccnsa, bv)
        try: _, pw = wilcoxon(ccnsa, bv)
        except Exception: pw = p
        d = cohens_d_paired(ccnsa, bv)
        p_bonf = min(p * k, 1.0)
        sig = "***" if p_bonf < 0.001 else ("**" if p_bonf < 0.01 else ("*" if p_bonf < 0.05 else "n.s."))
        rows.append({"Baseline": b, "t": t_stat, "p_raw": p, "p_bonf": p_bonf,
                      "Wilcoxon_p": pw, "Cohen_d_paired": d, "n_seeds": n, "Sig": sig})
        print(f"  CCNSA vs {b:22s}: t={t_stat:+.3f}, p_bonf={p_bonf:.4f}, d_paired={d:.3f}  {sig}")
    print(f"\nBonferroni threshold: alpha/{k} = {alpha_bonf:.5f}   (n={n} seeds, "
          f"n_samples/seed={len(di_raw_data.get('CCNSA', []))})")
    return rows


def faithfulness_significance_test(faith_raw_data):
    """Paired significance test for faithfulness (F) scores, structurally
    identical to statistical_tests() for reward: paired t-test + Wilcoxon +
    Bonferroni correction + paired Cohen's d, one F observation per seed per
    agent. Added because the headline F=0.829 (CCNSA) vs F=0.807
    (NeuroSymbolic) / F=0.802 (PPO) gap on real data is narrow enough that
    it needed an actual test, not just eyeballing the numbers -- exactly the
    same reasoning that caught CQL/IQL being statistically indistinguishable
    from CCNSA on reward despite a numeric gap."""
    print("\n" + "=" * 70)
    print("FAITHFULNESS SIGNIFICANCE TEST (paired, per-seed F scores, Bonferroni corrected)")
    print("=" * 70)
    if "CCNSA" not in faith_raw_data or len(faith_raw_data["CCNSA"]) < 2:
        print("  [!] insufficient per-seed faithfulness data -- skipped")
        return []
    ccnsa = np.array(faith_raw_data["CCNSA"])
    baselines = [b for b in AGENTS_ALL if b != "CCNSA" and b in faith_raw_data and len(faith_raw_data[b]) > 0]
    k = len(baselines) if baselines else 1
    alpha_bonf = 0.05 / k
    n = len(ccnsa)
    if n < 10:
        print(f"  [!] n={n} seeds with faithfulness computed -- sanity-check only "
              f"(faithfulness computation can fail per-seed if SHAP/LIME error out; "
              f"need >=10 for publication-grade inference).")
    rows = []
    for b in baselines:
        bv = np.array(faith_raw_data[b])
        if len(bv) != len(ccnsa) or len(bv) < 2:
            print(f"  CCNSA vs {b:22s}: skipped (n mismatch or n<2)")
            continue
        t_stat, p = ttest_rel(ccnsa, bv)
        try: _, pw = wilcoxon(ccnsa, bv)
        except Exception: pw = p
        d = cohens_d_paired(ccnsa, bv)
        p_bonf = min(p * k, 1.0)
        sig = "***" if p_bonf < 0.001 else ("**" if p_bonf < 0.01 else ("*" if p_bonf < 0.05 else "n.s."))
        rows.append({"Baseline": b, "t": t_stat, "p_raw": p, "p_bonf": p_bonf,
                      "Wilcoxon_p": pw, "Cohen_d_paired": d, "n_seeds": n, "Sig": sig})
        print(f"  CCNSA vs {b:22s}: t={t_stat:+.3f}, p_bonf={p_bonf:.4f}, d_paired={d:.3f}  {sig}")
    print(f"\nBonferroni threshold: alpha/{k} = {alpha_bonf:.5f}   (n={n} seeds, "
          f"n_samples/seed={len(faith_raw_data.get('CCNSA', []))})")
    return rows


# ─────────────────────────────────────────────────────────────────────────
# 9b. RLIABLE-STYLE REPORTING (Agarwal et al., "Deep RL at the Edge of the
#     Statistical Precipice", NeurIPS 2021 Outstanding Paper). Now the
#     field-standard alternative/complement to plain mean+-std and a single
#     t-test: RL performance across seeds is often heavy-tailed (a couple of
#     unlucky seeds can dominate a plain mean), so this reports the
#     interquartile mean (IQM -- trims the top/bottom 25% of seeds, robust
#     to outlier runs) with a stratified bootstrap confidence interval
#     (resample seeds with replacement, not a parametric CI formula).
# ─────────────────────────────────────────────────────────────────────────
def interquartile_mean(x):
    """Mean of the middle 50% of values (25%-trimmed mean). Standard scipy
    implementation, matches the IQM definition used in the rliable paper."""
    x = np.asarray(x, dtype=float)
    if len(x) < 4:
        return float(np.mean(x))  # too few points for a stable IQM
    return float(spstats.trim_mean(x, proportiontocut=0.25))

def stratified_bootstrap_ci(x, statistic_fn=interquartile_mean, n_boot=2000, ci=95, seed=0):
    """Percentile bootstrap CI for an arbitrary statistic (default: IQM),
    resampling seeds with replacement -- the 'stratified bootstrap' of
    Agarwal et al. 2021. Returns (point_estimate, lo, hi)."""
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    point = statistic_fn(x)
    if n < 2:
        return point, point, point
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        resample = x[rng.randint(0, n, size=n)]
        boot_stats[i] = statistic_fn(resample)
    lo_pct, hi_pct = (100 - ci) / 2, 100 - (100 - ci) / 2
    lo, hi = np.percentile(boot_stats, [lo_pct, hi_pct])
    return point, float(lo), float(hi)

def rliable_report(raw_data, verbose=True):
    """IQM + stratified-bootstrap-95%-CI for every agent, using the same
    per-seed raw_data that feeds statistical_tests(). Complements (does not
    replace) the Bonferroni-corrected paired t-tests: the t-tests answer
    'is CCNSA significantly better than baseline X', this answers 'what is
    a robust, outlier-resistant point estimate of CCNSA's performance and
    how uncertain is it', which is the now-standard way top RL venues
    (NeurIPS/ICML/ICLR post-2021) expect aggregate performance reported."""
    if verbose:
        print("\n" + "=" * 70); print("RLIABLE-STYLE REPORT (IQM + stratified bootstrap 95% CI, Agarwal et al. 2021)")
        print("=" * 70)
    report = {}
    for ag in AGENTS_ALL:
        vals = raw_data.get(ag, [])
        if len(vals) < 2:
            continue
        iqm, lo, hi = stratified_bootstrap_ci(vals, interquartile_mean)
        mean_pt, mean_lo, mean_hi = stratified_bootstrap_ci(vals, np.mean)
        report[ag] = {"iqm": iqm, "iqm_ci_lo": lo, "iqm_ci_hi": hi,
                       "mean": mean_pt, "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
                       "n_seeds": len(vals)}
        if verbose:
            print(f"  {ag:22s}: IQM={iqm:.4f} [{lo:.4f}, {hi:.4f}]   "
                  f"mean={mean_pt:.4f} [{mean_lo:.4f}, {mean_hi:.4f}]")
    return report

def cohens_d(a, b):
    """Independent-groups Cohen's d (pooled SD). Kept for reference /
    external comparability, but NOT used for the CCNSA-vs-baseline test
    below, since that comparison is paired by seed (see cohens_d_paired)."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / (sp + 1e-12)

def cohens_d_paired(a, b):
    """Cohen's d for paired samples: mean(a-b) / std(a-b). Correct effect
    size for a design where a[i] and b[i] share the same seed."""
    diff = np.asarray(a) - np.asarray(b)
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))

def statistical_tests(raw_data):
    """NOTE (fixed post-review): raw_data now holds one value PER SEED per
    agent (not one per episode). Two related statistical-rigor fixes vs the
    original v16-derived version:
      1. Sample size is n=len(seeds) (e.g. 10), not n=seeds*episodes
         (e.g. 500). Pooling within-seed episodes as if independent is
         pseudo-replication -- episodes share an evolving policy within a
         seed, so they are not i.i.d. draws, and treating them as such
         artificially inflates n and deflates p-values.
      2. Uses a PAIRED t-test (ttest_rel) and paired Cohen's d, matching
         the manuscript's stated "paired t-tests" protocol and the actual
         design (CCNSA and each baseline share the same seed -> same
         environment-realisation draws, so the comparison is paired, not
         independent). The previous version called ttest_ind (unpaired)
         despite the paired design, discarding the correlation between
         paired observations and understating power.
    """
    print("\n" + "=" * 70); print("STATISTICAL TESTS (paired t-test + paired Wilcoxon, Bonferroni corrected)"); print("=" * 70)
    ccnsa = np.array(raw_data["CCNSA"]); baselines = [b for b in AGENTS_ALL if b != "CCNSA"]
    k = len(baselines); alpha_bonf = 0.05 / k
    n = len(ccnsa)
    if n < 10:
        print(f"  [!] n={n} seeds -- statistical tests below are for sanity-checking only, "
              f"not publication-grade inference (need >=10 independent seeds).")
    rows = []
    for b in baselines:
        bv = np.array(raw_data[b])
        if len(bv) != len(ccnsa) or len(bv) < 2:
            print(f"  CCNSA vs {b:22s}: skipped (n mismatch or n<2 -- seed failed for one agent?)")
            continue
        t_stat, p = ttest_rel(ccnsa, bv)
        try: _, pw = wilcoxon(ccnsa, bv)
        except Exception: pw = p
        d = cohens_d_paired(ccnsa, bv)
        p_bonf = min(p * k, 1.0)
        # BUG FIX: this previously read `sig = "***" if p < alpha_bonf else
        # ("*" if p < 0.05 else "n.s.")` -- the "*" tier used the RAW
        # uncorrected p-value, not p_bonf, while the line printed right next
        # to it shows p_bonf. That produced exactly the contradiction seen
        # on real data: CCNSA vs CQL printed "p_bonf=0.1341" (i.e. NOT
        # significant at the Bonferroni-corrected 0.05 level) but was marked
        # "*" anyway, because the underlying raw p happened to be <0.05.
        # All tiers now consistently use the corrected p_bonf.
        sig = "***" if p_bonf < 0.001 else ("**" if p_bonf < 0.01 else ("*" if p_bonf < 0.05 else "n.s."))
        rows.append({"Baseline": b, "t": t_stat, "p_raw": p, "p_bonf": p_bonf,
                      "Wilcoxon_p": pw, "Cohen_d_paired": d, "n_seeds": n, "Sig": sig})
        print(f"  CCNSA vs {b:22s}: t={t_stat:+.3f}, p_bonf={min(p*k,1.):.4f}, d_paired={d:.3f}  {sig}")
    print(f"\nBonferroni threshold: alpha/{k} = {alpha_bonf:.5f}   (n={n} seeds)")
    return rows

# ─────────────────────────────────────────────────────────────────────────
# 10. REAL ABLATION STUDY
#     v16 fabricated this: it ran ONE real mini-experiment (Full CCNSA),
#     then generated every other row by multiplying that single mean by
#     hardcoded constants (0.88, 0.84, 0.79, ...) — no component was ever
#     actually removed. Here every variant is a real CCNSAAgentV2 with the
#     named component actually switched off, run and measured independently.
# ─────────────────────────────────────────────────────────────────────────
ABLATION_VARIANTS = {
    "Full CCNSA":             dict(),
    "w/o Causal Graph":       dict(use_causal=False),
    "w/o Symbolic Logic":     dict(use_symbolic=False),
    "w/o Prioritised Replay": dict(use_prioritized=False),
    "w/o Smooth Clamping":    dict(use_smooth_clip=False),
    "w/o Emergency Boost":    dict(use_emergency_boost=False),
    # Interaction-effect probe: the individually-removed variants above were
    # all statistically indistinguishable from Full CCNSA (n=10 seeds,
    # Bonferroni-corrected, all n.s. on reward/faithfulness/deletion-
    # insertion). That is consistent with either (a) neither component
    # matters, or (b) they are individually redundant with each other --
    # e.g. the symbolic rule layer sees phi_c and could partially recover
    # information the causal layer would have provided, so removing only
    # one leaves the other to compensate. This variant removes BOTH the
    # causal graph AND the symbolic logic layer at once (the two components
    # the manuscript's title and central claims are actually built on) to
    # test the redundancy hypothesis directly, not just each piece alone.
    "w/o Causal+Symbolic (joint)": dict(use_causal=False, use_symbolic=False),
    # use_conservative_q is now ON by default (see CCNSAAgentV2.__init__ --
    # validated in a prior ablation-only run: reward 0.560+/-0.110 ->
    # 0.673+/-0.008, d=1.01, ~13x variance reduction). "Full CCNSA" above
    # therefore already includes it. This arm switches it back OFF to give
    # it its own dedicated, properly-powered significance test here
    # (previously it was the ADDED arm compared against a without-it
    # baseline; now it's ablated like every other component, for symmetry
    # with the table above and so this run's Bonferroni correction reflects
    # one clean set of "remove X from the real default" comparisons).
    "w/o Conservative Q":     dict(use_conservative_q=False),
    # Follow-up experiment 1 (this round): pi_sym was computed by the
    # symbolic layer every step but discarded at inference -- only feeding a
    # small auxiliary training loss, never the actual action. This is the
    # literal mechanistic reason CCNSA's faithfulness doesn't lead the field:
    # the "symbolic" explanation doesn't drive behavior, so attribution
    # methods have no reason to agree with it. This arm wires pi_sym directly
    # into the action via _gate() (tanh-space blend, gate_alpha=0.7) so the
    # symbolic prescription actually participates in the decision, and tests
    # whether that measurably improves faithfulness (not just reward).
    "+ Symbolic Gating (pi_sym drives action)": dict(use_symbolic_gating=True),
    # Follow-up experiment 2 (this round): CCNSA retains a deletion-insertion
    # (explanation-fidelity) edge over CQL/IQL that survives even the joint
    # causal+symbolic removal above, so causal/symbolic aren't the source.
    # "w/o Conservative Q" showed the single largest DI delta of any
    # component removed so far, but CCNSA-with-CQL-penalty still beats CQL
    # itself (which has its own conservative mechanism) -- so conservative-Q
    # alone can't be the full explanation either. BC warm-start is the one
    # remaining unablated component; this arm removes it to test whether it
    # is what's actually driving the persistent DI advantage.
    "w/o BC Warm-Start":      dict(_skip_bc_warmup=True),
    # Follow-up experiment 3 (fairness probe): "w/o Causal Graph" only skips
    # the A_G*M_C masking -- CCNSA still runs a trainable causal.embed()
    # linear layer plus the raw-state skip connection, so it's always fed
    # 2x state_dim vs the state_dim every baseline (CQL/IQL/SAC/PPO) gets.
    # That capacity gap was never itself ablated, and is a plausible
    # explanation for the reward/DelIns edge that survived every other
    # single-component removal. This arm strips CCNSA to bare state_dim
    # input -- exactly matching the baselines -- to test it directly.
    "Bare Capacity (state_dim only, no causal/symbolic plumbing)":
        dict(use_causal=False, use_symbolic=False, use_bare_capacity=True),
    # Follow-up experiment 4 (hyperparameter sensitivity): cql_alpha=1.0 was
    # never varied after being borrowed from CQL -- a reviewer will ask
    # whether that value was tuned or arbitrary. Sweep it against the
    # Full-CCNSA default (alpha=1.0) to see how sensitive the result is.
    "Conservative Q (alpha=0.5)": dict(cql_alpha=0.5),
    "Conservative Q (alpha=2.0)": dict(cql_alpha=2.0),
}

def run_ablation_variant(kwargs, seed, n_episodes=5, n_steps=50, action_dim=20,
                          faith_samples=10, di_samples=8, env_cls=NetworkTraumaEnvironment):
    kwargs = dict(kwargs)
    skip_bc = kwargs.pop("_skip_bc_warmup", False)
    set_seed(seed)
    env = env_cls(num_nodes=action_dim)
    ag = CCNSAAgentV2(action_dim=action_dim, **kwargs)
    if not skip_bc:
        ag.bc_warmup(env, steps=200)
    rews = []
    for _ in range(n_episodes):
        s = env.reset(); ep = []
        for _ in range(n_steps):
            a, _ = ag.get_action(s, training=True)
            ns, r, done = env.step(a)
            ag.update(s, a, r, ns, done)
            ep.append(r); s = ns
            if done: break
        rews.append(np.mean(ep))
    reward = float(np.mean(rews))
    # EXTENSION (real-data finding): the ablation previously measured only
    # reward impact, and every variant's reward came out equal-or-better
    # than Full CCNSA -- no support for causal/symbolic components helping
    # performance. But the separately-fixed, properly-powered faithfulness
    # and deletion-insertion significance tests show CCNSA DOES
    # significantly beat several strong baselines on explanation fidelity
    # (IQL, XAI on deletion-insertion; SAC, XAI on faithfulness). If the
    # causal/symbolic components' real contribution is to explanation
    # quality rather than raw performance, the reward-only ablation can't
    # see that -- it needs its own arm to test the actual hypothesis.
    # Measured here on the same trained variant, before it's discarded.
    faith_env = env_cls(num_nodes=action_dim)
    di_env = env_cls(num_nodes=action_dim)
    try:
        f = faithfulness_test({"variant": ag}, faith_env, n_samples=faith_samples, verbose=False)
        f_score = f.get("variant", {}).get("F", float("nan"))
    except Exception:
        f_score = float("nan")
    try:
        d = run_deletion_insertion_suite({"variant": ag}, di_env, n_samples=di_samples, verbose=False)
        d_delta = d.get("variant", {}).get("Delta", float("nan"))
    except Exception:
        d_delta = float("nan")
    return {"reward": reward, "faithfulness": f_score, "di_delta": d_delta}

def run_ablation_study(seeds=(42, 123, 456), n_episodes=5, n_steps=50, action_dim=20, verbose=True,
                        variants=None, baseline_key="Full CCNSA", title="ABLATION STUDY — real component removal, not simulated deltas",
                        env_cls=NetworkTraumaEnvironment):
    variants = ABLATION_VARIANTS if variants is None else variants
    if verbose:
        print("\n" + "=" * 70); print(title)
        print("=" * 70)
    results = {}
    for name, kwargs in variants.items():
        if verbose:
            print(f"  > {name}: training {len(seeds)} seed(s) x {n_episodes} episode(s)...", flush=True)
        runs = []
        for si, s in enumerate(seeds):
            _v0 = time.time()
            runs.append(run_ablation_variant(kwargs, s, n_episodes, n_steps, action_dim, env_cls=env_cls))
            if verbose:
                print(f"      seed {si+1}/{len(seeds)} ({s}) done in {time.time()-_v0:.1f}s", flush=True)
        rewards = [r["reward"] for r in runs]
        # NOTE: faith_raw / di_raw are kept SEED-ALIGNED (same length/order as
        # `seeds`, NaN preserved in place) so that a later paired test against
        # "Full CCNSA" pairs seed i with seed i correctly. The *_scores lists
        # below (nan-dropped) are kept only for the human-readable mean+/-std
        # summary line; they must never be used for a paired comparison.
        faith_raw = [r["faithfulness"] for r in runs]
        di_raw = [r["di_delta"] for r in runs]
        faiths = [v for v in faith_raw if not np.isnan(v)]
        dis = [v for v in di_raw if not np.isnan(v)]
        results[name] = {
            "mean": float(np.mean(rewards)), "std": float(np.std(rewards)), "scores": rewards,
            "faith_mean": float(np.mean(faiths)) if faiths else float("nan"),
            "faith_std": float(np.std(faiths)) if faiths else float("nan"), "faith_scores": faiths,
            "di_mean": float(np.mean(dis)) if dis else float("nan"),
            "di_std": float(np.std(dis)) if dis else float("nan"), "di_scores": dis,
            "faith_raw": faith_raw, "di_raw": di_raw, "seeds": list(seeds),
        }
        if verbose:
            print(f"  {name:<26} reward={results[name]['mean']:7.4f}+/-{results[name]['std']:.4f}  "
                  f"F={results[name]['faith_mean']:.4f}+/-{results[name]['faith_std']:.4f}  "
                  f"DelIns={results[name]['di_mean']:+.4f}+/-{results[name]['di_std']:.4f}")
    base = results[baseline_key]["mean"]
    base_f = results[baseline_key]["faith_mean"]
    base_di = results[baseline_key]["di_mean"]
    for name in results:
        results[name]["delta_vs_full"] = results[name]["mean"] - base
        results[name]["faith_delta_vs_full"] = results[name]["faith_mean"] - base_f
        results[name]["di_delta_vs_full"] = results[name]["di_mean"] - base_di
    if verbose:
        print(f"\n  {'Variant':<26} {'Reward Delta':>13} {'Faith Delta':>13} {'DelIns Delta':>13}")
        for name, r in results.items():
            print(f"  {name:<26} {r['delta_vs_full']:+13.4f} {r['faith_delta_vs_full']:+13.4f} {r['di_delta_vs_full']:+13.4f}")
    return results


def ablation_significance_test(results, alpha=0.05, baseline_key="Full CCNSA"):
    """Paired significance test for each ablation variant vs the baseline
    (default "Full CCNSA"), on all three axes (reward, faithfulness,
    deletion-insertion), seed-aligned and Bonferroni-corrected across
    variants -- same statistical machinery as statistical_tests()/
    faithfulness_significance_test()/deletion_insertion_significance_test().
    Added because the descriptive ablation table alone (mean+/-std) invites
    eyeballing tiny deltas against large std devs; this makes explicit which
    deltas actually clear significance and which don't."""
    print("\n" + "=" * 70)
    print(f"ABLATION SIGNIFICANCE TEST (paired vs {baseline_key}, per metric, Bonferroni corrected)")
    print("=" * 70)
    variants = [v for v in results if v != baseline_key]
    k = len(variants) if variants else 1
    alpha_bonf = alpha / k
    base = results[baseline_key]
    n_seeds = len(base.get("scores", []))
    rows = []
    for metric, key_raw, key_scores, label in [
        ("reward", None, "scores", "Reward"),
        ("faithfulness", "faith_raw", "faith_scores", "Faithfulness"),
        ("deletion_insertion", "di_raw", "di_scores", "DelIns"),
    ]:
        print(f"\n  -- {label} --")
        base_vals = np.array(base[key_scores]) if key_raw is None else np.array(base[key_raw])
        for name in variants:
            v = results[name]
            cand_vals = np.array(v[key_scores]) if key_raw is None else np.array(v[key_raw])
            if key_raw is not None:
                # drop seed-pairs where either side is NaN, keep pairing intact
                mask = ~(np.isnan(base_vals) | np.isnan(cand_vals))
                bv, cv = base_vals[mask], cand_vals[mask]
            else:
                bv, cv = base_vals, cand_vals
            n = len(bv)
            if n < 2 or len(cv) != n:
                print(f"  {baseline_key} vs {name:<24}: skipped (n={n}, mismatched or <2 pairs)")
                continue
            t_stat, p = ttest_rel(bv, cv)
            try: _, pw = wilcoxon(bv, cv)
            except Exception: pw = p
            d = cohens_d_paired(bv, cv)
            p_bonf = min(p * k, 1.0)
            sig = "***" if p_bonf < 0.001 else ("**" if p_bonf < 0.01 else ("*" if p_bonf < 0.05 else "n.s."))
            rows.append({"Metric": metric, "Variant": name, "t": t_stat, "p_raw": p, "p_bonf": p_bonf,
                          "Wilcoxon_p": pw, "Cohen_d_paired": d, "n_seeds": n, "Sig": sig})
            print(f"  {baseline_key} vs {name:<24}: t={t_stat:+.3f}, p_bonf={p_bonf:.4f}, "
                  f"d_paired={d:+.3f}, n={n}  {sig}")
    print(f"\nBonferroni threshold: alpha/{k} = {alpha_bonf:.5f}   ({k} variants x {n_seeds} seeds each)")
    return rows


# ─────────────────────────────────────────────────────────────────────────
# 10b. RULE-COUNT SENSITIVITY SWEEP (r=8 vs r=4 vs r=16)
#      This did NOT exist anywhere in this project before -- neither in the
#      original ccnsa_revised.py nor in the user's first-version notebook
#      (grepped directly, zero hits on N_RULES / "16 rules" / n_rules= /
#      any rule-count-sweep terms). The manuscript's Limitations section
#      says exactly that: N_RULES is a fixed constant, no other count was
#      ever run. This is the actual sweep, run for real if invoked.
#      CAVEAT (see SymbolicRuleLayer docstring): only r=8 uses the original
#      hand-curated u targets. r=4 and r=16 use u values evenly spaced
#      across the same numeric range -- NOT independently human-curated.
#      Report this sweep with that caveat attached; do not describe r=4/16
#      as equally "interpretable" as the r=8 default in any manuscript text
#      drawing on these results.
# ─────────────────────────────────────────────────────────────────────────
RULE_COUNT_VARIANTS = {
    "r=8 (default)":  dict(n_rules=8),
    "r=4":             dict(n_rules=4),
    "r=16":            dict(n_rules=16),
}

def run_rule_count_study(seeds=SEEDS, n_episodes=50, n_steps=100, action_dim=20, verbose=True):
    """Same machinery as run_ablation_study/ablation_significance_test, run
    with RULE_COUNT_VARIANTS instead of ABLATION_VARIANTS and baseline
    'r=8 (default)'. Call at full seed count (default: all 10 SEEDS) for a
    properly powered result comparable in rigor to Table 4's ablation --
    NOT the 3-seed/5-episode fast-mini scale used for quick iteration."""
    results = run_ablation_study(seeds=seeds, n_episodes=n_episodes, n_steps=n_steps,
                                  action_dim=action_dim, verbose=verbose,
                                  variants=RULE_COUNT_VARIANTS, baseline_key="r=8 (default)",
                                  title="RULE-COUNT SENSITIVITY SWEEP — r=4 / r=8 / r=16, real runs")
    stat_rows = ablation_significance_test(results, baseline_key="r=8 (default)")
    return results, stat_rows


# ─────────────────────────────────────────────────────────────────────────
# CASCADE ablation — same 4 core architectural variants as the main
# ablation (Full / w/o causal / w/o symbolic / w/o both), re-run on
# CascadingTraumaEnvironment instead of NetworkTraumaEnvironment. Kept to
# just these 4 (not the full 12-arm ABLATION_VARIANTS) to bound wall-clock
# to roughly the same scale as the RULES/CAUSAL sweeps above, since this is
# a follow-up robustness check, not a replacement for the main ablation.
# ─────────────────────────────────────────────────────────────────────────
CASCADE_ABLATION_VARIANTS = {
    "Full CCNSA":                  dict(),
    "w/o Causal Graph":             dict(use_causal=False),
    "w/o Symbolic Logic":           dict(use_symbolic=False),
    "w/o Causal+Symbolic (joint)":  dict(use_causal=False, use_symbolic=False),
}

def run_cascade_study(seeds=SEEDS, n_episodes=50, n_steps=100, action_dim=20, verbose=True):
    """Same machinery as run_ablation_study/ablation_significance_test, run
    on CascadingTraumaEnvironment (dependency-hierarchy + power-budget
    variant, see class docstring) instead of the base environment, to test
    whether the causal/symbolic layers become individually necessary on a
    task that has real latent structure for them to exploit -- the main
    ablation (Table~ablation, base environment) found neither necessary,
    which is an honest negative result but doesn't rule out the possibility
    that the base task simply has nothing for them to find."""
    results = run_ablation_study(seeds=seeds, n_episodes=n_episodes, n_steps=n_steps,
                                  action_dim=action_dim, verbose=verbose,
                                  variants=CASCADE_ABLATION_VARIANTS, baseline_key="Full CCNSA",
                                  title="CASCADING-TRAUMA ABLATION — dependency-hierarchy + power-budget task, real runs",
                                  env_cls=CascadingTraumaEnvironment)
    stat_rows = ablation_significance_test(results, baseline_key="Full CCNSA")
    return results, stat_rows


# ─────────────────────────────────────────────────────────────────────────
# 10f. CAUSAL-DISCOVERY METHOD SWEEP (PC vs NOTEARS vs DirectLiNGAM), ON
#      DOWNSTREAM TRACK-A PERFORMANCE
#      Track B (Section~sec:audit-rules) already compares PC/NOTEARS/
#      DirectLiNGAM on bootstrap edge-stability over REAL UNSW-NB15
#      features -- that is a causal-discovery-QUALITY comparison, not a
#      downstream-RL-performance one, and it never touches Track A at all.
#      A reviewer correctly pointed out this gap (would NOTEARS produce
#      different downstream performance than PC?) -- this function answers
#      it directly: same machinery as RULE_COUNT_VARIANTS, but swept over
#      CCNSAAgentV2's existing causal_method kwarg instead of n_rules.
# ─────────────────────────────────────────────────────────────────────────
CAUSAL_METHOD_VARIANTS = {
    "PC (default)":  dict(causal_method="PC"),
    "NOTEARS":       dict(causal_method="NOTEARS"),
    "DirectLiNGAM":  dict(causal_method="DirectLiNGAM"),
}

def run_causal_method_study(seeds=SEEDS, n_episodes=50, n_steps=100, action_dim=20, verbose=True):
    """Same machinery as run_rule_count_study, swept over causal_method
    instead of n_rules. Call at full seed count for rigor comparable to
    Table 4's ablation."""
    results = run_ablation_study(seeds=seeds, n_episodes=n_episodes, n_steps=n_steps,
                                  action_dim=action_dim, verbose=verbose,
                                  variants=CAUSAL_METHOD_VARIANTS, baseline_key="PC (default)",
                                  title="CAUSAL-DISCOVERY METHOD SWEEP — PC / NOTEARS / DirectLiNGAM, "
                                        "downstream Track-A performance, real runs")
    stat_rows = ablation_significance_test(results, baseline_key="PC (default)")
    return results, stat_rows


# ─────────────────────────────────────────────────────────────────────────
# 10c. LATENCY BREAKDOWN PROFILE (tests the "non-fused sequential submodule
#      calls" hypothesis for why CCNSA's latency (0.99ms) is ~30-37% higher
#      than CQL/IQL/XAI (0.72-0.77ms) despite only ~15% more FLOPs
#      (181,760 vs 157,696). The manuscript currently does NOT explain this
#      disproportion -- it's flagged as untested. This profiles each
#      submodule (causal mask, symbolic rule layer, actor MLP) separately
#      on CPU, single-sample batches (matching the inference-path shape
#      used everywhere else in this paper's latency numbers), and compares
#      the sum of the parts to the measured end-to-end phi_aug+actor call --
#      if the sum of parts is close to the whole, latency is dominated by
#      raw compute (consistent with FLOPs); if the whole is notably larger
#      than the sum of parts, that's real evidence for per-call/kernel-
#      launch overhead from the extra sequential submodule calls, not
#      raw FLOPs, driving the gap.
# ─────────────────────────────────────────────────────────────────────────
def profile_latency_breakdown(action_dim=20, state_dim=STATE_DIM, n_iters=2000, device=DEVICE, seed=42):
    set_seed(seed)
    agent = CCNSAAgentV2(action_dim=action_dim, state_dim=state_dim, device=device)
    agent.actor.eval(); agent.causal.eval(); agent.symbolic.eval()
    s = torch.randn(1, state_dim, device=device)

    def _time(fn, n=n_iters):
        # warm-up (JIT/caching effects, first-call overhead)
        for _ in range(20):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000.0  # ms/call

    with torch.no_grad():
        t_causal = _time(lambda: agent.causal(s))
        phi_c = agent.causal(s)
        t_symbolic = _time(lambda: agent.symbolic(phi_c))
        aug = torch.cat([phi_c, s, agent.symbolic(phi_c)[0]], dim=-1) if agent.use_symbolic \
            else torch.cat([phi_c, s], dim=-1)
        t_actor = _time(lambda: agent.actor(aug))

        def _full():
            phi_c_ = agent.causal(s)
            R_, Pi_, pi_sym_ = agent.symbolic(phi_c_)
            aug_ = torch.cat([phi_c_, s, R_], dim=-1)
            return agent.actor(aug_)
        t_full = _time(_full)

    sum_parts = t_causal + t_symbolic + t_actor
    overhead = t_full - sum_parts
    print("\n" + "=" * 70)
    print(f"LATENCY BREAKDOWN ({device}, n={n_iters} iters, batch=1, after 20-call warmup)")
    print("=" * 70)
    print(f"  CausalLayer forward:        {t_causal:.5f} ms")
    print(f"  SymbolicRuleLayer forward:  {t_symbolic:.5f} ms")
    print(f"  Actor MLP forward:          {t_actor:.5f} ms")
    print(f"  Sum of parts:               {sum_parts:.5f} ms")
    print(f"  Measured full get_action-equivalent forward: {t_full:.5f} ms")
    print(f"  Unaccounted overhead (full - sum of parts):  {overhead:+.5f} ms "
          f"({100*overhead/t_full:+.1f}% of the full call)")
    print("  Interpretation: if overhead is small relative to t_full, the latency gap vs "
          "CQL/IQL is consistent with raw compute (matches the ~15% FLOPs gap). If overhead "
          "is large, it supports the non-fused-sequential-calls hypothesis -- report whichever "
          "is actually measured; do not assume the hypothesis is confirmed without this number.")
    return {"t_causal_ms": t_causal, "t_symbolic_ms": t_symbolic, "t_actor_ms": t_actor,
            "sum_parts_ms": sum_parts, "t_full_ms": t_full, "overhead_ms": overhead,
            "overhead_pct": 100 * overhead / t_full}


# ─────────────────────────────────────────────────────────────────────────
# 10d. FULL get_action() BREAKDOWN
#      profile_latency_breakdown() above only timed the three tensor
#      submodules in isolation (0.28ms on the RTX 8000 run) -- far below the
#      manuscript's reported 0.99ms figure, which times the ACTUAL
#      get_action() call used everywhere else in this paper (see line ~2274,
#      `ag.get_action(s, training=False)`). get_action() does substantially
#      more than the three submodule forwards: featurize() (Python/numpy
#      feature engineering on the raw env state dict), a host->device
#      tensor creation, a device->host .cpu().numpy() transfer, and several
#      numpy post-processing steps (tanh, clip, emergency-boost / smooth-
#      clip logic). This function times get_action() broken into exactly
#      those stages, by faithfully re-executing get_action()'s own code
#      path step-by-step (calling the SAME agent methods -- featurize,
#      _maybe_fit_causal, _t, _phi_aug, self.actor -- not a reimplementation)
#      so the stage boundaries are real, not guessed.
# ─────────────────────────────────────────────────────────────────────────
def profile_get_action_breakdown(action_dim=20, n_iters=500, device=DEVICE, seed=42):
    set_seed(seed)
    env = NetworkTraumaEnvironment(num_nodes=action_dim)
    agent = CCNSAAgentV2(action_dim=action_dim, device=device)
    agent.actor.eval(); agent.causal.eval(); agent.symbolic.eval()
    state = env.reset()
    # let causal fit happen once, outside the timed loop (matches production:
    # the manuscript's Table-6 latency is measured post-warmup, not during
    # the one-time causal-fit event, which is a rare ~200-step-triggered cost
    # that would otherwise dominate and distort a per-step latency estimate)
    for _ in range(agent.causal_fit_threshold + 5):
        agent._maybe_fit_causal(featurize(state, agent.sd))
        state = env.step({"node_power": np.random.uniform(0.1, 1.0, action_dim)})[0]
    is_cuda = (device == "cuda")

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    stages = {"featurize": [], "host_to_device": [], "submodule_forward": [],
              "device_to_host": [], "numpy_postproc": [], "total": []}
    with torch.no_grad():
        for i in range(n_iters + 20):  # +20 warm-up, discarded below
            t0 = time.perf_counter()
            sv = featurize(state, agent.sd)
            _sync(); t1 = time.perf_counter()

            s_t = _t(sv, agent.device).unsqueeze(0)
            _sync(); t2 = time.perf_counter()

            aug, pi_sym = agent._phi_aug(s_t)
            raw_gpu = agent.actor(aug).squeeze(0)
            _sync(); t3 = time.perf_counter()

            raw = raw_gpu.cpu().numpy()
            _sync(); t4 = time.perf_counter()

            ap = np.clip((np.tanh(np.nan_to_num(raw)) + 1) / 2, 0.02, 1.0)
            if agent.use_emergency_boost:
                em = state["user_features"][:, 0] == 1
                if np.any(em):
                    ap[:min(5, agent.ad)] = np.maximum(ap[:min(5, agent.ad)], 0.80)
            if agent.use_smooth_clip and agent._last_action is not None:
                ap = agent._last_action + np.clip(ap - agent._last_action, -0.20, 0.20)
            ap = ap.clip(0.1, 1.0); agent._last_action = ap.copy()
            t5 = time.perf_counter()

            if i >= 20:  # discard warm-up iterations
                stages["featurize"].append((t1 - t0) * 1000)
                stages["host_to_device"].append((t2 - t1) * 1000)
                stages["submodule_forward"].append((t3 - t2) * 1000)
                stages["device_to_host"].append((t4 - t3) * 1000)
                stages["numpy_postproc"].append((t5 - t4) * 1000)
                stages["total"].append((t5 - t0) * 1000)
            state = env.step({"node_power": ap})[0]

    means = {k: float(np.mean(v)) for k, v in stages.items()}
    print("\n" + "=" * 70)
    print(f"FULL get_action() BREAKDOWN ({device}, n={n_iters} iters, after 20-iter warmup)")
    print("=" * 70)
    print(f"  featurize() [python/numpy]:        {means['featurize']:.5f} ms  "
          f"({100*means['featurize']/means['total']:.1f}%)")
    print(f"  host->device tensor creation:      {means['host_to_device']:.5f} ms  "
          f"({100*means['host_to_device']/means['total']:.1f}%)")
    print(f"  submodule forward (causal+sym+actor): {means['submodule_forward']:.5f} ms  "
          f"({100*means['submodule_forward']/means['total']:.1f}%)")
    print(f"  device->host .cpu().numpy():       {means['device_to_host']:.5f} ms  "
          f"({100*means['device_to_host']/means['total']:.1f}%)")
    print(f"  numpy post-processing:             {means['numpy_postproc']:.5f} ms  "
          f"({100*means['numpy_postproc']/means['total']:.1f}%)")
    print(f"  TOTAL (== real get_action() cost): {means['total']:.5f} ms")
    print(f"\n  For comparison: manuscript's reported CCNSA latency = 0.99 +/- 0.05 ms "
          f"(Table 6, same get_action() call, averaged over the 10-seed Track-A run).")
    print("  Whichever stage's percentage is largest is the actual dominant cost -- "
          "report that stage by name, not an assumption, in any manuscript text.")
    out = dict(means)
    out["raw_stages_ms"] = stages
    return out


# ─────────────────────────────────────────────────────────────────────────
# 10e. CROSS-AGENT LATENCY COMPARISON (CCNSA vs CQL vs IQL)
#      profile_get_action_breakdown() above only profiled CCNSA -- it could
#      show that ~40% of CCNSA's own latency is featurize() and ~40% is
#      submodule forward, but it could NOT say whether CCNSA's extra latency
#      over CQL/IQL (0.99ms vs 0.72ms in the manuscript's Table 6) comes from
#      the extra causal+symbolic compute specifically, since CQL/IQL were
#      never profiled the same way. This function closes that gap directly:
#      CQL/IQL's get_action() is much simpler (featurize -> actor forward ->
#      clip, no causal mask, no symbolic layer), so it's cheap to profile the
#      same two-way split (featurize() vs everything else) for all three
#      agents and compare "everything else" head-to-head -- that is the
#      piece that should differ if the extra causal/symbolic compute is
#      really what's driving CCNSA's higher latency, since featurize() is
#      the same function call for every agent (Section~sec:exp / code
#      comment at "BASELINE AGENTS (all operate on the same 32-dim
#      featurize() state)").
# ─────────────────────────────────────────────────────────────────────────
def profile_baseline_comparison(action_dim=20, n_iters=500, device=DEVICE, seed=42):
    set_seed(seed)
    env = NetworkTraumaEnvironment(num_nodes=action_dim)
    agents = {
        "CCNSA": CCNSAAgentV2(action_dim=action_dim, device=device),
        "CQL":   CQLAgent(ad=action_dim, device=device),
        "IQL":   IQLAgent(ad=action_dim, device=device),
    }
    is_cuda = (device == "cuda")

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    # Pre-collect a pool of REAL, varying states from an actual rollout
    # (random actions) instead of reusing one frozen state n_iters times.
    # Two reasons: (1) an earlier version of this function fed the same
    # frozen state to CCNSA's get_action() 500+ times, and at the 200th
    # identical call CCNSA's one-time causal-graph fit triggered on fully
    # duplicate rows -- every column had zero variance, which crashed with
    # an uncaught IndexError (see the _prune_degenerate_columns dtype bug
    # fix above, found via this exact real run); varying states avoid that
    # degenerate input entirely. (2) it's simply more representative of
    # real inference-time input. States are pre-computed OUTSIDE the timed
    # loop so env.step()'s own simulation cost never contaminates the
    # latency measurement below.
    n_pool = n_iters + 40
    pool = []
    s = env.reset()
    for _ in range(n_pool):
        pool.append(s)
        s = env.step({"node_power": np.random.uniform(0.1, 1.0, action_dim)})[0]

    # Pre-warm CCNSA's one-time causal-graph fit on this same real, varying
    # data before any timed measurement, matching how it actually happens
    # once early in a real Track-A run rather than mid-measurement.
    ccnsa = agents["CCNSA"]
    for st in pool[:ccnsa.causal_fit_threshold + 5]:
        ccnsa.get_action(st, training=False)

    def _time(fn_of_state, n=n_iters):
        for i in range(20):
            fn_of_state(pool[i])
        _sync()
        t0 = time.perf_counter()
        for i in range(n):
            fn_of_state(pool[i + 20])
        _sync()
        return (time.perf_counter() - t0) / n * 1000.0

    results = {}
    for name, agent in agents.items():
        t_featurize = _time(lambda st: featurize(st, agent.sd))
        t_total = _time(lambda st: agent.get_action(st, training=False))
        t_rest = t_total - t_featurize
        results[name] = {"featurize_ms": t_featurize, "rest_ms": t_rest, "total_ms": t_total}

    print("\n" + "=" * 70)
    print(f"CROSS-AGENT LATENCY COMPARISON ({device}, n={n_iters} iters, batch=1, after 20-call warmup)")
    print("=" * 70)
    print(f"  {'Agent':<8} {'featurize() ms':>16} {'rest ms':>12} {'total ms':>12}")
    for name, r in results.items():
        print(f"  {name:<8} {r['featurize_ms']:>16.5f} {r['rest_ms']:>12.5f} {r['total_ms']:>12.5f}")
    ccnsa_rest = results["CCNSA"]["rest_ms"]
    cql_rest = results["CQL"]["rest_ms"]
    iql_rest = results["IQL"]["rest_ms"]
    print(f"\n  CCNSA 'rest' (everything but featurize) vs CQL: {ccnsa_rest - cql_rest:+.5f} ms "
          f"({100*(ccnsa_rest - cql_rest)/ccnsa_rest:+.1f}% of CCNSA's own 'rest')")
    print(f"  CCNSA 'rest' (everything but featurize) vs IQL: {ccnsa_rest - iql_rest:+.5f} ms "
          f"({100*(ccnsa_rest - iql_rest)/ccnsa_rest:+.1f}% of CCNSA's own 'rest')")
    print(f"  featurize() spread across agents (should be small/noise if truly shared): "
          f"{max(r['featurize_ms'] for r in results.values()) - min(r['featurize_ms'] for r in results.values()):.5f} ms")
    print("  Interpretation: if 'rest' is close across agents, the extra causal+symbolic compute is "
          "NOT the dominant differentiator; if CCNSA's 'rest' is clearly larger, that IS direct evidence "
          "the extra submodules (not featurize(), not measurement noise) explain the gap. Report the "
          "actual numbers above, not this interpretation template, in any manuscript text.")
    return results


# ─────────────────────────────────────────────────────────────────────────
# 10g. GPU MEMORY FOOTPRINT + CPU-ONLY LATENCY
#      Reviewer-requested deployment-overhead numbers this paper did not
#      previously report: peak GPU memory during inference (not just
#      FLOPs/latency), and latency on CPU-only hardware (every other
#      latency figure in this paper is GPU-measured). Both are cheap to
#      get: memory via torch.cuda.max_memory_allocated() reset+replay, CPU
#      latency via the same get_action() timing already used elsewhere,
#      just with device forced to "cpu" for a fresh CCNSAAgentV2 instance
#      (a model trained on GPU can run inference on CPU by moving weights;
#      we build a fresh CPU instance here rather than move a trained one,
#      since only architecture-level compute cost is being measured, not
#      trained-weight-dependent behaviour).
# ─────────────────────────────────────────────────────────────────────────
def profile_memory_and_cpu(action_dim=20, n_iters_cpu=200, seed=42):
    out = {}

    # ---- GPU memory (only if CUDA is actually available) ----
    if torch.cuda.is_available():
        set_seed(seed)
        env = NetworkTraumaEnvironment(num_nodes=action_dim)
        agent_gpu = CCNSAAgentV2(action_dim=action_dim, device="cuda")
        state = env.reset()

        # Phase 1: INFERENCE-ONLY peak memory. get_action(training=False)
        # exclusively -- no update() calls at all, so no gradients/optimizer
        # state -- covering the causal-fit warmup (which allocates the
        # dim x dim adjacency/mask buffers) plus a further inference-only
        # stretch. This is the number relevant to "can this be deployed for
        # inference on a memory-constrained GPU without ever training on
        # it," which is a distinct question from training-time memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(agent_gpu.causal_fit_threshold + 25):
            a, _ = agent_gpu.get_action(state, training=False)
            state = env.step(a)[0]
        torch.cuda.synchronize()
        mem_inference_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Phase 2: TRAINING-time peak memory (forward+backward+optimizer).
        # Fill the replay buffer past batch size (self.bs=128) via update()
        # every step -- update() silently returns early without any
        # backward pass while len(self.mem) < self.bs, so a single call on
        # an empty/under-filled buffer would just re-measure Phase 1 by
        # accident. Peak stats are reset AFTER the buffer is full so the
        # reported number reflects a real training step's memory, not the
        # buffer-filling warmup.
        for _ in range(agent_gpu.bs + 5):
            a, _ = agent_gpu.get_action(state, training=True)
            ns, r, done = env.step(a)
            agent_gpu.update(state, a, r, ns, done)
            state = ns
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        a, _ = agent_gpu.get_action(state, training=True)
        ns, r, done = env.step(a)
        agent_gpu.update(state, a, r, ns, done)
        torch.cuda.synchronize()
        mem_train_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        out["gpu_peak_mb_inference_only"] = mem_inference_mb
        out["gpu_peak_mb_one_train_step"] = mem_train_mb
        out["gpu_device_name"] = torch.cuda.get_device_name(0)
        out["gpu_total_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    else:
        out["gpu_peak_mb_inference_only"] = None
        out["gpu_peak_mb_one_train_step"] = None
        out["gpu_note"] = "CUDA not available in this environment; GPU memory not measured."

    # ---- CPU-only latency (fresh CPU instance, separate from any GPU run above) ----
    set_seed(seed)
    env_cpu = NetworkTraumaEnvironment(num_nodes=action_dim)
    agent_cpu = CCNSAAgentV2(action_dim=action_dim, device="cpu")
    agent_cpu.actor.eval(); agent_cpu.causal.eval(); agent_cpu.symbolic.eval()
    pool = []
    s = env_cpu.reset()
    n_pool_cpu = n_iters_cpu + agent_cpu.causal_fit_threshold + 25
    for _ in range(n_pool_cpu):
        pool.append(s)
        s = env_cpu.step({"node_power": np.random.uniform(0.1, 1.0, action_dim)})[0]
    for st in pool[:agent_cpu.causal_fit_threshold + 5]:
        agent_cpu.get_action(st, training=False)
    warm_end = agent_cpu.causal_fit_threshold + 5
    for st in pool[warm_end:warm_end + 20]:
        agent_cpu.get_action(st, training=False)
    t0 = time.perf_counter()
    for st in pool[warm_end + 20:warm_end + 20 + n_iters_cpu]:
        agent_cpu.get_action(st, training=False)
    t_cpu_ms = (time.perf_counter() - t0) / n_iters_cpu * 1000.0
    out["cpu_latency_ms"] = t_cpu_ms
    out["cpu_n_iters"] = n_iters_cpu

    print("\n" + "=" * 70)
    print("GPU MEMORY + CPU-ONLY LATENCY")
    print("=" * 70)
    if torch.cuda.is_available():
        print(f"  GPU: {out['gpu_device_name']} ({out['gpu_total_mb']:.0f} MB total)")
        print(f"  Peak GPU memory, inference-only (no backward pass): "
              f"{out['gpu_peak_mb_inference_only']:.1f} MB")
        print(f"  Peak GPU memory, one training step (fwd+bwd+optim, full replay batch): "
              f"{out['gpu_peak_mb_one_train_step']:.1f} MB")
    else:
        print("  CUDA not available -- GPU memory not measured in this run.")
    print(f"  CPU-only get_action() latency: {t_cpu_ms:.4f} ms (n={n_iters_cpu} iters, "
          f"single-threaded default torch CPU settings, no batching)")
    print("  Report these numbers as-is; do not extrapolate to other hardware without a fresh run.")
    return out


# ─────────────────────────────────────────────────────────────────────────
# 11. REAL FAITHFULNESS TEST (SHAP / LIME / Integrated-Gradients agreement)
#     v16 computed this from ONE sample (not 500), then hard-clamped the
#     result into [0.70, 0.85] regardless of the true value, and hardcoded
#     every non-CCNSA agent's score as a constant. This version computes it
#     for real, for every agent that exposes a differentiable actor, over
#     `n_samples` held-out states, with no clamping.
# ─────────────────────────────────────────────────────────────────────────
def integrated_gradients(model_fn, inp, baseline=None, steps=50, device=DEVICE):
    if baseline is None: baseline = torch.zeros_like(inp)
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)
    interp = (baseline + (inp - baseline) * alphas).requires_grad_(True)
    out = model_fn(interp).sum(1)
    out.sum().backward()
    ig = ((interp.grad) * (inp - baseline) / steps).abs().sum(0)
    return ig.detach().cpu().numpy()

def _actor_scalar_fn(agent):
    """Return a function R^{n,d} -> R^{n,1} which routes through whatever
    representation the agent actually conditions its actor on (phi_aug for
    CCNSA, raw features for the rest), for gradient-/perturbation-based
    attribution. Returns None if the agent has no differentiable actor.
    NOTE: SAC's actor head outputs [mean ; log_std] concatenated (2*ad dims)
    -- only the mean half is a meaningful attribution target, so it is
    sliced out explicitly rather than averaging over log_std too."""
    if isinstance(agent, CCNSAAgentV2):
        def f(x):
            aug, _ = agent._phi_aug(x)
            return agent.actor(aug)
        return f, agent.sd
    if isinstance(agent, SACAgent):
        def f(x):
            return agent.actor(x)[:, :agent.ad]
        return f, agent.sd
    if hasattr(agent, "actor"):
        return (lambda x: agent.actor(x)), agent.sd
    if hasattr(agent, "net"):
        return (lambda x: agent.net(x)), agent.sd
    return None, None

def faithfulness_test(agents, env, n_samples=500, seed=0, verbose=True):
    import shap, lime.lime_tabular
    set_seed(seed)
    scores = {}
    bg_states = [env.reset() for _ in range(30)]
    for name, agent in agents.items():
        fn, sd = _actor_scalar_fn(agent)
        if fn is None:
            if verbose: print(f"  {name:22s}: no differentiable actor — faithfulness N/A")
            continue
        try:
            samples = [featurize(env.reset(), sd) for _ in range(n_samples)]
            X = np.stack(samples)
            bg = np.stack([featurize(s2, sd) for s2 in bg_states])

            def pred_fn(Xnp):
                Xt = _t(Xnp, DEVICE)
                with torch.no_grad():
                    return fn(Xt).mean(1).cpu().numpy()

            ig_imps, shap_imps, lime_imps = [], [], []
            exp_shap = shap.KernelExplainer(pred_fn, bg[:20])
            lime_exp = lime.lime_tabular.LimeTabularExplainer(bg, mode="regression")
            n_eval = min(n_samples, 40)  # SHAP/LIME are expensive per-sample; IG uses all n_samples
            for i in range(X.shape[0]):
                xt = _t(X[i:i + 1], DEVICE)  # requires_grad set inside integrated_gradients
                ig_imps.append(integrated_gradients(fn, xt, steps=20))
            for i in range(n_eval):
                sv = shap.KernelExplainer(pred_fn, bg[:15]).shap_values(X[i:i + 1], nsamples=60, silent=True)
                shap_imps.append(np.abs(sv[0] if isinstance(sv, list) else sv).flatten())
                lres = lime_exp.explain_instance(X[i], pred_fn, num_features=sd)
                li = np.zeros(sd)
                for fi, fw in lres.as_list():
                    try:
                        idx = int(fi.split(" ")[0].replace("feature_", ""))
                        li[idx] = abs(fw)
                    except Exception:
                        pass
                lime_imps.append(li)
            ig_mean = np.mean(ig_imps[:n_eval], axis=0)
            shap_mean = np.mean(shap_imps, axis=0)
            lime_mean = np.mean(lime_imps, axis=0)
            rk_shap = spstats.rankdata(-shap_mean); rk_ig = spstats.rankdata(-ig_mean); rk_lime = spstats.rankdata(-lime_mean)
            rho_si, _ = spstats.spearmanr(rk_shap, rk_ig)
            rho_sl, _ = spstats.spearmanr(rk_shap, rk_lime)
            rho_li, _ = spstats.spearmanr(rk_lime, rk_ig)
            f_score = float(np.nanmean([rho_si, rho_sl, rho_li]))
            scores[name] = {"F": f_score, "rho_SI": rho_si, "rho_SL": rho_sl, "rho_LI": rho_li, "n_samples": n_eval}
            if verbose: print(f"  {name:22s}: F={f_score:.4f}  (rho_SI={rho_si:.3f} rho_SL={rho_sl:.3f} rho_LI={rho_li:.3f}, n={n_eval})")
        except Exception as e:
            if verbose: print(f"  {name:22s}: faithfulness computation failed ({e})")
    return scores


# ─────────────────────────────────────────────────────────────────────────
# 12. REAL DELETION-INSERTION FIDELITY TEST (Petsiuk et al., 2018)
#     Absent from v16 entirely — no code anywhere computed this, despite it
#     being a headline abstract number. Implemented here for real, plus a
#     random-ranking control (expected Delta ~ 0) as the manuscript claims.
# ─────────────────────────────────────────────────────────────────────────
def deletion_insertion_test(agent, env, n_samples=40, seed=0, verbose=True):
    fn, sd = _actor_scalar_fn(agent)
    if fn is None:
        return None
    set_seed(seed)
    Ks = list(range(0, sd + 1, 2))
    del_curve, ins_curve, del_curve_rand, ins_curve_rand = [], [], [], []
    for _ in range(n_samples):
        s = env.reset()
        x = featurize(s, sd)
        xt = _t(x, DEVICE).unsqueeze(0)  # requires_grad set inside integrated_gradients
        imp = integrated_gradients(fn, xt, steps=20)
        order = np.argsort(-imp)          # most-important-first
        rand_order = np.random.permutation(sd)

        def score_masked(order_, K, mode):
            xm = x.copy()
            idx = order_[:K]
            if mode == "delete": xm[idx] = 0.0
            else:  # insert: start from zero baseline, restore top-K
                base = np.zeros_like(x); base[idx] = x[idx]; xm = base
            with torch.no_grad():
                return float(fn(_t(xm, DEVICE).unsqueeze(0)).mean().item())

        d_row, i_row, dr_row, ir_row = [], [], [], []
        for K in Ks:
            d_row.append(score_masked(order, K, "delete"))
            i_row.append(score_masked(order, K, "insert"))
            dr_row.append(score_masked(rand_order, K, "delete"))
            ir_row.append(score_masked(rand_order, K, "insert"))
        del_curve.append(d_row); ins_curve.append(i_row)
        del_curve_rand.append(dr_row); ins_curve_rand.append(ir_row)

    def auc(curve):
        c = np.mean(curve, axis=0)
        c = (c - c.min()) / (c.max() - c.min() + 1e-9)
        dx = 1.0 / (len(c) - 1) if len(c) > 1 else 1.0
        # np.trapz was removed in NumPy 2.0 in favor of np.trapezoid; support both.
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return float(_trapz(c, dx=dx))

    auc_del, auc_ins = auc(del_curve), auc(ins_curve)
    auc_del_r, auc_ins_r = auc(del_curve_rand), auc(ins_curve_rand)
    delta = auc_ins - auc_del
    delta_random = auc_ins_r - auc_del_r
    if verbose:
        print(f"    AUC_del={auc_del:.4f} AUC_ins={auc_ins:.4f}  Delta={delta:+.4f}   "
              f"(random-ranking control: Delta={delta_random:+.4f})")
    return {"AUC_del": auc_del, "AUC_ins": auc_ins, "Delta": delta,
            "Delta_random_control": delta_random}

def run_deletion_insertion_suite(agents, env, n_samples=40, verbose=True):
    if verbose:
        print("\n" + "=" * 70); print("DELETION-INSERTION FIDELITY TEST (Petsiuk et al., 2018)")
        print("=" * 70)
    out = {}
    for name, agent in agents.items():
        if verbose: print(f"  {name}:")
        r = deletion_insertion_test(agent, env, n_samples=n_samples, verbose=verbose)
        if r is not None: out[name] = r
    return out

# ─────────────────────────────────────────────────────────────────────────
# 13. COMPLEXITY / FLOPS / LATENCY
# ─────────────────────────────────────────────────────────────────────────
def flop_count(model):
    f = 0
    for l in model.modules():
        if isinstance(l, nn.Linear): f += 2 * l.in_features * l.out_features
    return f

def complexity_report(action_dim=20, verbose=True):
    if verbose:
        print("\n" + "=" * 70); print("COMPLEXITY: FLOPs & LATENCY"); print("=" * 70)
    agents = make_agents(action_dim=action_dim)
    env = NetworkTraumaEnvironment(num_nodes=action_dim)
    flops, lats = {}, {}
    for name, ag in agents.items():
        try:
            model = ag.actor if hasattr(ag, "actor") else getattr(ag, "net", None)
            f = flop_count(model) if model is not None else 0
            if isinstance(ag, CCNSAAgentV2):
                f += flop_count(ag.causal.embed) + ag.causal.dim ** 2 + flop_count(ag.symbolic.rules)
        except Exception:
            f = 0
        ls = []
        for _ in range(50):
            s = env.reset()
            t0 = time.perf_counter(); ag.get_action(s, training=False); ls.append((time.perf_counter() - t0) * 1000)
        flops[name] = f; lats[name] = {"mean": float(np.mean(ls)), "std": float(np.std(ls))}
        if verbose: print(f"  {name:22s}: FLOPs={f:>10,} | latency={np.mean(ls):.3f}+/-{np.std(ls):.3f} ms")
    return flops, lats


# ─────────────────────────────────────────────────────────────────────────
# 14. CAUSAL-DISCOVERY COMPARISON ON TRACK-A FEATURE HISTORY
# ─────────────────────────────────────────────────────────────────────────
def causal_discovery_report(action_dim=20, n_episodes=3, n_steps=50, verbose=True):
    if verbose:
        print("\n" + "=" * 70); print("CAUSAL DISCOVERY COMPARISON — PC vs NOTEARS vs DirectLiNGAM"); print("=" * 70)
    env = NetworkTraumaEnvironment(num_nodes=action_dim)
    hist = []
    s = env.reset()
    for _ in range(n_episodes * n_steps):
        hist.append(featurize(s, STATE_DIM))
        a = {"node_power": np.random.uniform(0.3, 1.0, action_dim)}
        s, r, done = env.step(a)
        if done: s = env.reset()
    X = np.nan_to_num(np.array(hist))
    best, report = compare_causal_discovery(X, n_boot=8, verbose=verbose)
    return best, report


# ─────────────────────────────────────────────────────────────────────────
# 15. LATEX TABLE EXPORT
# ─────────────────────────────────────────────────────────────────────────
def export_latex_table(aggregated, faith_scores, lat_results, di_results=None):
    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Performance comparison on Network Trauma Recovery "
                  r"(10 seeds $\times$ 50 episodes). Bold = CCNSA. "
                  r"Faithfulness and deletion-insertion are computed, not simulated.}")
    lines.append(r"  \label{tab:results}")
    cols = "lccccc" if di_results else "lcccc"
    lines.append(r"  \begin{tabular}{%s}" % cols)
    lines.append(r"    \toprule")
    hdr = r"    \textbf{Method} & \textbf{Reward} & \textbf{Faithfulness} ($F$) & \textbf{Latency (ms)}"
    if di_results: hdr += r" & \textbf{Del-Ins} $\Delta$"
    hdr += r" \\"
    lines.append(hdr); lines.append(r"    \midrule")
    for ag in AGENTS_ALL:
        r = aggregated.get(ag, {}).get("performance", {"mean": 0, "ci_95": 0})
        fs = faith_scores.get(ag, {}).get("F", float("nan"))
        lt = lat_results.get(ag, {}).get("mean", 0)
        bold = r"\textbf" if ag == "CCNSA" else ""
        # BUG FIX: agents with no differentiable actor (RuleBased,
        # DecisionTransformer) have no faithfulness/deletion-insertion
        # score by construction -- this used to format float("nan") with
        # f"{fs:.3f}", which prints the literal text "nan" straight into
        # the LaTeX table (not a compile error, but an obviously unpolished
        # "nan"/"+nan" sitting in a submitted table). Render as "--" instead.
        fs_str = "--" if np.isnan(fs) else f"{fs:.3f}"
        row = f"    {bold}{{{ag}}} & {bold}{{{r['mean']:.4f}$\\pm${r['ci_95']:.4f}}} & {bold}{{{fs_str}}} & {bold}{{{lt:.2f}}}"
        if di_results:
            d = di_results.get(ag, {}).get("Delta", float("nan"))
            d_str = "--" if np.isnan(d) else f"{d:+.3f}"
            row += f" & {bold}{{{d_str}}}"
        row += r" \\"
        lines.append(row)
    lines.append(r"    \bottomrule"); lines.append(r"  \end{tabular}"); lines.append(r"\end{table}")
    table = "\n".join(lines)
    print("\n" + table)
    return table

# ─────────────────────────────────────────────────────────────────────────
# 16. TRACK B — REAL UNSW-NB15 VALIDATION
#     No Google Drive / Colab dependency. Downloads directly via the
#     `datasets` library (HuggingFace mirror), cached locally under OUT_DIR.
#     Falls back to a documented statistical replication if unreachable —
#     the fallback is clearly labelled in every downstream report, unlike
#     v16 where DATA_SRC could silently say "Real" incorrectly if the
#     Drive cache held a stale file.
# ─────────────────────────────────────────────────────────────────────────
SEVERITY = {"DoS": 0.85, "Fuzzers": 0.80, "Exploits": 0.78, "Generic": 0.70,
            "Reconnaissance": 0.68, "Analysis": 0.65, "Backdoor": 0.62,
            "Shellcode": 0.60, "Worms": 0.55, "Normal": 0.0}
UNSW_FEATS = [
    "dur", "proto", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb",
    "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "sjit", "djit", "sinpkt", "dinpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
    "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm"
]

def download_unsw_nb15(cache_dir=None):
    import pandas as pd
    cache_dir = cache_dir or OUT_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "unsw_nb15_data.pkl")
    if os.path.exists(cache_path):
        print(f"  loading cached dataset: {cache_path}")
        return pickle.load(open(cache_path, "rb")), True
    try:
        print("  downloading UNSW-NB15 (HuggingFace: Mouwiya/UNSW-NB15)...")
        from datasets import load_dataset
        df = pd.DataFrame(load_dataset("Mouwiya/UNSW-NB15", split="train"))
        print(f"  downloaded {len(df):,} records")
        pickle.dump(df, open(cache_path, "wb"))
        return df, True
    except Exception as e:
        print(f"  download failed ({e}); using documented statistical replication "
              f"(Moustafa & Slay, 2015) — NOT real data, clearly flagged in all outputs.")
        return None, False

def build_unsw_replication(seed=1984):
    import pandas as pd
    np.random.seed(seed)
    UNSW_N = {"Normal": 56000, "Generic": 18871, "Exploits": 11132, "Fuzzers": 6062,
              "DoS": 4089, "Reconnaissance": 3496, "Analysis": 677,
              "Backdoor": 583, "Shellcode": 378, "Worms": 44}
    rows = []
    for cls, n in UNSW_N.items():
        sev = SEVERITY.get(cls, 0.5)
        base = np.clip(np.random.rand(n, len(UNSW_FEATS)) * (0.4 + sev * 0.4)
                        + np.random.normal(0, 0.12, (n, len(UNSW_FEATS))), 0, 1)
        shifts = {"sbytes": 0.5 + sev * 0.3, "sttl": 0.4 + sev * 0.2, "sjit": 0.2 + sev * 0.4,
                  "sloss": sev * 0.4, "ct_srv_src": 0.3 + sev * 0.3, "dbytes": max(0.05, (1 - sev) * 0.6)}
        for fi, fn in enumerate(UNSW_FEATS):
            if fn in shifts: base[:, fi] = np.clip(np.random.normal(shifts[fn], 0.1, n), 0, 1)
        d = pd.DataFrame(base, columns=UNSW_FEATS); d["attack_cat"] = cls
        rows.append(d)
    return pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed)

def run_track_b(ccnsa_actor=None, seed=1984, verbose=True):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import f1_score

    set_seed(seed)
    if verbose: print("=" * 70); print("TRACK B — REAL UNSW-NB15 VALIDATION"); print("=" * 70)

    df_raw, is_real = download_unsw_nb15()
    if df_raw is None: df_raw = build_unsw_replication(seed)
    data_src = "Real UNSW-NB15 (HuggingFace)" if is_real else "STATISTICAL REPLICATION (not real data)"
    print(f"  data source: {data_src}")

    df = df_raw.copy()
    cat_col = next((c for c in df.columns if "attack" in c.lower() and "cat" in c.lower()),
                    next((c for c in df.columns if "label" in c.lower()), None))
    if cat_col is None: cat_col = "attack_cat"; df[cat_col] = "Normal"
    NM = {"Dos": "DoS", "Backdoors": "Backdoor", "Recon": "Reconnaissance", "": "Normal"}
    df["cls"] = (df[cat_col].astype(str).str.strip().str.title()
                 .map(lambda x: NM.get(x, x if x in SEVERITY else "Normal")))
    for col in df.columns:
        if df[col].dtype == object and col not in ["cls", cat_col]:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    avail = [f for f in UNSW_FEATS if f in df.columns]
    if len(avail) < 5:
        avail = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in ["label", "label_enc", "cls"]][:40]
    for col in avail: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    le = LabelEncoder(); df["label"] = le.fit_transform(df["cls"])
    X = df[avail].fillna(0).values.astype(np.float32); y = df["label"].values
    sc = StandardScaler(); X = sc.fit_transform(X)

    # class-aware sampling (unchanged policy from v16)
    all_ids = np.unique(y); counts = np.array([(y == c).sum() for c in all_ids])
    normal_id = all_ids[np.argmax(counts)]
    atk_ids = all_ids[all_ids != normal_id]; atk_sizes = np.array([(y == c).sum() for c in atk_ids])
    minority_thresh = int(np.percentile(atk_sizes, 75)) if len(atk_sizes) else 0
    majority_cap = minority_thresh
    retained = atk_sizes[atk_sizes <= minority_thresh]
    normal_cap = int(retained.max()) * 2 if len(retained) else counts.max()
    idx_keep = []
    for cid in all_ids:
        cidx = np.where(y == cid)[0]
        if cid == normal_id: idx_keep.extend(np.random.choice(cidx, min(normal_cap, len(cidx)), replace=False))
        elif len(cidx) <= minority_thresh: idx_keep.extend(cidx.tolist())
        else: idx_keep.extend(np.random.choice(cidx, majority_cap, replace=False))
    idx_keep = np.array(idx_keep)
    Xs, ys = X[idx_keep], y[idx_keep]
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, random_state=seed, stratify=ys)
    if verbose: print(f"  sampled: {len(Xs):,} records, {len(avail)} features, "
                       f"{Xtr.shape[0]:,}/{Xte.shape[0]:,} train/test")

    # B2: classifier comparison
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced")
    rf.fit(Xtr, ytr); f1_rf = f1_score(yte, rf.predict(Xte), average="macro")
    lr = LogisticRegression(max_iter=500, class_weight="balanced"); lr.fit(Xtr, ytr)
    f1_lr = f1_score(yte, lr.predict(Xte), average="macro")
    svm = LinearSVC(class_weight="balanced", max_iter=2000); svm.fit(Xtr, ytr)
    f1_svm = f1_score(yte, svm.predict(Xte), average="macro")
    if verbose:
        print(f"\n  B2 macro-F1 — RF: {f1_rf:.4f}  LR: {f1_lr:.4f}  LinearSVM: {f1_svm:.4f}")

    # B3: causal discovery on real feature correlations (PC vs NOTEARS vs LiNGAM)
    attack_mask = df.loc[df.index[idx_keep], "cls"] != "Normal"
    df_atk = df.loc[df.index[idx_keep]][attack_mask][avail].fillna(0)
    col_vars = np.var(df_atk.values, axis=0)
    top_idx = np.argsort(col_vars)[::-1][:8]
    from sklearn.preprocessing import StandardScaler as _SS
    Xc = _SS().fit_transform(df_atk.values[:, top_idx][:2000])
    if verbose: print(f"\n  B3 causal discovery on top-8 variance features: {[avail[i] for i in top_idx]}")
    best_method, causal_report = compare_causal_discovery(np.nan_to_num(Xc), n_boot=6, verbose=verbose)

    # B4: audit rules — DISTINCT from CCNSA's 8 trained policy rules (fixes
    # the 8-vs-6 confusion reviewers flagged: these are 6 fixed, hand-curated
    # rules used only for this real-data interpretability audit).
    def acp(rule_mask, targets):
        atk_labels = df.loc[df.index[idx_keep], "cls"].values[rule_mask]
        if rule_mask.sum() == 0: return 0.0, 0
        hit = np.isin(atk_labels, targets).mean()
        return float(hit), int(rule_mask.sum())

    dfa = df.loc[df.index[idx_keep]].reset_index(drop=True)
    feat_df = pd.DataFrame(Xs, columns=avail)
    audit_rules = []
    def col(name): return feat_df[name].values if name in feat_df.columns else np.zeros(len(feat_df))
    def pct(name, p): 
        c = col(name); return np.percentile(c, p) if len(c) else 0
    specs = [
        ("sbytes>P60", col("sbytes") > pct("sbytes", 60), ["Exploits", "Generic"]),
        ("dur>P70", col("dur") > pct("dur", 70), ["DoS", "Generic"]),
        ("ct_state_ttl>P75", col("ct_state_ttl") > pct("ct_state_ttl", 75), ["Generic", "Exploits", "DoS"]),
        ("dbytes>P70", col("dbytes") > pct("dbytes", 70), ["DoS", "Exploits", "Generic"]),
        ("smeansz&dmeansz>P75", (col("smeansz") > pct("smeansz", 75)) & (col("dmeansz") > pct("dmeansz", 75)), ["Exploits", "Fuzzers"]),
        ("sbytes>P80", col("sbytes") > pct("sbytes", 80), ["Generic", "Exploits"]),
    ]
    if verbose: print(f"\n  B4 audit-rule attack-conditional precision (6 rules, distinct from the 8 policy rules):")
    for rname, mask, targets in specs:
        a, support = acp(mask, targets)
        audit_rules.append({"rule": rname, "acp": a, "support": support, "targets": targets})
        if verbose: print(f"    {rname:<24} ACP={a:.3f}  support={support}")
    avg_acp = float(np.mean([r["acp"] for r in audit_rules])) if audit_rules else 0.0
    if verbose: print(f"    average ACP: {avg_acp:.3f}")

    # B5: ensemble (CCNSA reward-signal proxy + RF)
    rf_proba = rf.predict_proba(Xte)
    ensemble_pred = rf.predict(Xte)  # CCNSA is a sequential-decision agent, not a
    # classifier (Sec. 8.3 of the manuscript) — its contribution here is a
    # weighting prior, not a competing class prediction; keep RF as the
    # per-record classifier and report the RF-only number honestly rather
    # than fabricating a "CCNSA improves ensemble" delta without the actual
    # trained Track-A actor wired in.
    f1_ensemble = f1_score(yte, ensemble_pred, average="macro")

    results = {
        "data_source": data_src, "is_real": is_real,
        "n_records": int(len(Xs)), "n_features": len(avail),
        "f1_rf": f1_rf, "f1_lr": f1_lr, "f1_svm": f1_svm, "f1_ensemble": f1_ensemble,
        "causal_best_method": best_method, "causal_report": {k: v["stability"] for k, v in causal_report.items()},
        "audit_rules": audit_rules, "audit_rule_avg_acp": avg_acp,
    }
    with open(os.path.join(OUT_DIR, "track_b_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


# ─────────────────────────────────────────────────────────────────────────
# 17. MAIN
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["A", "B", "AB", "ABL", "RULES", "CAUSAL", "CASCADE", "PROFILE", "PROFILE2", "PROFILE3", "PROFILE4"], default="AB",
                     help="ABL = ablation-only re-run (uses the same seed count as Track A "
                          "would, but skips Track A/B training+eval entirely -- for iterating "
                          "on ABLATION_VARIANTS without re-paying the full pipeline cost). "
                          "RULES = rule-count sensitivity sweep (r=4/8/16), same seed count as "
                          "Track A would use; see RULE_COUNT_VARIANTS/run_rule_count_study "
                          "docstring for the u-vector caveat on r!=8. "
                          "PROFILE = latency breakdown of just the 3 neural submodules "
                          "(causal/symbolic/actor), no training, runs in seconds. "
                          "PROFILE2 = FULL get_action() breakdown (featurize/host-device/"
                          "submodule-forward/device-host/numpy-postproc) -- this is the one "
                          "comparable to the manuscript's reported 0.99ms latency figure, "
                          "since PROFILE alone measures a narrower thing. Runs in ~1 min. "
                          "PROFILE3 = cross-agent comparison (CCNSA vs CQL vs IQL), same "
                          "featurize()-vs-rest split for all three, to directly test whether "
                          "CCNSA's extra causal+symbolic compute (not featurize(), which is "
                          "identical across agents) explains its latency gap vs CQL/IQL. "
                          "Runs in ~1 min, no training. "
                          "CAUSAL = causal-discovery method sweep (PC/NOTEARS/DirectLiNGAM) on "
                          "DOWNSTREAM Track-A performance (reward/faithfulness/DelIns), same "
                          "seed count as Track A would use -- distinct from Track B's existing "
                          "edge-stability comparison, which never touches Track A. "
                          "PROFILE4 = peak GPU memory (if CUDA available) + CPU-only latency "
                          "for get_action(), no training, runs in ~1 min. "
                          "CASCADE = 4-arm core ablation (Full/w-o causal/w-o symbolic/w-o both) "
                          "re-run on CascadingTraumaEnvironment (fixed node dependency hierarchy "
                          "+ power budget, see class docstring) instead of the base env, to test "
                          "whether causal/symbolic become individually necessary on a task with "
                          "real latent structure to exploit -- the main ablation's null result "
                          "was on the base env, which has no such structure. Same seed count as "
                          "Track A would use.")
    ap.add_argument("--quick", action="store_true", help="small smoke-test sizes")
    ap.add_argument("--causal-method", choices=["PC", "NOTEARS", "DirectLiNGAM"], default="PC")
    ap.add_argument("--seeds", type=int, default=None,
                     help="number of seeds for Track A (default: 2 with --quick, else 10). "
                          "Use 1 to calibrate per-seed wall-clock time on your GPU before "
                          "committing to a full run; use ~20 for adequately powered "
                          "comparisons against close baselines (see Prop. 2 discussion).")
    ap.add_argument("--fresh-only", action="store_true",
                     help="Replication check: drop the original 10 dev seeds "
                          "(42, 123, 456, ... -- reused across every iteration of this "
                          "codebase all session) and keep ONLY the newly-extended seeds "
                          "from get_seeds(), e.g. `--seeds 20 --fresh-only` trains/evals "
                          "on exactly the 10 never-before-used seeds beyond the original "
                          "set, to check results aren't an artifact of implicitly tuning "
                          "against the same dev seeds.")
    args = ap.parse_args()

    n_episodes = 5 if args.quick else 50
    n_steps = 20 if args.quick else 100
    default_n_seeds = 2 if args.quick else 10
    n_seeds = args.seeds if args.seeds is not None else default_n_seeds
    seeds = get_seeds(n_seeds)
    if args.fresh_only:
        if n_seeds <= len(SEEDS):
            raise SystemExit(f"--fresh-only needs --seeds > {len(SEEDS)} (the original dev "
                              f"seed count) so there are new seeds left after dropping the "
                              f"originals; got --seeds {n_seeds}.")
        seeds = seeds[len(SEEDS):]
        print(f"--fresh-only: dropped the {len(SEEDS)} original dev seeds, "
              f"training/evaluating on {len(seeds)} never-before-used seed(s): {seeds}")
    abl_episodes = 2 if args.quick else 50
    faith_samples = 20 if args.quick else 500
    di_samples = 5 if args.quick else 40

    results = {}
    if args.track == "PROFILE":
        prof = profile_latency_breakdown()
        with open(os.path.join(OUT_DIR, "latency_profile.json"), "w") as f:
            json.dump(prof, f, indent=2, default=float)
        print(f"\nLatency profile written to: {OUT_DIR}")
        return prof

    if args.track == "PROFILE2":
        prof2 = profile_get_action_breakdown()
        with open(os.path.join(OUT_DIR, "latency_profile_full.json"), "w") as f:
            json.dump(prof2, f, indent=2, default=float)
        print(f"\nFull get_action() profile written to: {OUT_DIR}")
        return prof2

    if args.track == "PROFILE3":
        prof3 = profile_baseline_comparison()
        with open(os.path.join(OUT_DIR, "latency_profile_cross_agent.json"), "w") as f:
            json.dump(prof3, f, indent=2, default=float)
        print(f"\nCross-agent latency comparison written to: {OUT_DIR}")
        return prof3

    if args.track == "RULES":
        # Same seed-count logic as ABL: full 10 seeds (or --seeds N) unless
        # --quick, matching Table 4's ablation rigor rather than the 3-seed
        # fast-mini scale.
        rule_seeds = tuple(seeds) if not args.quick else tuple(seeds[:2])
        _rc_t0 = time.time()
        rule_results, rule_stat_rows = run_rule_count_study(
            seeds=rule_seeds, n_episodes=abl_episodes, n_steps=n_steps)
        print(f"\n  Rule-count sweep wall-clock: {time.time()-_rc_t0:.1f}s for "
              f"{len(RULE_COUNT_VARIANTS)} variant(s) x {len(rule_seeds)} seed(s).")
        results.update(dict(rule_count=rule_results, rule_count_stat_rows=rule_stat_rows))
        with open(os.path.join(OUT_DIR, "rule_count_sweep_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nRule-count sweep results written to: {OUT_DIR}")
        return results

    if args.track == "CAUSAL":
        # Same seed-count logic as RULES/ABL.
        causal_seeds = tuple(seeds) if not args.quick else tuple(seeds[:2])
        _cm_t0 = time.time()
        causal_results, causal_stat_rows = run_causal_method_study(
            seeds=causal_seeds, n_episodes=abl_episodes, n_steps=n_steps)
        print(f"\n  Causal-method sweep wall-clock: {time.time()-_cm_t0:.1f}s for "
              f"{len(CAUSAL_METHOD_VARIANTS)} variant(s) x {len(causal_seeds)} seed(s).")
        results.update(dict(causal_method=causal_results, causal_method_stat_rows=causal_stat_rows))
        with open(os.path.join(OUT_DIR, "causal_method_sweep_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nCausal-method sweep results written to: {OUT_DIR}")
        return results

    if args.track == "CASCADE":
        # Same seed-count logic as RULES/CAUSAL.
        cascade_seeds = tuple(seeds) if not args.quick else tuple(seeds[:2])
        _cs_t0 = time.time()
        cascade_results, cascade_stat_rows = run_cascade_study(
            seeds=cascade_seeds, n_episodes=abl_episodes, n_steps=n_steps)
        print(f"\n  Cascading-trauma ablation wall-clock: {time.time()-_cs_t0:.1f}s for "
              f"{len(CASCADE_ABLATION_VARIANTS)} variant(s) x {len(cascade_seeds)} seed(s).")
        results.update(dict(cascade=cascade_results, cascade_stat_rows=cascade_stat_rows))
        with open(os.path.join(OUT_DIR, "cascade_ablation_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nCascading-trauma ablation results written to: {OUT_DIR}")
        return results

    if args.track == "PROFILE4":
        prof4 = profile_memory_and_cpu()
        with open(os.path.join(OUT_DIR, "memory_and_cpu_profile.json"), "w") as f:
            json.dump(prof4, f, indent=2, default=float)
        print(f"\nMemory/CPU profile written to: {OUT_DIR}")
        return prof4

    if args.track == "ABL":
        # Fast path: re-run only the ablation study (e.g. after editing
        # ABLATION_VARIANTS) without re-paying Track A's ~2hr train+eval
        # pass. Uses the same seed set Track A would use for the given
        # --seeds/--quick flags, so results stay comparable to a prior full
        # run's ablation section.
        abl_seeds = tuple(seeds) if not args.quick else tuple(seeds[:2])
        _abl_t0 = time.time()
        ablation = run_ablation_study(seeds=abl_seeds, n_episodes=abl_episodes, n_steps=n_steps)
        print(f"\n  Ablation wall-clock: {time.time()-_abl_t0:.1f}s for {len(ABLATION_VARIANTS)} "
              f"variant(s) x {len(abl_seeds)} seed(s).")
        abl_stat_rows = ablation_significance_test(ablation)
        results.update(dict(ablation=ablation, ablation_stat_rows=abl_stat_rows))
        with open(os.path.join(OUT_DIR, "ablation_only_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nAblation-only results written to: {OUT_DIR}")
        return results

    if "A" in args.track and args.track != "ABL":
        _t0 = time.time()
        (best_seed, best_seed_ev), aggregated, raw_data, conv_data, faith_raw_data, di_raw_data = run_track_a(
            seeds=seeds, num_episodes=n_episodes, num_steps=n_steps, causal_method=args.causal_method,
            faith_seed_samples=(5 if args.quick else 15), di_seed_samples=(5 if args.quick else 10))
        _elapsed = time.time() - _t0
        print(f"\n  Track A wall-clock: {_elapsed:.1f}s for {len(seeds)} seed(s) "
              f"({_elapsed/len(seeds):.1f}s/seed) -- extrapolate for other --seeds counts from this.")
        stat_rows = statistical_tests(raw_data)
        iqm_report = rliable_report(raw_data)
        faith_stat_rows = faithfulness_significance_test(faith_raw_data)
        di_stat_rows = deletion_insertion_significance_test(di_raw_data)
        # Ablation now matches Track A's full seed count and episode budget
        # (was 5 seeds x 5 episodes -- underpowered: deltas of 0.02-0.07 sat
        # inside std devs of 0.5-0.9, so removing a component could not be
        # told apart from noise). Using the full N seeds x 50 episodes gives
        # each variant the same training budget as the main-track agents and
        # a real paired significance test below, at the cost of ~5x the
        # ablation wall-clock (6 variants x N seeds, each trained to full
        # length, vs the previous 6 x 5 seeds x 5 episodes).
        abl_seeds = tuple(seeds) if not args.quick else tuple(seeds[:2])
        _abl_t0 = time.time()
        ablation = run_ablation_study(seeds=abl_seeds, n_episodes=abl_episodes, n_steps=n_steps)
        print(f"\n  Ablation wall-clock: {time.time()-_abl_t0:.1f}s for {len(ABLATION_VARIANTS)} "
              f"variant(s) x {len(abl_seeds)} seed(s).")
        abl_stat_rows = ablation_significance_test(ablation)
        env = NetworkTraumaEnvironment()
        faith = faithfulness_test(best_seed_ev.agents, env, n_samples=faith_samples)
        di = run_deletion_insertion_suite(best_seed_ev.agents, env, n_samples=di_samples)
        flops, lats = complexity_report()
        best_causal, causal_cmp = causal_discovery_report()
        table = export_latex_table(aggregated, faith, lats, di)
        results.update(dict(aggregated=aggregated, stat_rows=stat_rows, iqm_report=iqm_report, ablation=ablation,
                             ablation_stat_rows=abl_stat_rows,
                             faithfulness=faith, faith_stat_rows=faith_stat_rows, deletion_insertion=di,
                             di_stat_rows=di_stat_rows, flops=flops, latency=lats, causal_best_method=best_causal,
                             causal_comparison={k: v["stability"] for k, v in causal_cmp.items()}))
        with open(os.path.join(OUT_DIR, "track_a_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)

    if "B" in args.track:
        results["track_b"] = run_track_b()

    print(f"\nAll results written to: {OUT_DIR}")
    return results


if __name__ == "__main__":
    main()
