"""
chatbot_env.py
==============
Natural-language chatbot environment for the gradient-free Active Inference
architecture.  Replaces ``non_stationary_grid.py``.

Design goals
────────────
1. **Observation space** — incoming user text, tokenised into discrete
   word tokens by a whitespace + punctuation splitter.  No neural
   embeddings, no sub-word BPE — pure discrete symbols.

2. **Action space** — outgoing sequence of word tokens assembled by the
   agent's policy layer.  The environment presents these back to the
   user as a plain string.

3. **Anti-Dark-Room penalty** — Active Inference agents are prone to
   minimising surprise by *refusing to act* (the "Dark Room" problem).
   This environment imposes a **homeostatic drive penalty** whenever
   the agent:
     (a) emits an empty response, or
     (b) repeats the exact same response K times in a row.
   The penalty is applied to the *Affiliation* and *Energy* drives so
   that silence is mathematically more surprising than speaking.

4. **Caregiver (user) sentiment** — a trivial keyword detector classifies
   user replies as positive / negative / neutral, providing Type-I /
   Type-II social feedback to the Dirichlet world model.

Mathematical note
─────────────────
The observation index fed to the Active Inference engine is the hash of
the dominant (most activated) word node after spreading activation.
This preserves the discrete-state contract  o ∈ {0, …, O−1}  without
any continuous representation.

No gradients, no neural networks — all processing is combinatorial.
"""

from __future__ import annotations

import re
import string
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Tokeniser — pure discrete, no neural sub-word model
# ═══════════════════════════════════════════════════════════════════════

# Pre-compiled regex: split on whitespace and separate punctuation tokens.
# E.g. "Hello, world!" → ["hello", ",", "world", "!"]
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[^\s\w]")


def tokenise(text: str) -> List[str]:
    """
    Split *text* into lowercase word tokens and punctuation tokens.

    Examples
    --------
    >>> tokenise("Hello, world!  How are you?")
    ['hello', ',', 'world', '!', 'how', 'are', 'you', '?']

    >>> tokenise("")
    []

    Complexity: O(|text|).  No learned parameters.
    """
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def detokenise(tokens: List[str]) -> str:
    """
    Reassemble a token list into a human-readable string.

    Heuristic: punctuation attaches to the preceding word (no leading
    space); everything else is space-separated.
    """
    if not tokens:
        return ""
    parts: List[str] = [tokens[0]]
    for tok in tokens[1:]:
        # Punctuation-only tokens glue to the previous word
        if all(ch in string.punctuation for ch in tok):
            parts.append(tok)
        else:
            parts.append(" " + tok)
    return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Sentiment detector — keyword-based, zero parameters
# ═══════════════════════════════════════════════════════════════════════

# Positive keywords signalling caregiver approval
_POSITIVE_KW: set = {
    "good", "great", "yes", "yeah", "correct", "right", "nice",
    "thanks", "thank", "awesome", "wonderful", "exactly", "agree",
    "perfect", "ok", "okay", "sure", "love", "like", "excellent",
    "well", "fine",
}

# Negative keywords signalling caregiver disapproval
_NEGATIVE_KW: set = {
    "no", "bad", "wrong", "nope", "stop", "terrible", "awful",
    "hate", "dislike", "incorrect", "fail", "ugly", "dumb", "stupid",
    "never", "worse", "worst",
}


def detect_sentiment(tokens: List[str]) -> int:
    """
    Classify token list as positive (+1), negative (−1), or neutral (0).

    Uses a simple majority-vote over keyword hits.  No ML model.

    Returns
    -------
    +1  if net positive keywords > net negative keywords
    −1  if net negative > net positive
     0  otherwise (neutral / ambiguous)
    """
    pos = sum(1 for t in tokens if t in _POSITIVE_KW)
    neg = sum(1 for t in tokens if t in _NEGATIVE_KW)
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


# ═══════════════════════════════════════════════════════════════════════
# Chatbot Environment
# ═══════════════════════════════════════════════════════════════════════

class ChatbotEnv:
    """
    Conversational environment for the gradient-free Active Inference agent.

    Lifecycle per turn
    ──────────────────
    1.  User types a message  →  ``env.observe(user_text)``
    2.  Agent generates tokens →  ``env.act(agent_tokens)``
    3.  Environment evaluates homeostatic drives and returns ``info``.

    Homeostatic drives
    ──────────────────
    • **Energy**    — monotonically decreases each turn; the agent must
                      "earn" energy by producing non-trivial responses
                      that elicit positive caregiver feedback.
    • **Affiliation** — social bonding drive; increases with positive
                        feedback, decreases with silence or repetition.

    Anti-Dark-Room mechanism
    ────────────────────────
    If the agent outputs an empty string  *or*  repeats the exact same
    response ``repeat_tolerance`` times consecutively, both Energy and
    Affiliation receive a **massive penalty** (``dark_room_penalty``).

    This penalty enters the preference prior C(o) of the Active Inference
    engine so that silence becomes the *most surprising* outcome —
    mathematically compelling the agent to communicate.

    Parameters (from ChatConfig dataclass)
    ──────────────────────────────────────
    initial_energy          : int     Starting energy.
    initial_affiliation     : float   Starting affiliation score.
    energy_cost_per_turn    : int     Passive energy drain each turn.
    energy_gain_positive    : int     Energy earned from positive feedback.
    energy_loss_negative    : int     Energy lost from negative feedback.
    affiliation_gain        : float   Affiliation bump from positive feedback.
    affiliation_decay       : float   Multiplicative decay per turn.
    dark_room_penalty       : float   Penalty for silence / repetition.
    repeat_tolerance        : int     Consecutive identical replies before penalty.
    num_obs_buckets         : int     Hash buckets for observation index.
    """

    def __init__(self, cfg, rng: np.random.Generator) -> None:
        """
        Parameters
        ----------
        cfg : ChatConfig
            Environment hyper-parameters (see config.py).
        rng : numpy.random.Generator
            Seeded RNG for any stochastic elements.
        """
        self.cfg = cfg
        self.rng = rng

        # ── Homeostatic drives ────────────────────────────────────────
        self.energy: int = cfg.initial_energy
        self.affiliation: float = cfg.initial_affiliation

        # ── Conversation history (for repetition detection) ───────────
        # Stores the last ``repeat_tolerance`` agent responses as strings.
        self._recent_responses: deque = deque(
            maxlen=cfg.repeat_tolerance + 1
        )

        # ── Turn counter ──────────────────────────────────────────────
        self.turn: int = 0

        # ── Latest user tokens (set by observe()) ────────────────────
        self._user_tokens: List[str] = []
        self._user_sentiment: int = 0

        # ── Vocabulary registry (for unknown-word detection) ──────────
        # Maps word → node ID assigned by the memory graph.  Populated
        # externally by the agent when new words are encountered.
        self.vocab: Dict[str, int] = {}

    # ── Observe: process incoming user message ────────────────────────

    def observe(self, user_text: str) -> Dict:
        """
        Tokenise the user's message and prepare the observation.

        Returns
        -------
        obs : dict
            "tokens"    : List[str]   — tokenised user input.
            "sentiment" : int         — +1 / 0 / −1 sentiment of user.
            "unknown"   : List[str]   — words not yet in agent vocab.
            "obs_idx"   : int         — discrete observation index (hash bucket).
            "turn"      : int         — current turn number.
        """
        self.turn += 1
        self._user_tokens = tokenise(user_text)
        self._user_sentiment = detect_sentiment(self._user_tokens)

        # ── Identify unknown words (triggers BME in the agent) ────────
        unknown = [w for w in self._user_tokens if w not in self.vocab]

        # ── Discrete observation index: hash of token tuple mod buckets ─
        # This is a purely combinatorial mapping — no embeddings.
        token_hash = hash(tuple(self._user_tokens)) % self.cfg.num_obs_buckets
        obs_idx = token_hash

        return {
            "tokens": self._user_tokens,
            "sentiment": self._user_sentiment,
            "unknown": unknown,
            "obs_idx": obs_idx,
            "turn": self.turn,
        }

    # ── Act: evaluate agent's response against homeostatic drives ─────

    def act(self, agent_tokens: List[str]) -> Dict:
        """
        Accept the agent's generated token sequence and compute
        homeostatic drive updates.

        Parameters
        ----------
        agent_tokens : List[str]
            The word tokens produced by the agent's policy layer.

        Returns
        -------
        info : dict
            "response"          : str     — detokenised agent reply.
            "energy"            : int     — updated energy level.
            "affiliation"       : float   — updated affiliation score.
            "dark_room_hit"     : bool    — True if anti-dark-room triggered.
            "sentiment"         : int     — sentiment of the *previous* user msg.
            "turn"              : int     — current turn number.
            "penalty_applied"   : float   — total penalty applied this turn.
        """
        response_str = detokenise(agent_tokens)
        penalty: float = 0.0
        dark_room_hit = False

        # ── Anti-Dark-Room check ──────────────────────────────────────
        # Condition 1: empty response
        is_empty = len(agent_tokens) == 0

        # Condition 2: exact repetition  K  times in a row
        self._recent_responses.append(response_str)
        is_repetition = (
            len(self._recent_responses) >= self.cfg.repeat_tolerance
            and len(set(
                list(self._recent_responses)[-self.cfg.repeat_tolerance:]
            )) == 1
            and len(self._recent_responses) >= self.cfg.repeat_tolerance
        )

        if is_empty or is_repetition:
            dark_room_hit = True
            penalty = self.cfg.dark_room_penalty
            # Apply penalty to BOTH homeostatic drives
            self.energy = max(0, self.energy - int(penalty))
            self.affiliation = max(0.0, self.affiliation - penalty)

        # ── Passive energy cost per turn ──────────────────────────────
        self.energy = max(0, self.energy - self.cfg.energy_cost_per_turn)

        # ── Affiliation decay  (multiplicative) ──────────────────────
        self.affiliation *= self.cfg.affiliation_decay

        # ── Caregiver sentiment feedback ──────────────────────────────
        # Applied based on the *previous* user message's sentiment,
        # which is evaluating the agent's *prior* response.
        if self._user_sentiment > 0:
            # Positive: "Good", "Yes" — reward
            self.energy += self.cfg.energy_gain_positive
            self.affiliation += self.cfg.affiliation_gain
        elif self._user_sentiment < 0:
            # Negative: "No", "Bad" — punish
            self.energy = max(0, self.energy - self.cfg.energy_loss_negative)
            self.affiliation = max(
                0.0, self.affiliation - self.cfg.affiliation_gain
            )

        return {
            "response": response_str,
            "energy": self.energy,
            "affiliation": self.affiliation,
            "dark_room_hit": dark_room_hit,
            "sentiment": self._user_sentiment,
            "turn": self.turn,
            "penalty_applied": penalty,
        }

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self) -> Dict:
        """
        Reset the environment to its initial state.

        Returns the observation dict for a blank opening turn with
        a system greeting prompt.
        """
        self.energy = self.cfg.initial_energy
        self.affiliation = self.cfg.initial_affiliation
        self._recent_responses.clear()
        self.turn = 0
        self.vocab.clear()
        # Return a synthetic "opening" observation
        return self.observe("hello")

    # ── Utility ───────────────────────────────────────────────────────

    def is_alive(self) -> bool:
        """
        True if the agent still has energy to continue.

        When energy hits 0 the conversation "dies" — analogous to the
        grid agent running out of energy.  This forces the agent to
        maintain engagement (anti-dark-room through thermodynamics).
        """
        return self.energy > 0

    @property
    def user_tokens(self) -> List[str]:
        """Most recent user tokens (read-only access for the agent)."""
        return self._user_tokens

    @property
    def user_sentiment(self) -> int:
        """Sentiment of the most recent user message."""
        return self._user_sentiment
