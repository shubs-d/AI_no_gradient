"""
core_agent.py
=============
Main cognitive loop that binds together:

  1. Active Inference Engine   → belief updating, action selection
  2. Memory Graph              → spreading activation, binary features
  3. Tsetlin Machine           → logic-based policy learning  (O(C×L))
  4. Structure Learner         → BME expansion / BMR pruning
  5. ** Policy Layer **         → SBCP text generation  (chatbot mode)

The agent supports TWO operating modes:
  • **Grid mode** (legacy):  sense → infer → remember → decide → learn
    cycle for the 2-D non-stationary grid world.
  • **Chat mode** (v2):  observe user tokens → activate lexical graph →
    infer → generate response → apply social feedback → learn.

No gradient computation, no floating-point weight matrices — all learning
is count-based or logic-based.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import Config, NUM_ACTIONS
from agent.active_inference import ActiveInferenceEngine
from agent.memory_graph import MemoryGraph, extract_triplets
from agent.tsetlin_logic import TsetlinMachine
from agent.structure_learning import StructureLearner
from agent.policy_layer import (
    SBCPolicy, EOS_TOKEN, BOS_TOKEN, build_syntactic_features,
)


class CognitiveAgent:
    """
    Gradient-free cognitive agent that combines discrete active inference,
    spreading-activation memory, and Tsetlin-machine logic.

    Supports both grid-world and chatbot operating modes.
    """

    def __init__(self, cfg: Config, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

        # ── Module instantiation ──────────────────────────────────────
        # Start with a modest number of hidden states;
        # structure learning will grow it as needed.
        self._initial_states = 16

        self.inference = ActiveInferenceEngine(
            num_states=self._initial_states,
            cfg=cfg.inference,
        )
        self.memory = MemoryGraph(cfg.memory)
        self.tsetlin = TsetlinMachine(cfg.tsetlin, rng)
        self.structure = StructureLearner(cfg.structure)

        # ── Policy Layer (chatbot mode) ───────────────────────────────
        self.policy = SBCPolicy(
            dirichlet_prior=cfg.policy.dirichlet_prior,
            max_response_len=cfg.policy.max_response_len,
            eos_prior_boost=cfg.policy.eos_prior_boost,
            omega=cfg.policy.policy_omega,
            eta=cfg.policy.policy_eta,
            feedback_strength=cfg.policy.feedback_strength,
            target_mass=cfg.policy.target_mass,
            precision_beta=cfg.policy.precision_beta,
            use_dcm=cfg.policy.use_dcm,
            clause_boost_scale=cfg.policy.clause_boost_scale,
            echo_copy_penalty=cfg.policy.echo_copy_penalty,
            rng=rng,
        )

        # Seed the memory graph with observation symbol nodes (grid mode)
        for i in range(cfg.inference.num_obs_symbols):
            self.memory.add_node(label=f"obs_{i}")

        # Seed special tokens for chatbot mode
        self._bos_id = self.memory.get_or_create_word_node(BOS_TOKEN)
        self._eos_id = self.memory.get_or_create_word_node(EOS_TOKEN)
        self.policy.bos_id = self._bos_id
        self.policy.eos_id = self._eos_id

        # ── Tracking variables ────────────────────────────────────────
        self._prev_state: int = 0            # MAP state at t−1
        self._prev_action: int = 0           # action at t−1
        self._step: int = 0
        self._last_obs_idx: int = 0

    # ══════════════════════════════════════════════════════════════════
    # Grid-world mode  (legacy — preserved for benchmarks)
    # ══════════════════════════════════════════════════════════════════

    def act(
        self,
        ego_view: np.ndarray,
        internal: np.ndarray,
        obs_idx: int,
    ) -> Tuple[int, float]:
        """
        Full cognitive cycle for grid-world mode.

        Parameters
        ----------
        ego_view : ndarray   5×5 egocentric grid view (informational).
        internal : ndarray   1-D internal state (energy etc.).
        obs_idx  : int       Dominant observation type in the view.

        Returns
        -------
        action : int          Chosen action.
        vfe    : float        Variational Free Energy (surprise).
        """
        self._step += 1

        # ── 1. SENSE: ground observation in memory graph ──────────────
        if obs_idx < self.memory.num_nodes:
            self.memory.inject_observation(obs_idx, strength=2.0)

        # Inject energy as activation on node 0 (a proxy)
        energy_strength = float(internal[0]) * 2.0
        if self.memory.num_nodes > 0:
            self.memory.inject_observation(0, strength=energy_strength)

        # ── 2. INFER: update beliefs using active inference ───────────
        vfe = self.inference.update_belief(obs_idx)

        # ── 3. REMEMBER: propagate activation through memory graph ───
        self.memory.step()

        # Hebbian edge strengthening between previous & current obs nodes
        cur_state = self.inference.most_likely_state()
        if self._step > 1 and obs_idx < self.memory.num_nodes:
            prev_obs_node = min(self._prev_state, self.memory.num_nodes - 1)
            self.memory.strengthen_edge(prev_obs_node, obs_idx)

        # ── 4. DECIDE: combine active-inference EFE + Tsetlin vote ───
        # Get binary features from the memory graph
        binary_features = self.memory.get_binary_features(
            self.cfg.tsetlin.num_literals
        )

        # Active-inference preferred action (epistemic + pragmatic)
        ai_action = self.inference.select_action()

        # Tsetlin machine vote
        tm_action = self.tsetlin.predict(binary_features)

        # Blend: use AI action primarily, but let TM override when its
        # vote confidence is high relative to the threshold
        votes = self.tsetlin.vote(binary_features)
        max_vote = int(votes.max())
        T = self.cfg.tsetlin.threshold

        if max_vote > T // 2:
            # TM is confident — use its suggestion
            action = tm_action
        else:
            # Fall back to active-inference EFE
            action = ai_action

        # Exploration: small ε-random chance to override
        if self.rng.random() < max(0.01, 0.1 * (vfe / 5.0)):
            action = int(self.rng.integers(NUM_ACTIONS))

        # ── 5. LEARN ─────────────────────────────────────────────────

        # 5a. Update Dirichlet counts (pure counting)
        self.inference.update_counts(
            prev_state=self._prev_state,
            action=self._prev_action,
            next_state=cur_state,
            obs_idx=obs_idx,
        )

        # 5b. Update Tsetlin Machine (VFE-driven feedback)
        self.tsetlin.update(binary_features, target_action=ai_action, vfe=vfe)

        # 5c. Structure learning
        self.structure.record_vfe(vfe)

        #   BME: expand if cumulative surprise is too high
        if self.structure.should_expand():
            new_node = self.memory.add_node(
                label=f"auto_{self.memory.num_nodes}"
            )
            # Expand AI state space only up to max_ai_states cap
            max_states = self.cfg.structure.max_ai_states
            if self.inference.num_states < max_states:
                target = min(
                    max(self.inference.num_states + 1,
                        self.memory.num_nodes),
                    max_states,
                )
                self.inference.expand_state_space(target)
            # ── Dynamically grow Tsetlin literal/clause pool ────────
            new_lit = max(
                self.tsetlin.current_num_literals,
                self.memory.num_nodes,
            )
            self.tsetlin.expand_literals(new_lit)
            self.structure.acknowledge_expansion()

        #   BMR: periodic MDL pruning
        if self.structure.should_prune():
            self.structure.mdl_prune_graph(
                self.memory, self.tsetlin, recent_vfe=vfe
            )

        # ── Bookkeeping ──────────────────────────────────────────────
        self._prev_state = cur_state
        self._prev_action = action

        return action, vfe

    # ══════════════════════════════════════════════════════════════════
    # Chatbot mode  (v2 — NLP pivot)
    # ══════════════════════════════════════════════════════════════════

    def chat_observe(
        self,
        obs: Dict,
    ) -> Tuple[List[str], float]:
        """
        Full cognitive cycle for chatbot mode: observe user tokens,
        ground them in the memory graph, run spreading activation,
        generate a response, and learn.

        Parameters
        ----------
        obs : dict
            Output of ``ChatbotEnv.observe()``.  Keys:
              "tokens"    : List[str]   — tokenised user input.
              "sentiment" : int         — +1 / 0 / −1.
              "unknown"   : List[str]   — words not in vocab.
              "obs_idx"   : int         — discrete observation index.
              "turn"      : int         — turn number.

        Returns
        -------
        response_tokens : List[str]
            The token sequence generated by the agent's policy.
        vfe : float
            Variational Free Energy (surprise) for this observation.
        """
        self._step += 1
        tokens: List[str] = obs["tokens"]
        sentiment: int = obs["sentiment"]
        unknown: List[str] = obs["unknown"]
        obs_idx: int = obs["obs_idx"]
        chat_cfg = self.cfg.chat

        # ── 1. SENSE: ground user tokens in lexical memory graph ──────
        # Map each token to a node ID (creating new nodes as needed)
        token_node_ids: List[int] = []
        for tok in tokens:
            nid = self.memory.get_or_create_word_node(tok)
            token_node_ids.append(nid)

        # Inject activation spikes on all user-token nodes
        self.memory.inject_token_activations(
            token_node_ids,
            strength=chat_cfg.activation_spike_strength,
        )

        # Learn co-occurrence edges from the token sequence
        self.memory.learn_cooccurrences(
            token_node_ids,
            window=chat_cfg.cooccurrence_window,
        )

        # ── 1b. TRIPLET EXTRACTION: learn semantic S→P→O links ───────
        user_text = " ".join(tokens)
        grammar_cfg = self.cfg.grammar
        triplet_ids = self.memory.learn_triplets_from_text(
            user_text,
            strength=grammar_cfg.triplet_edge_strength,
        )

        # ── 2. INFER: update beliefs ─────────────────────────────────
        # Ensure obs_idx fits the observation space
        if obs_idx >= self.inference.num_obs:
            self.inference.expand_obs_space(obs_idx + 1)
        vfe = self.inference.update_belief(obs_idx)

        # ── 3. REMEMBER: propagate spreading activation ──────────────
        #    A(t+1) = λ A(t) + W A(t)
        self.memory.step()

        # Get TopK highest-activated nodes = contextual comprehension
        top_k_pairs = self.memory.get_contextual_top_k(k=chat_cfg.top_k)
        node_labels = self.memory.get_all_labels()
        # Filter out non-lexical scaffold nodes from language generation.
        top_k_ids = []
        activation_scores = {}
        for nid, act in top_k_pairs:
            lbl = node_labels.get(nid, "")
            if not lbl:
                continue
            if lbl in {BOS_TOKEN, EOS_TOKEN, "<UNK>"}:
                continue
            if lbl.startswith("obs_"):
                continue
            top_k_ids.append(nid)
            activation_scores[nid] = act

        # ── 4. BME: handle unknown words ─────────────────────────────
        if unknown:
            new_ids = self.structure.handle_unknown_words(
                unknown_words=unknown,
                memory_graph=self.memory,
                current_context_ids=token_node_ids,
                inference_engine=self.inference,
            )
            # Dynamically grow Tsetlin literal pool
            if new_ids:
                new_lit = max(
                    self.tsetlin.current_num_literals,
                    self.memory.num_nodes,
                )
                self.tsetlin.expand_literals(new_lit)

        # ── 5. GENERATE: Sparse Bayesian Competitive Policy ──────────
        # Observe user token bigrams to learn language structure
        self.policy.observe_user_tokens(token_node_ids)

        # localised Dirichlet forgetting on the active sub-graph
        active_ids = self.memory.get_active_subgraph_ids()
        self.policy.apply_localised_forgetting(active_ids)

        # ── 5b. Compute clause votes for candidate words ──────────
        # Build binary features from the current memory graph state
        # and get per-candidate TM votes to modulate Dirichlet alpha.
        binary_features = self.memory.get_binary_features(
            self.cfg.tsetlin.num_literals
        )
        clause_votes_raw = self.tsetlin.vote(binary_features)  # (A,)
        # Expand to candidate-length (TopK + EOS): map action votes
        # to word candidates by distributing the most relevant vote.
        # Each candidate gets the max TM vote (action-agnostic boost).
        clause_vote_best = int(clause_votes_raw.max())
        clause_votes_for_gen = np.full(
            len(top_k_ids) + 1,  # +1 for EOS
            clause_vote_best,
            dtype=np.float64,
        )

        # Generate response via precision-scaled Bayesian sampling
        response_tokens = self.policy.generate(
            top_k_ids=top_k_ids,
            node_labels=node_labels,
            activation_scores=activation_scores,
            grammar_boost=True,
            role_boost_subject=grammar_cfg.role_boost_subject,
            role_boost_predicate=grammar_cfg.role_boost_predicate,
            role_boost_object=grammar_cfg.role_boost_object,
            clause_votes=clause_votes_for_gen,
            clause_threshold=self.cfg.tsetlin.threshold,
            avoid_token_ids=set(token_node_ids),
        )

        # Identity-query retrieval override (count-based, no gradients):
        # For prompts like "what is my name", retrieve the strongest
        # learned continuation of the chain my→name→is→X from policy
        # Dirichlet bigram counts.
        if self._is_name_query(tokens):
            retrieved = self._retrieve_name_statement(node_labels)
            if retrieved is not None:
                response_tokens = retrieved

        # Coherence fallback: if policy output is too short/repetitive,
        # synthesise a minimal grammatical response from learned triplets.
        if self._is_low_quality_response(response_tokens):
            response_tokens = self._fallback_response_from_context(
                triplet_ids=triplet_ids,
                top_k_ids=top_k_ids,
                node_labels=node_labels,
            )

        # ── 6. LEARN: update Dirichlet counts ────────────────────────
        cur_state = self.inference.most_likely_state()

        # 6a. Standard Dirichlet count update
        self.inference.update_counts(
            prev_state=self._prev_state,
            action=self._prev_action % NUM_ACTIONS,
            next_state=cur_state,
            obs_idx=obs_idx % self.inference.num_obs,
        )

        # 6b. Social Active Inference: apply caregiver feedback
        if sentiment != 0:
            self.inference.apply_social_feedback(
                sentiment=sentiment,
                current_state=cur_state,
                obs_idx=obs_idx % self.inference.num_obs,
            )
            # Also adjust policy bigram counts
            self.policy.apply_feedback(sentiment)

        # 6c. Localised Dirichlet forgetting on AI engine
        # Map active graph nodes to active hidden states via belief
        active_states: Set[int] = set()
        for s in range(self.inference.num_states):
            if self.inference.belief[s] > 0.05:
                active_states.add(s)
        self.inference.apply_localised_forgetting(active_states)

        # 6d. Structure learning (VFE accumulation + BME/BMR)
        self.structure.record_vfe(vfe)

        # Tsetlin Machine binary-feature update (with grammar features)
        binary_features = self.memory.get_binary_features(
            self.cfg.tsetlin.num_literals
        )
        ai_action = self.inference.select_action()
        # Build syntactic features from the response tokens for TM learning
        syn_feat = build_syntactic_features(response_tokens)
        self.tsetlin.update_with_grammar(
            binary_features, syn_feat,
            target_action=ai_action, vfe=vfe,
        )

        # General BME (from cumulative VFE)
        if self.structure.should_expand():
            self.memory.add_node(
                label=f"auto_{self.memory.num_nodes}"
            )
            max_states = self.cfg.structure.max_ai_states
            if self.inference.num_states < max_states:
                target = min(
                    max(self.inference.num_states + 1,
                        self.memory.num_nodes),
                    max_states,
                )
                self.inference.expand_state_space(target)
            new_lit = max(
                self.tsetlin.current_num_literals,
                self.memory.num_nodes,
            )
            self.tsetlin.expand_literals(new_lit)
            self.structure.acknowledge_expansion()

        # BMR
        if self.structure.should_prune():
            self.structure.mdl_prune_graph(
                self.memory, self.tsetlin, recent_vfe=vfe
            )

        # Edge-level MDL pruning (separate, less frequent cadence)
        if self.structure.should_prune_edges():
            self.structure.mdl_prune_edges(
                self.memory, recent_vfe=vfe
            )

        # ── Bookkeeping ──────────────────────────────────────────────
        self._prev_state = cur_state
        self._prev_action = ai_action
        self._last_obs_idx = obs_idx

        return response_tokens, vfe

    def _is_low_quality_response(self, tokens: List[str]) -> bool:
        """Heuristic gate for degenerate outputs (empty/short/repetitive)."""
        if not tokens:
            return True
        if len(tokens) < 2:
            return True
        uniq = len(set(tokens))
        # Low lexical diversity indicates looping / babbling.
        if uniq <= 1:
            return True
        diversity = uniq / max(len(tokens), 1)
        if diversity < 0.55:
            return True
        if len(tokens) >= 8 and diversity < 0.70:
            return True
        # If one token dominates the utterance, treat as babbling.
        counts: Dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        if max(counts.values()) / max(len(tokens), 1) > 0.60:
            return True
        return False

    def _is_name_query(self, user_tokens: List[str]) -> bool:
        """Detect identity queries such as 'what is my name'."""
        tok = [t.lower() for t in user_tokens]
        if "name" not in tok:
            return False
        if "what" in tok:
            return True
        if "who" in tok and "i" in tok:
            return True
        if "remember" in tok and "name" in tok:
            return True
        return False

    def _retrieve_name_statement(
        self,
        node_labels: Dict[int, str],
    ) -> Optional[List[str]]:
        """
        Retrieve the most likely learned identity phrase using the
        bigram-count chain:  my → name → is → X.
        """
        my_id = self.memory.word_to_id("my")
        name_id = self.memory.word_to_id("name")
        is_id = self.memory.word_to_id("is")
        if my_id is None or name_id is None or is_id is None:
            return None

        # Validate that my→name and name→is are supported by counts.
        my_to_name = self.policy._get_count(my_id, name_id)
        name_to_is = self.policy._get_count(name_id, is_id)
        prior = self.policy.dirichlet_prior
        if my_to_name <= prior * 2.0 or name_to_is <= prior * 2.0:
            return None

        # Find strongest continuation of is→X.
        row = self.policy._bigram.get(is_id, {})
        if not row:
            return None

        best_dst = None
        best_count = -1.0
        for dst, cnt in row.items():
            label = node_labels.get(dst, "")
            if label in {"<BOS>", "<EOS>", "<UNK>", "my", "name", "is"}:
                continue
            if cnt > best_count:
                best_count = cnt
                best_dst = dst

        if best_dst is None or best_count <= prior * 2.0:
            return None

        name_token = node_labels.get(best_dst, "").strip()
        if not name_token:
            return None

        return ["my", "name", "is", name_token]

    def _fallback_response_from_context(
        self,
        triplet_ids: List[Tuple[int, int, int]],
        top_k_ids: List[int],
        node_labels: Dict[int, str],
    ) -> List[str]:
        """
        Build a concise, grammatical fallback response from semantic
        triplets first, then from active lexical context if no triplet
        is available.
        """
        special = {BOS_TOKEN, EOS_TOKEN, "<UNK>"}

        if triplet_ids:
            s_id, p_id, o_id = triplet_ids[-1]
            s = node_labels.get(s_id, "")
            p = node_labels.get(p_id, "")
            o = node_labels.get(o_id, "")
            toks = [t for t in [s, p, o] if t and t not in special]
            if len(toks) >= 2:
                return toks

        # Top-k lexical fallback
        out: List[str] = []
        for nid in top_k_ids:
            w = node_labels.get(nid, "")
            if not w or w in special:
                continue
            if w in out:
                continue
            out.append(w)
            if len(out) >= 4:
                break

        if len(out) >= 2:
            return out
        if out:
            return ["i", "understand", out[0]]
        return ["i", "am", "listening"]

    # ── Diagnostics ───────────────────────────────────────────────────

    def get_diagnostics(self) -> Dict:
        """
        Return a snapshot of the agent's internal complexity metrics,
        including Dirichlet phase diagnostics (v4).
        """
        return {
            "num_memory_nodes": self.memory.num_nodes,
            "num_hidden_states": self.inference.num_states,
            "active_clauses": self.tsetlin.total_clauses_active(),
            "available_clauses": self.tsetlin.total_clauses_available(),
            "clause_saturation": self.tsetlin.clause_saturation_ratio(),
            "lyapunov_energy": self.memory.lyapunov_energy(),
            "model_bits": StructureLearner.model_description_length(
                self.memory, self.tsetlin
            ),
            "tsetlin_int_ops": self.tsetlin.int_ops,
            "bigram_entries": self.policy.total_bigram_entries(),
            "triplet_count": self.memory.triplet_count,
            # ── Dirichlet phase diagnostics (v4) ─────────────────────
            "policy_entropy_ratio": getattr(
                self.policy, '_last_entropy_ratio', 1.0
            ),
            "thompson_gap": getattr(
                self.policy, '_last_thompson_gap', 0.0
            ),
        }
