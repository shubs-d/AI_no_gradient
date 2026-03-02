# Gradient-Free Active Inference Chatbot

A discrete, non-neural cognitive architecture for language learning and generation.

This project uses:
- **Active Inference** (Dirichlet count tables, Expected Free Energy)
- **Spreading-Activation Memory Graph** (Lyapunov-stable)
- **Tsetlin Machine Logic** (rule learning, no gradients)
- **Semantic Triplet Knowledge Graph** (Subject → Predicate → Object)
- **MDL Structure Learning** (expansion + pruning)

No backpropagation, no continuous neural weights, no transformer stack.

---

## 1) What this project is

This is a research-style, gradient-free language agent that learns from interaction and corpora.

It supports:
- Interactive chat mode
- Automated caregiver curriculum training
- Bulk corpus training
- Full checkpoint save/load

It does **not** start with pretrained world knowledge like LLM APIs.

---

## 2) Project structure

```text
AI_no_gradient/
├── agent/
│   ├── active_inference.py
│   ├── core_agent.py
│   ├── memory_graph.py
│   ├── policy_layer.py
│   ├── structure_learning.py
│   └── tsetlin_logic.py
├── environment/
│   ├── chatbot_env.py
│   └── non_stationary_grid.py
├── experiments/
│   └── continual_learning_bench.py
├── auto_caregiver.py
├── bulk_trainer.py
├── chat_main.py
├── config.py
├── main.py
└── persistence.py
```

---

## 3) Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
pip install numpy scipy matplotlib
```

(If your system Python is externally managed, use your distro/venv policy.)

---

## 4) Quick start

From project root:

```bash
cd /home/shubs/Coding/Python/AI_testing/gradient_free_ai/AI_no_gradient
```

### Run interactive chatbot

```bash
python3 chat_main.py --no-load --save checkpoints/chat_state.pkl
```

- Type messages in the terminal
- Type `exit` or `quit` to save and stop

### Resume from checkpoint

```bash
python3 chat_main.py --save checkpoints/chat_state.pkl
```

---

## 5) Training modes

### A) Caregiver curriculum training

```bash
python3 auto_caregiver.py --cycles 2000 --save checkpoints/caregiver_state.pkl
```

Use this for controlled conversational bootstrapping.

### B) Bulk corpus training

Self-test corpus:

```bash
python3 bulk_trainer.py --selftest --cycles 500 --checkpoint 100 --save checkpoints/bulk_state.pkl
```

Custom corpus file:

```bash
python3 bulk_trainer.py --file /path/to/corpus.txt --cycles 10000 --checkpoint 500 --save checkpoints/bulk_state.pkl
```

Resume:

```bash
python3 bulk_trainer.py --file /path/to/corpus.txt --cycles 10000 --load checkpoints/bulk_state.pkl --save checkpoints/bulk_state.pkl
```

---

## 6) Persistence

State is saved with `pickle` in `persistence.py`.

Saved components include:
- Memory graph (labels, counts, activation)
- Active inference tables (`A`, `B`, beliefs)
- Policy bigram counts
- Tsetlin automata state
- Structure learner state
- Optional environment homeostasis (energy/affiliation/turn)

---

## 7) Current language pipeline (chat mode)

For each user turn:
1. Tokenize + ground tokens in memory graph
2. Learn co-occurrence edges
3. Extract triplets and add typed S→P→O links
4. Belief update (Active Inference)
5. Spreading activation over graph
6. Generate response (Dirichlet Thompson sampling + grammar boost)
7. Apply feedback + forgetting + structure learning/pruning

---

## 8) Why responses are not fully LLM-like

This system is intentionally non-neural and starts near blank.

You can get good **simple-domain fluency** with curriculum and corpus training, but it will not match large pretrained transformer models on broad open-domain reasoning.

---

## 9) Suggested training strategy for better conversational quality

1. Start clean:

```bash
python3 chat_main.py --no-load --save checkpoints/simple_lang.pkl
```

2. Train with short, repetitive simple English patterns first:

```bash
python3 bulk_trainer.py --file /path/to/simple_english.txt --cycles 20000 --checkpoint 1000 --save checkpoints/simple_lang.pkl
```

3. Then chat on the same checkpoint:

```bash
python3 chat_main.py --save checkpoints/simple_lang.pkl
```

---

## 10) Safety + scope

This is a local experimental research agent. Validate behavior before using in production or safety-critical workflows.

---

## 11) License

Add your preferred license file if you plan to share/publicly distribute this repository.
