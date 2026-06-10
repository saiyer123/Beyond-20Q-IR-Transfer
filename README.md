# Beyond 20Q: Zero-Shot Transfer from Game-Trained Policies to IR Clarification

**Johns Hopkins University — Human-AI Collaboration Lab**

A research implementation showing that a policy trained on the 20 Questions game via PPO with domain randomization can **zero-shot transfer** to Information Retrieval (IR) clarification tasks — with no retraining, no task-specific data, and no architectural changes.

The key insight: by representing both environments with the **same 5-dimensional domain-agnostic feature vector per action**, the policy generalizes across domains it has never seen.

---

## Core Idea

Standard clarification models are trained on IR-specific data. This work shows that framing clarification as a sequential decision problem and training on a *simpler* domain (20Q) transfers to *harder* real-world IR tasks when both share the same observation representation.

**Feature vector (per action):**

| Index | Feature | Description |
|---|---|---|
| 0 | `split_balance` | min(p_yes, p_no) — how evenly a question splits candidates |
| 1 | `EIG` | Expected Information Gain |
| 2 | `frac_yes` | Fraction of items answering yes |
| 3 | `frac_no` | Fraction of items answering no |
| 4 | `viability` | Viable ratio (questions) or belief (guesses) |

Same features, same network, different domains → zero-shot transfer works.

---

## Environments

**TwentyQuestionsEnv**
- Procedurally generated ontologies (6–16 items, 4–10 attributes)
- Domain randomization every episode for generalization
- Supports noisy answerers (noise_level parameter)
- Gymnasium-compatible

**IRClarificationEnv**
- Identical feature format to TwentyQuestionsEnv
- Items = document facets / search intents
- Attributes = clarifying questions
- Evaluated on ClariQ (real conversational search benchmark)

---

## Training

PPO with domain randomization on procedurally generated 20Q ontologies:
- Every episode: random num_items ∈ [6,16], num_attributes ∈ [4,10], max_questions ∈ [3,6]
- PolicyNetwork processes `(num_actions, 5)` feature matrix → per-action scores
- GAE advantage estimation, clipped surrogate objective, entropy regularization
- 500 iterations, 20 episodes per iteration

---

## Experiments

**Zero-Shot Transfer to IR (k = 2, 3, 4 questions)**
Evaluates trained policy on 500 IR scenarios without any fine-tuning. Compares against:
- Random baseline
- Greedy EIG (always picks highest expected information gain)
- Policy (20Q-trained, zero-shot)

**ClariQ Benchmark**
Loads real ClariQ dataset (conversational clarification for web search), converts to IR config format, and evaluates all three baselines. Includes transcript visualization showing policy decision reasoning.

**Noise Robustness**
Evaluates at noise_level ∈ {0.0, 0.1, 0.2, 0.3} — how well the policy holds up when answerers give incorrect responses.

**Few-Shot Fine-tuning**
`kshot_finetune()` — fine-tunes on k IR episodes (default k=100) to measure how quickly the zero-shot policy adapts with minimal data.

**Confidence Intervals**
Bootstrap 95% CIs on transfer results across K ∈ {2, 3, 4}.

---

## Policy Architecture

```
Input: (num_actions, 5) feature matrix

action_scorer:
  Linear(5 → 64) → LayerNorm → ReLU
  Linear(64 → 64) → LayerNorm → ReLU
  Linear(64 → 1)                        → per-action logits

value_head (max-pooled features):
  Linear(5 → 64) → ReLU → Linear(64 → 1)   → state value
```

Variable-size action space handled naturally — the same network scores any number of actions.

---

## Requirements

```bash
pip install numpy torch matplotlib tqdm gymnasium scipy pandas
git clone https://github.com/aliannejadi/ClariQ.git   # for ClariQ experiments
```

Run on Colab GPU (T4) — full pipeline takes ~20–30 min.

---

## Related Work

- ClariQ dataset: Aliannejadi et al. (2021), SIGIR
- Expected Information Gain for question selection: classical information-theoretic approach
- Domain randomization for generalization: OpenAI's work on sim-to-real transfer
- PPO: Schulman et al. (2017)
