# R1.4 — query budget: derivation and verification

Two distinct quantities, not to be conflated:

1. **Evaluated candidates per instance = 1,260 (derived from the loop, stored per array).**
   In `generator.py::genetic_search` the budget variable `nq` starts at `POP = 60` and in each of
   `GENS = 40` generations adds `POP − ELITE = 30` new children:
   `nq = 60 + 40 × 30 = 1,260`. This is accumulated by the generator's own loop and written into
   every genetic-arm result array as the `queries` field. It is a derivation from the code's
   constants, not a separate profiling hook. Arm C (nearest real flow) is non-genetic, so its
   `queries` field is 0.

2. **predict_proba CALLS per cell = 41 (batched over all instances).**
   `score()` reshapes the `(N instances × k candidates × p)` array to `(N·k, p)` and issues ONE
   `predict_proba` per generation over the whole cell. With the initial population plus 40
   generations that is `1 + 40 = 41` model calls per cell of 1,000 instances — not per instance.

**Verification.** `exp3_reproduce_gate.py` re-reads the `queries` field from every saved array and
checks it equals the expected 1,260 for genetic arms (0 for C). The gate log
(`../logs/reproduce_gate.json`, `query_budget`) shows exactly the two distinct per-instance values
`{0, 1260}` across all 48 arm×cell rows. So "1,260 evaluated candidates per instance" is a
code-derived budget re-verified against the stored arrays; it is **not** an instrumented counter
wrapping `predict_proba`, and it is **not** the model-call count (41 per cell).
