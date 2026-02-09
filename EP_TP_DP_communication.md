# Communication Pattern Summary: EP + TP + DP (Inference)

| Stage                                   | EP Communication                         | TP Communication                                         | DP Communication         |
|-----------------------------------------|-------------------------------------------|-----------------------------------------------------------|---------------------------|
| **1. Dense (non‑MoE) layers**           | ❌ none                                   | ✔️ all‑reduce / all‑gather / reduce‑scatter               | ❌ none                   |
| **2. Router (gating) computation**      | ❌ none                                   | ❌ none (usually replicated per TP rank)                  | ❌ none                   |
| **3. MoE token dispatch**               | ✔️ all‑to‑all across EP ranks             | ❌ none                                                   | ❌ none                   |
| **4. Expert computation**               | ❌ none                                   | ✔️ TP collectives inside each expert                      | ❌ none                   |
| **5. MoE token combine (gather)**       | ✔️ all‑to‑all across EP ranks             | ❌ none                                                   | ❌ none                   |
| **6. Output → next dense layer**        | ❌ none                                   | ✔️ TP collectives in next dense block                     | ❌ none                   |
| **Between DP replicas**                 | ❌ none                                   | ❌ none                                                   | ❌ none (inference only)  |