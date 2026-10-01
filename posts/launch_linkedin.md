# LinkedIn (Fri morning IST, company page + personal)

Routing customer tickets still means choosing between a closed API
(per-call bill, your data leaves the building) and an LLM that costs
too much per decision to run inline.

We built a third option and open-sourced it: NIRNAY 450M, Apache-2.0.

Banking77 intent, 3,080 test cases: 87.9% accuracy with fitted ECE
0.045. Same cases through a leading closed API: 80.3%. Ours runs on
your own hardware (209ms on Apple Silicon, 361ms CPU), no per-call
cost, no data leaving your walls. Trained on one Mac. Cloud bill: $0.

Caveat, stated openly: we fine-tuned on the domain, the API answered
zero-shot. For triage, routing, and guardrails, that is exactly the
trade worth making.

Repo + weights + evals: github.com/eulogik/nirnay,
huggingface.co/eulogik/nirnay-450m. Built by Eulogik (eulogik.com).
Limits published in the repo: 512-token context, long-doc reasoning
in progress.
