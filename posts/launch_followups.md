# Follow-up posts (Oct 5, after launch week 1 flatline)

Context: Reddit localllama + LinkedIn + X are live, HN was killed on
arrival (no trace in HN search, likely posted while repo was private or
flagged as promo), traction flat (GH 9 stars). Do NOT repost on HN yet.
Do NOT make new Reddit threads (reposts get removed, comment instead).

## 1. Reddit r/LocalLLaMA: comment on your own post (not a new thread)

Update: ran the same-metric check against Cloudflare's Decision Index
table. Macro-F1, measured on our side on the same 3,080 cases: Clef 27B
0.942 / Flash 9B 0.909 / ours 0.8785 / Kev 0.848 / Jev 0.797. Per-class
scores are in eval/banking77_phase_b_macro.json in the repo. Same metric
family, different harnesses, so treat it as a cross-check, not a
leaderboard. We beat Jev by 8 and Kev by 3 at 60x/20x the size, on a Mac
instead of an H200.

## 2. LinkedIn follow-up post

Cloudflare just walked into our lane.

Their Clef 27B posts 94.2 macro-F1 on Banking77. Flash 9B posts 90.9.
Big models, H200s, full Cloudflare distribution behind them.

We ran the same metric on our 450M: 87.9. Same 3,080 cases, measured
here, per-class scores in the repo.

So the honest table reads: Clef 94.2 / Flash 90.9 / NIRNAY 87.9 /
Kev 84.8 / Jev 79.7. We trail the Cloudflare pair and beat everyone
else, at 60x fewer params, trained on one Mac, $0 cloud, Apache-2.0.

Caveats, as always: their suite vs ours (cross-check, not leaderboard),
and we fine-tuned while Jev answered zero-shot.

Giants validating your category is the best marketing you cannot buy.
We are the tiny open option in it. Repo and weights in comments.

## 3. X follow-up (3 posts, reply to your own thread)

1/3
Cloudflare entered typed decisions: Clef 27B posts 94.2 macro-F1 on
Banking77. We ran the same metric on our 450M: 87.85. Same 3,080 cases.

2/3
Same-metric row: Clef 94.2 / Flash 90.9 / NIRNAY 87.9 / Kev 84.8 /
Jev 79.7. We beat Jev by 8 at 60x smaller, on a Mac instead of an H200.

3/3
Caveats: their suite vs ours, cross-check not leaderboard. We
fine-tuned, Jev zero-shot. Per-class scores open in the repo:
github.com/eulogik/nirnay

## 4. HN: wait, then come back stronger

Do not resubmit the same link now, it gets flagged as a duplicate and
burns the account. Rule of thumb: one retry allowed with real news.
Ours: live Space demo + the macro-F1 row + Cloudflare angle, in 1-2
weeks. New title then, something like: Show HN: 450M open decision
model holds 87.9 macro-F1 next to Cloudflare's 27B. If you want the
actual removal reason, mail hn@ycombinator.com, they answer.

## 5. Demo is live (post these now, Oct 5)

Space verified end to end through the public API (correct prediction,
0.96 confidence). Link: https://huggingface.co/spaces/GautamKishore/nirnay-demo

X (2 posts, reply to your thread):

1/2
Live demo is up, no install. Type a banking message, get the intent
plus probabilities: https://huggingface.co/spaces/GautamKishore/nirnay-demo
2/2
Runs free on ZeroGPU. Same 450M weights as the release (87.9%
Banking77, ECE 0.045). Break it and tell me where.

LinkedIn (short follow-up):

The demo is live, no install needed. Type a customer message, get the
intent plus calibrated probabilities, right in the browser:
https://huggingface.co/spaces/GautamKishore/nirnay-demo
Same 450M Apache-2.0 weights as the release. If you run triage or
routing, try your three hardest messages on it and post what happens.

## The real traction fix (bigger than any post)

Text posts without a runnable thing die. The single highest-leverage
move left: put the Space live, then pin its URL in every thread above.
Second: run the Colab once yourself so the badge works first click.
