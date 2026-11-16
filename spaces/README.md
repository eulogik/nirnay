# NIRNAY demo Space (LIVE)

Live: https://huggingface.co/spaces/GautamKishore/nirnay-demo
(ZeroGPU, free tier, under the personal account since orgs need Team
for Gradio Spaces; both ZeroGPU slots were free. See MEMORY.md.)

The Space repo now bakes `phase_b.pt` (466MB) so a cold start does not
re-download the checkpoint. The Laya base still comes from the Hub on
first request. Dependencies pin `nirnay-450m` from git main; bump the
pin or the package version when the API changes.

Local test: `NIRNAY_CHECKPOINT=/path/to/phase_b.pt python app.py`.
No HF_TOKEN secret is needed (model repo is public).
