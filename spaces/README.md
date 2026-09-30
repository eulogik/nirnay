# NIRNAY demo Space (go-live checklist)

This folder stages the Hugging Face Space. It is NOT live yet: the model
repo stays private until verification finishes.

Go-live (2 steps, owner only):

1. Create Space `eulogik/nirnay-demo` (Gradio SDK, CPU basic, public),
   upload `app.py` + `requirements.txt` from this folder.
2. In Space Settings → Variables, add secret `HF_TOKEN` (read access to
   the private `eulogik/nirnay-450m` model repo, or flip that repo public
   first and drop the secret).

Then open the Space URL and classify. Tested locally 2026-09-30
(CPU + MPS, banking + garbage inputs, see MEMORY.md).
