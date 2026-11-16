# Deployment

## Docker

The `Dockerfile` at the repo root installs `nirnay-450m` from PyPI and
runs `nirnay serve` on port 8000. Weights download once into the
`/data/hf` volume.

```bash
docker build -t nirnay-450m:0.2.1 .
docker volume create nirnay_models
docker run --rm -p 8000:8000 -v nirnay_models:/data/hf nirnay-450m:0.2.1
curl localhost:8000/healthz
```

Status: Dockerfile written, but the build was **not verified** on this
machine. OrbStack was started, the build ran for 30 minutes on package
downloads and produced no image, so we are not claiming it works.
Verify on a machine with a warm Docker cache before publishing the
command.

## Space

The demo lives at
https://huggingface.co/spaces/GautamKishore/nirnay-demo and bakes
`phase_b.pt` in the repo. First request measured 12.4s, warm 1.8s.

## Server directly

```bash
pip install nirnay-450m
nirnay serve --host 0.0.0.0 --port 8000
```

The first start downloads the 466MB checkpoint into the Hugging Face
cache. After that it runs offline with `HF_HUB_OFFLINE=1`.
