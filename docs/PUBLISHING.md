# Publishing nirnay-450m

**Status: 0.2.1 published 2026-11-16 (adds `nirnay fit`).**
https://pypi.org/project/nirnay-450m/

The distribution is `nirnay-450m`, not `nirnay`. The name `nirnay` belongs
to an unrelated Sarvam API wrapper on PyPI (Dobbyisus/nirnay, 0.2.0), and
it installs a top-level `nirnay` module. Do not publish under that name.

## One-time credentials

Create a PyPI API token for the `nirnay-450m` project (or a scoped token
for the account), then either:

```bash
export UV_PUBLISH_TOKEN=pypi-...
```

or write `~/.pypirc` with the token. Nothing is stored in this repo.

## Build and check

```bash
uv build
uvx twine check dist/*
```

## Publish

```bash
uv publish --token "$UV_PUBLISH_TOKEN"
```

If the package does not exist yet, `uv publish` creates it on first upload.
Expected artifacts:

```
dist/nirnay_450m-0.2.0-py3-none-any.whl
dist/nirnay_450m-0.2.0.tar.gz
```

## Verify from a clean venv

```bash
uv venv /tmp/nirnay-pypi-test
uv pip install --python /tmp/nirnay-pypi-test/bin/python nirnay-450m
/tmp/nirnay-pypi-test/bin/nirnay doctor
/tmp/nirnay-pypi-test/bin/nirnay classify "My card was charged twice" --labels banking77
```

The first classify downloads the 466MB checkpoint from
`eulogik/nirnay-450m` into the Hugging Face cache. One-time cost.

## Do not

* Do not upload to `nirnay`. It is not ours.
* Do not bump the version without rebuilding and re-running `twine check`.
* Do not publish from a dirty worktree.
