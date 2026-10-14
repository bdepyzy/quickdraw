# Model references

The 15 fixtures contain synthetic inputs, expected outputs, and expected cache
updates. `manifest.json` records provenance and dependency versions. They use
Transformers attention/recurrence with independent ModelOpt precision adapters.
Default tolerances are `rtol=0.02`, `atol=0.002`.

```sh
uv run -m tests.models.mathcheck --strict
uv run -m tests.models.referencegen
```

GPU kernel checks consume the same fixtures. Full-model comparison uses the
separate saved vLLM capture at `references/engine/vllm.pt`.
