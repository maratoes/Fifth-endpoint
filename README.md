# Fifth Endpoint (Experimental BU-30B-A3B)

Experimental RunPod endpoint for `browser-use/bu-30b-a3b-preview` on vLLM.

## Build

```bash
./scripts/build.sh
```

## Push

```bash
./scripts/push.sh
```

## RunPod settings

- GPU: NVIDIA A100 80GB
- Workers: min `0`, max `1`
- Idle timeout: `120s`
- Env:
  - `MODEL_NAME=browser-use/bu-30b-a3b-preview`
  - `MAX_MODEL_LEN=65536`
  - `GPU_MEMORY_UTILIZATION=0.90`
  - `TRUST_REMOTE_CODE=True`
  - `DTYPE=float16`
  - `HF_TOKEN=<token>`
