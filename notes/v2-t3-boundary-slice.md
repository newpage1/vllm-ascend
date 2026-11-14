# DSV4.1 layerwise v2 — T3 boundary / step / ring slice

Engine-side half of `docs/v2-api-contract.md` §6 for the vLLM-Ascend repo
(branch `dsv41_lw_v2`). This slice only adds the engine seam; the transport
codec (wire descriptor, staging, schema2 encode/decode) stays on the
LMCache-Ascend side and is not modified here.

## Files

| File | Change |
| --- | --- |
| `vllm_ascend/attention/dsv41_lw_v2_boundary.py` | new engine seam: DTOs, codec protocol, install, step/ring binding, group hook |
| `vllm_ascend/worker/v1/model_runner.py` | bind one forward step in `_model_forward` (cleared in `finally`), install the codec in `initialize_kv_cache` |
| `tests/ut/attention/test_dsv41_lw_v2_boundary.py` | 43 CPU-only tests |

The module imports no `dsv41_layerwise_*` module: no admission registry, no
lease, no snapshot lookup, no prefix key, and no `admit_step` call. Real
forward facts come only from the scheduler input batch, the live ring tensor
and the vLLM forward context.

## Engine interface

* `BoundaryRequestView` (frozen): `request_id`, `request_index`, `step_id`,
  `registration_epoch`, `server_generation`, `resource_id`, `dispatch_layer`,
  `execution_layer`, `prefix_end`, `step_tokens`, `ring_length`, `ring_page`,
  `next_slot`, `compression_ratio`, `compression_phase`,
  `position_control=(prefix_end, step_tokens, out_base, in_base, ring_page)`,
  `written_rows` (absolute rows inside the request's own live page).
  `__post_init__` enforces the schema2 page envelope: one page per request,
  `len(written_rows) == min(step_tokens, ring_length)`, every row inside
  `[ring_page*ring_length, (ring_page+1)*ring_length)`, a 5-word position
  control and a phase inside `[0, ratio)`. `boundary_end` is
  `prefix_end + step_tokens`.
* `BoundaryTensorView` (frozen): real `ring` tensor, `page_ids`, `ring_rows`,
  `ring_head_size`, plus `dtype_name`, `element_size`, `page_for(request)` and
  `planes()` -> `(kv, scores)` plane descriptors. The two planes share one
  payload (`row_offset_bytes` 0 / `head*element_size`); they are not separate
  boundary resources.
* `DSV41LwV2BoundaryCodec` protocol:
  `capture_layer_boundary(request_view, tensor_view, *, stream=None)` and
  `restore_layer_boundary(...)`. Implemented by T4 on the LMCache-Ascend side.
* `DSV41LwV2BoundaryHook`: engine implementation of the connector's
  `DSV41LwV2BoundaryHook` (`capture_group_boundary(group_id)` /
  `restore_group_boundary(group_id)`). It resolves the bound group, the live
  transport `StepView`, the per-group compressor-state ring and the real page
  set, then calls the codec once per request that owns a real
  `load_prefix_end > 0` (restore) / `store_prefix_end > 0` (capture). Empty
  sets are explicit no-ops; a missing codec is `BOUNDARY_NOT_BOUND`.

## Install point

`install_dsv41_lw_v2_codec(codec, *, model)` runs from
`NPUModelRunner.initialize_kv_cache`, after
`get_kv_transfer_group().register_kv_caches(...)` and
`install_boundary_provider(model=...)`, i.e. before the first real forward.

It validates, fail-closed:

1. both codec methods exist and accept `(request, tensor, stream=None)`;
2. the v2 engine seam is already bound (the connector is the hook receiver);
3. the model exposes exactly the three `*.compressor.state_cache` boundary
   resources of the bound plan and no others;
4. every live ring is a float32, contiguous 3-D full-page tensor
   (`[pages, rows, 2*head]`, even width, strided row-pair planes) and its
   geometry is recorded for later drift checks.

Re-installing the same codec instance is idempotent; a different instance
raises `DSV41BoundaryBindingError`. The installer then binds
`DSV41LwV2BoundaryHook` into the connector (`connector.bind_boundary_hook`).
`install_from_connector(model, connector=None)` is the model-runner entry
point: stock connectors return `False` untouched; a v2 connector without a
codec logs a warning and stays inert (the connector itself fails closed with
`BOUNDARY_NOT_BOUND` on the first real boundary hit).

## Step / ring binding

`bind_forward_step(runner, decoder_replay_dummy=..., synthetic=...)` is called
from `_model_forward` before the model runs and cleared in the `finally` block
(`clear_forward_step(token)` restores the previous binding). Until a codec is
installed it returns a sentinel and reads nothing.

* `synthetic` (engine init phase / profile / warmup), `decoder_replay_dummy`
  and graph capture produce a step with no requests, so no boundary operation
  can ever be emitted for a dummy forward;
* real steps read host-only scheduler facts: `input_batch.req_ids`,
  `num_computed_tokens_cpu`, `runner.query_lens` and
  `block_table.block_tables[0].get_cpu_tensor()`. Device tensors are rejected
  (`must stay on the host`), so the hot path never synchronizes;
* per request the engine records `prefix_end`, `step_tokens`, `out_base`,
  `in_base` and the request's own live ring page. The real step and ring facts
  are then taken in the hook from `attn.v41_impl.compressor_state_prefix`,
  `attn.compressor.state_cache.kv_cache[0]` and
  `forward_context.attn_metadata[resource_id]` (`num_reqs`, `compress_ratio`,
  `is_compressor_state`, `block_table`).

Restore always writes the request's current live page: a request captured on
page 3 and restored on page 7 gets page-7 rows and page 7 in
`position_control`, so a source page is never written back to a different
request.

## Failure behavior

* real boundary hit + no installed codec -> `DSV41BoundaryNotBoundError`
  (`error_code="BOUNDARY_NOT_BOUND"`), never a silent no-op (contract D3);
* missing/synthetic transport or forward step, non-compressor metadata, batch
  mismatch, unknown request, ring geometry drift -> `DSV41BoundaryStateError`;
* codec exceptions are wrapped in `DSV41BoundaryStateError` with the original
  exception as `__cause__`, so the step fails instead of continuing locally;
* misconfiguration (codec signature, resource set, engine seam) ->
  `DSV41BoundaryBindingError` at install time.

## Tests

`tests/ut/attention/test_dsv41_lw_v2_boundary.py` (stubs vLLM modules, loads
topology / group seam / boundary seam by path; no NPU needed):

```
/Users/hexiaoying/workspace/dsv41-layerwise/.venv/bin/python -m pytest --noconftest \
  -q tests/ut/attention/test_dsv41_lw_v2_boundary.py
# 43 passed
```

Covered: install validation (missing seam, missing capture, wrong signature,
missing resource, float16 / odd-width / non-contiguous ring), install
idempotency and codec conflict, stock connector untouched, v2 connector
install, codec-less install warning; real host step facts, meta-device
rejection, unset/duplicate request ids, synthetic (engine-init and decoder
replay) steps, step-id advance and `finally` restore; real DTO facts
(step/generation/resource/layer identity, prefix, cursor, phase, control,
page-aligned rows, two-plane tensor view), live-page relocation (capture page
3 -> restore page 7), per-request pages and indices, hit/store set filtering,
empty-set no-op, g20 without boundary, codec-less real hit ->
`BOUNDARY_NOT_BOUND`, missing transport/forward step, batch mismatch,
non-compressor metadata, geometry drift, unknown request, codec exception
wrapping; DTO schema2 rejections (off-page rows, short control, phase,
`next_slot`, row count, bool page id); static checks (no legacy layerwise
import, no `admit_step` / `.tolist(` / `.cpu(` / `.item(`, model_runner wires
bind/clear/install).

Together with the T3 tests:

```
pytest --noconftest -q tests/ut/attention/test_dsv41_lw_v2_boundary.py \
  tests/ut/attention/test_dsv41_lw_v2_hooks.py \
  tests/ut/models/test_deepseek_v41_transfer_resources.py \
  tests/ut/attention/test_dsv41_layerwise_store_hook.py \
  tests/ut/attention/test_dsv41_layerwise_init_phase.py
# 92 passed
```

Pre-existing environment limits (not caused by this slice): the
`tests/ut/conftest.py` imports vLLM, so the suite needs `--noconftest` here;
`test_dsv41_layerwise_c2*.py` already fail to import `dsv41_sampling` at clean
HEAD; `test_dsa_v41.py` needs a vLLM install.

## Not covered in this slice

* the transport codec (schema2 encode/decode, staging, store/load RPCs) is
  T4's LMCache-Ascend slice; this repo only defines the protocol and DTO facts;
* `DSV41LwV2Connector.current_step` / `boundary_codec` public accessors are T4
  work; the engine already reads them fail-closed (`None` transport step is a
  state error, missing codec is `BOUNDARY_NOT_BOUND`);
* TP-consistent abort of a failed boundary restore, group-load error
  propagation to stock `failed_recving` and hardware validation are T5/T6
  gates;
* a prefetch/`next_group` caller is not added here; the connector drives
  prefetch inside its own plan.
