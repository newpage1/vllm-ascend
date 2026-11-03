# DSV4.1 Cache Resources for Layerwise Transfer Planning

The `deepseek_v41.topology` module contains the existing source/consumer graph
and a device-independent cache-resource inventory. `model.py` re-exports the
existing `build_layer_plan`, `DeepseekV41Topology` and `DeepseekV41LayerRole`
names so current callers keep their import paths.

`topology.cache_resources(prefix=...)` reports per-layer SWA, source long KV,
source index K/scale, conditional folded index K and C2 compressor state.
The inventory describes ownership and compute lifetimes, not physical tensor
layout or asynchronous transfer completion.

For a configuration with KV source 20 and index sources 20, 24, 28, 32 and 36,
long KV from layer 20 stays live through layer 39. Its index K is needed at the
index-source layers; layer 26 uses the top-k generated at 24 and reads the long
KV from 20. Changing top-k source does not release the long KV.

Top-k and candidate query results are not persistent cache resources. SWA and
compressor state remain explicit even when an external cache plans to restore
them through replay or aligned snapshots. Optional folded resources must be
matched against the actual hardware cache registration.

The topology rejects malformed source ordering, missing owners and incompatible
compression ratios before cache registration. This prototype does not add
layerwise hooks, alter buffer allocation or change graph execution. Device
consumer events and any asynchronous copy must both finish before storage can
be overwritten.
