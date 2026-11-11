# GLM-5.3-Flash PCP + layerwise pooling: selected evidence

These are **allowlisted excerpts of historical records**, not a new test run or a full raw-log bundle. Each JSON identifies its source by a relative filename and the original file's SHA-256. Internal hosts, accounts, absolute paths, device pointers, process identities and launch-environment values are omitted. Actual wire keys, pool GVAs, effective copy ranges and SDK result codes are retained because they establish the collision.

The hardware scope is an A3 two-chip run using a fixed four-layer BF16 random checkpoint with the original GLM-5.3-Flash cache dimensions, native KDA, PCP2, TP1/PP1/DCP1, colocated prefill/decode, eager execution and synchronous scheduling. This checks state-transfer correctness and numerical consistency, not original-model task accuracy or full-model performance.

## 1. Missing PCP identity causes a real destination collision

[key-diagnostic.json](key-diagnostic.json) contains the **key1 metadata-only diagnostic window**. The baseline workers report actual PCP size 2 and PCP ranks 0/1, but the bound key builder reports PCP size 1. Both workers construct the same full key for each of the three KDA groups.

For group 2, both ranks allocate this exact key:

```text
glm53-four-layer-bf16@2@555d5d9cd124f64500a0e016229ef2d1f014295f0e85c3b17a2b5f05b86b81d2@0
```

Both allocations return base GVA `72057594042941440`, with allocation size `4489216` bytes. Both ranks copy to the same effective ranges: conv at offset 0, size `73728`; recurrent state at offset `73728`, size `1048576`. Copy calls return 0 and publication via `batch_write_finish` returns 0 for the same key. Groups 3 and 4 reproduce the same collision pattern.

Allocation-to-copy associations come from the original diagnostic's request/group/range mapping, retained here as mapping IDs; they were not inferred by choosing the nearest allocation in time. This window captured CPU metadata only, with no payload reads or device synchronization. Its `diagnostic_status=passed` means the observation succeeded, **not that model validation passed**. The baseline full validation remained `numerical_failed`.

The candidate adds `@pcp_size:2@pcp_rank:0` / `@pcp_size:2@pcp_rank:1`. The recorded allocations and effective KDA copy ranges are separate across ranks; copy and publication still succeed.

## 2. Different payloads and incorrect reload are separate evidence

[payload-overwrite-b1.json](payload-overwrite-b1.json) is the earlier **b1 payload-capturing window**, not the key1 window above. It records six KDA effective ranges where two PCP ranks write different bytes to the same destination, followed by six own-rank GET-versus-PUT mismatches. One conv range contains `66213` differing bytes out of `73728`.

The original report contains ten overlapping descriptors in total. Only six KDA descriptors have different payloads; the remaining four replicated MLA/Indexer descriptors have equal payloads. The fixed-32-step cold/reload logits are finite but fail `atol=0.02, rtol=0.02`, with maximum absolute error `8.203125` and top-1 agreement 0. This run did not capture raw key strings, so it must not be presented as a direct key-string trace.

[key-fix-validation.json](key-fix-validation.json) contains the subsequent fixed-key numerical and transfer results at chunk budget 4608. After removing the diagnostic hooks, cold/reload and cold/same-PCP-no-pool comparisons have shape `[32,154880]`, maximum absolute error 0, and top-1 agreement 1. Copy-integrity and ownership checks pass with zero cross-rank write conflicts.

## 3. Validation exposed an independent checkpoint-alignment bug

[checkpoint-alignment.json](checkpoint-alignment.json) keeps the original key-fixed **s1** failures and the later **a1/a2** results separate. A chunk budget is the scheduler's batched-token budget, not the prompt length or the maximum externally cacheable prefix.

For a 2305-token prompt with actual common cache granularity 1152, the expected complete cached prefix is `floor((2305-1)/1152)*1152 = 2304` tokens. At budget 2048, the original consumer control flow reaches `0 -> 1152 -> 2305`, skipping the 2304 KDA checkpoint. The recorded first sampling context is `computed=1152, end=2305`; only one KDA PUT per layer is present. Reload hits only 1152 tokens with local APC 0. Successful copies and cold/reload logits error 0 do not satisfy that missing-hit gate.

The independent scheduler fix preserves ordinary checkpoint alignment and only bypasses it for verifier windows beyond already committed tokens. After the alignment fix:

| Historical run | Chunk budget | Input lengths | External reload tokens | Local APC | Fixed-32-step error vs same-PCP/same-budget no-pool |
|---|---:|---|---|---:|---:|
| a1 | 2048 | 129 / 2305 / 9217 | 0 / 2304 / 9216 | 0 | 0 |
| a2 | 1024 | 129 / 2305 / 9217 | 0 / 2304 / 9216 | 0 | 0 |

Cold/reload natural outputs agree. Valid PUT/GET bytes match their own rank/group/layer source and cross-rank effective PUT overlap is zero. The first sampling context becomes `computed=2304, end=2305`; the second KDA state is saved and restored. The checkpoint interpretation combines source control flow, hit extent, sampling context and state-transfer records. It is **not** a direct token-boundary-to-key commit trace. The historical summary's unrelated no-pool reference is excluded from same-budget conclusions.

[shared-prefix-validation.json](shared-prefix-validation.json) adds the final shared-prefix case at budget 2048: A has 2305 tokens; B has 3457 tokens with exactly 2304 shared tokens. A is submitted first in a new pool, then local APC is reset before B. B has APC 0 and external hit 2304. Four B execution paths × two ranks × 32 raw-logit files were independently compared: all eight repeat/history/pool comparisons have actual maximum error 0 and finite values. Each rank completes four successful GETs totalling `5873664` bytes. This case records SDK completions without transfer payloads, so it is not an additional payload-ownership audit.

## CPU/static checks after rebase

[cpu-validation.json](cpu-validation.json) summarizes the checks reported for the proposed changes rebased onto main `1af7c5fc8`. The new PCP regression tests alone fail on unfixed main: 3 failures in 0.77 s. With the key fix, the focused CPU suite passes 208 tests in 6.25 s. The independent alignment suite passes 67 tests in 0.26 s. Ruff passes for the eight changed files.

Both `format.sh ci` invocations stop with exit code 3 while fetching `ruff-pre-commit` because certificate verification fails. They are recorded as blocked, not successful full CI runs. Exact source and raw-log hashes are pending addition by the PR author. These are CPU/static results for the rebase, **not a new NPU validation**.

## Recorded model provenance

[model-provenance.json](model-provenance.json) preserves the locally recorded seed, config SHA and fixture-manifest SHA. The fixed BF16 fixture uses seed `20261109`; its config SHA-256 is `a5eae530cc96239973d356e896dfa18db939a03e775cb057695d47ff30ec900d`. Three existing run-preparation records report fixture-manifest SHA-256 `0d73da7fba46ed38fd42f027f1997702637241051acc6c0afeb9df27bb7dca92`.

This is a recorded manifest digest, not an individual weight-shard digest or a newly computed aggregate weights hash. The full fixture manifest and shard hashes were not present in the inspected local material; no large weight files were read or rehashed to prepare this bundle.

## Revision and coverage boundaries

- Historical baseline: `6e3299ab492ddbadb4c8e3a00b88c982c9a28131`.
- Historical key fix: `bc525b47198f1807304a6b95fdc8a28bdbf57a4a`.
- Historical alignment fix: `113f8acae1b2240745580217f012672134f1c2ec`; the tested alignment overlay changed one Python file while preserving the candidate native files.
- Later KeyBuilder refactors and a rebase onto newer main must be described with their own CPU/static checks. These historical A3 records do **not** certify that a rebased PR head was executed on NPU.
- Hardware evidence here is for Memcache layerwise PCP2. Mooncake/PP/key-format CPU coverage must not be described as equivalent hardware coverage.
- The private K-pool tail follows reconstruction; it has no independent tail-byte oracle here. Graph, MTP, concurrent correctness, full-scale task accuracy and performance are outside these results.
- Raw tensors, complete logs and host-specific launch commands are intentionally not included. Preparing these excerpts did not restart or rerun any validation.

## Rebased CPU log excerpts

[CPU/static summary](cpu-validation.json) includes the tested source hashes and original log hashes. The [negative-control log](cpu-logs/pcp-red.txt), [PCP positive log](cpu-logs/pcp-green.txt), [alignment positive log](cpu-logs/align-green.txt), and [PCP](cpu-logs/pcp-format.txt) / [alignment](cpu-logs/align-format.txt) full-format failures retain assertions and results with checkout/artifact paths replaced. The test environment reported vLLM `0.33.0+empty`; these are CPU/mock suites, not production-runtime or NPU qualification.
