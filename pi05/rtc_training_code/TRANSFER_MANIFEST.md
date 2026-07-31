# PI0.5 Training-Time RTC Transfer Manifest

## Purpose

This directory is a repository overlay for the source that produced and loads
the JZ Robot PI0.5 three-camera, model16, expert-only, training-time RTC
checkpoint at step 010600. It intentionally does not contain the checkpoint
weights or dataset.

## Included

- Complete `src/lerobot/policies/pi05/`.
- Complete `src/lerobot/policies/rtc/` (the training-time implementation is in
  PI0.5, but the existing inference-time RTC imports remain part of the load
  surface).
- Complete JZ raw18/model16 schema and processor boundary.
- Processor pipeline, PI0.5 policy registration/config parsing, policy factory,
  pretrained loader, and training entry point files needed by the overlay.
- JZ PI0.5 training hook, three-camera training scripts, checkpoint loader,
  server/client source, inference scripts, and runtime tests. Generated output,
  logs, runtime caches, backups, and benchmarks are excluded.
- PI0/PI0.5 and RTC policy tests plus the JZ training-schema test.
- Source commit, full `git status --short`, focused tracked diff, relevant
  untracked file list, commands, inference contract, test report, complete file
  list, and SHA256 checksums.

The keyword search was scoped to the PI0.5/RTC/JZ training and inference
surface. Unrelated `pi0_fast`, bridge-capture RTSP, and visualization files
that happen to contain generic names such as `prefix_mask` or `max_delay` are
not part of this checkpoint implementation and are not included.

## Tokenizer

Tokenizer source on the training server:

```text
/mnt/data2/ybd/vla_act/cqy/assets/modelscope/google/paligemma-3b-pt-224
```

It is not copied because the checkpoint references the standard local
PaliGemma tokenizer and no target mismatch was reported.

```text
ef6773c135b77b834de1d13c75a4c98ab7a3684ffd602d1831e1f1bf5467c563  tokenizer.json
3259402b1d1802e02417d7bff75a889ec61d359d15be6050a957b307c48edbbe  tokenizer_config.json
```

## Excluded

- `model.safetensors`, adapter weights, and all other model weights.
- Checkpoint directories, datasets, videos, training outputs, logs,
  benchmarks, caches, and backup archives.
- Tokenizer files, conda environment, API tokens, SSH keys, and credential
  files.
- Unrelated repository changes and projects.

## Known missing work

- The old HTTP `rtc` mode implements inference-time VJP RTC. It has not yet
  been extended with training-time `action_prefix`/`prefix_length` fields.
- No robot-side training-time RTC deployment or robot motion was run.
- The repository has extensive unrelated modified/untracked files; therefore
  `GIT_STATUS.txt` is large. `UNCOMMITTED_CHANGES.diff` is intentionally scoped
  to this overlay.
- `COMMITTED_CHANGES.diff` is empty because the current HEAD commit has no
  changes under the selected JZ PI0.5/RTC paths; the relevant implementation
  is uncommitted and captured by the overlay plus focused diff/untracked list.
