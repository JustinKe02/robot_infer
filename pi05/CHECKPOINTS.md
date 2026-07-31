# PI0.5 Checkpoint Inventory

Checkpoint directories are named after the actual optimizer step recorded by
the source checkpoint directory, using six decimal digits:

```text
checkpoints/<actual-step>/pretrained_model
```

Do not derive the actual checkpoint step from `train_config.json:steps`. That
field is the configured training total. A checkpoint is complete only when its
source directory step equals that configured total.

`checkpoints/current` is a convenience symlink. Its target does not change a
checkpoint's identity or completeness.

## Inventory

| Local step | Epoch | Configured steps | Status | Current | Checkpoint fingerprint |
| --- | --- | --- | --- | --- | --- |
| `010600` | 10/15 | 15900 | intermediate | yes | `4698315f6936f9e9ef19017cfdb873588eba771fdb23595879ce2a7703b4c8dd` |

The fixed runtime profile for this row is documented at
`profiles/step_010600/README.md`.

### Step 010600

Training host:

```text
cqy@10.1.26.37 (ubuntu-Z790-EAGLE-AX)
```

Source directory:

```text
/data/cqy_workspace/jz_robot/flexible_lerobot/my_devs/jz_robot_pin_timed/pi05/outputs/pi05_jz_robot_pin_timed_merged_100eps_20260728_pi05_head_right_full_e15_b32_pi05_100eps_b32_full_then_expert_20260728_220410_full/checkpoints/010600/pretrained_model
```

The local and source directories were compared byte-for-byte through SHA-256.
All seven files matched:

| File | SHA-256 |
| --- | --- |
| `config.json` | `d78c68f4c43d27a436f30c8adf0e8f9ff4f71696ff81edfc96578186670170ff` |
| `model.safetensors` | `34fb5d0752fc04cadef7e71365e9922280b11b654f2e41381d9add320e4def24` |
| `policy_postprocessor.json` | `21e036d535326e9cac81ca5ae14504d5bf1e64b0766180f5d35be976507d2ad4` |
| `policy_postprocessor_step_0_unnormalizer_processor.safetensors` | `5f676bd3c2ab9fa2758fdec576f57ea6f964b183a6901c9ff49904dd89fe00ca` |
| `policy_preprocessor.json` | `a83195deef840e0dcc7d61423966d218b77a57d4644c93975c64d581f0097dcb` |
| `policy_preprocessor_step_3_normalizer_processor.safetensors` | `5f676bd3c2ab9fa2758fdec576f57ea6f964b183a6901c9ff49904dd89fe00ca` |
| `train_config.json` | `b8656a54e7beff2c4c3e0a4569dea012f8c9bd88ecf2b871fc698c9f92fd065a` |

The model size is:

```text
model.safetensors bytes: 7473096344
```

This checkpoint is `10600/15900`, so server startup requires the explicit
`REQUIRE_COMPLETE_STEP=false` override. Clients must provide the full audited
checkpoint contract documented in `README.md`; incomplete checkpoints are not
accepted implicitly.

## Adding A Checkpoint

1. Preserve the numeric step from the source `checkpoints/<step>` directory.
2. Copy the complete `pretrained_model` directory, including both processor
   JSON files and their processor state files.
3. Compare SHA-256 manifests between the source and local directories.
4. Record the actual step, configured total, source path, status, checkpoint
   fingerprint, model size, and full model SHA-256 in this inventory.
5. Update `checkpoints/current` only after the new entry has been verified.
6. Never rename an intermediate checkpoint to the configured final step.
