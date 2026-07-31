# PI0.5 Training-Time RTC Inference Contract

## Checkpoint contract

- Policy type: `pi05`.
- Checkpoint config: `rtc_training.enabled=true`, `max_delay=10`,
  `min_postfix_steps=1`.
- Input cameras: head `(3,720,1280)`, left `(3,480,640)`, and right
  `(3,480,640)`.
- Serialized processor boundary: raw state/action 18 dimensions to model state
  and action 16 dimensions, then model action 16 back to wire action 18.
- Action horizon `H=50`; internal padded action dimension `Dpad=32`.

## Training tensors

- `actions`, `noise`, and `x_t`: `[B,H,Dpad]`, normally `[B,50,32]` after
  padding model16 actions.
- Sampled `delay`: `[B]`, integer action steps. It is the committed prefix
  length for each batch item.
- `prefix_mask`: `[B,H]` boolean. `True` means a committed clean action token.
- `token_flow_time`: `[B,H]` floating point. Clean-prefix tokens use `time=0`;
  postfix tokens use the sample's global flow time. In this parameterization,
  `time=0` is clean data and `time=1` is noise.
- At least `min_postfix_steps` remain trainable. The flow-matching loss excludes
  prefix tokens and normalizes each sample by its own postfix element count.

## Offline inference arguments

`PI05Policy.predict_action_chunk` forwards these arguments to
`PI05Pytorch.sample_actions`:

- `action_prefix`: `[P,D]` only for batch size 1, or `[B,P,D]`. `D` may be 16
  for this checkpoint and must not exceed the padded dimension 32. Values must
  be in the model's normalized action space, before the checkpoint
  postprocessor converts model16 back to raw18.
- `prefix_length`: scalar Python integer in
  `[0,min(P,H)]`. If omitted, all `P` supplied steps are used.
- Internal `prefix_values`: `[B,50,32]` after zero padding.
- Internal inference `prefix_mask`: `[B,50]` boolean.
- Internal `token_flow_time`: `[B,50]`; prefix positions are always `0`, and
  postfix positions use the current Euler time.

At every Euler iteration the implementation clamps prefix values before the
denoiser, zeros the predicted prefix velocity, applies the Euler update, and
clamps the prefix again. The returned prefix is therefore exact, not merely a
soft condition.

`prefix_length=0` is supported. Calling without `action_prefix`, or with an
empty `[B,0,16]` prefix and `prefix_length=0`, runs ordinary full-chunk
single-step inference for this RTC-trained checkpoint. With fixed inputs and
noise, those two calls are expected to be identical.

## Delay mapping

Runtime delay is measured in action steps and maps to the number of already
committed clean actions: `prefix_length = clamp(delay, 0, H-1)`. The caller is
responsible for selecting the corresponding normalized model16 action prefix.
The model does not accept a separate `delay` argument on the training-time RTC
path.

## Incompatibility with inference-time VJP RTC

`rtc_training.enabled=true` and `rtc_config.enabled=true` are intentionally
mutually exclusive. The existing `rtc_infer` HTTP server's `rtc` request mode
installs inference-time VJP guidance and does not pass `action_prefix`; it is
not the runtime for this B checkpoint. Its checkpoint loader and single-step
mode remain useful. A live training-time RTC client/server still needs to carry
`action_prefix` and `prefix_length` through its wire protocol before robot use.
