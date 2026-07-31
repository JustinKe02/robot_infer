# RTC-conditioned 010600 robot client profile

This profile connects the audited JZ robot client to the independent
`torch_rtc_conditioned` server at `127.0.0.1:18089`. It locks the transferred
checkpoint fingerprint, the path-unproven `null` step contract, the
three-camera observation profile, the exact training task, Orin
`192.168.1.81`, state `0.0.0.0:39010`, command port `39020`, and the 250 ms
camera/state receive-skew gate.

The locked training task is:

```text
Put the bottle on the right into the basket on the left.
```

Every command requires:

```bash
TK_PI05_RTC_CONDITIONED_010600_CONFIRMED=1
```

Run stages in order:

```text
run_inference_smoke.sh      one live observation and inference; no send_action
run_single_step_dry_run.sh  local transport; no command UDP
run_single_step_armed.sh    UDP, 5 Hz, one second, at most one action
run_rtc_dry_run.sh          local transport at 20 Hz
run_rtc_armed.sh            UDP, one second, at most ten actions
```

Armed entrypoints retain all three global robot confirmations. RTC armed also
requires `JZ_PI05_RTC_CONDITIONED_SINGLE_STEP_ARMED_PASSED=1`, which may be set
only after the bounded single-step armed result has been reviewed.

An explicitly authorized single-step-only bypass is available after a normal
dry-run is blocked by the 0.02 rad checks. It requires
`JZ_PI05_RTC_CONDITIONED_JOINT_DELTA_BYPASS_CONFIRMED=1`. Single-step bypass
launchers force one second and at most one action. RTC bypass launchers force
20 Hz, one second, and at most ten actions; RTC armed also requires the prior
single-step pass confirmation. Shape, finiteness, raw18 force, state
freshness, camera/state skew, gripper, checkpoint, queue fail-closed, and
armed-confirmation checks remain active.

Continuous RTC bypass is a separate launcher. It first requires the fixed
five-second local qualification to complete with multiple inference requests.
After reviewing that summary, continuous armed additionally requires
`JZ_PI05_RTC_CONDITIONED_CONTINUOUS_DRY_RUN_PASSED=1` and
`JZ_PI05_RTC_CONDITIONED_CONTINUOUS_ARMED_CONFIRMED=1`. Continuous means
`run_time_s=0` and `max_sent_actions=0`; it runs until Ctrl+C or a fail-closed
condition. Bounded time/action overrides are rejected on that launcher.
