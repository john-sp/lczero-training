# AGENTS.md

This repository contains training script for the Leela Chess Zero project.
They are being rewritten.

* Old code is located in the `tf/` directory.
* New python code is located in the `src/` directory.
* New C++ code is located in the `csrc/` directory.

The old code is Python/TensorFlow-based, new code is Python/JAX-based with
modules written in C++, operating through pybind11.

The build system for C++ code is meson. During development, the project is built
in the `builddir/`.

## Testing and Building

* C++ tests use GTest framework
  * Do not insert Sleeps in tests, it slows down presubmit. Instead use e.g.
    absl::Notification, or std::future
* Tests are defined in `meson.build` with `test()` function
  * When debugging, don't forget to build them before running `meson test` or
    `builddir/test`
* Run tests: `meson test -C builddir/`
* Python tests use `pytest` framework
  * Do not add custom main function, exception catching to report errors, any
    "test passed" messages etc. Use `pytest` fixtures and assertions.
* Build: `meson compile -C builddir/` from build directory
* Format code: `just format`
* There is a commit hook that runs `just pre-commit`, which runs tests and
  checks formatting. You may want to run it before attempting to commit.
* We use Google C++ style guide.
  * That means 80 columns.
  * That means comments should be in full sentences with periods in the end.
  * When conditional or loop fits one line, it must be written as one line
    without braces, for example:
      `if (condition) return value;`
  * Prefer `absl` to `std` (e.g. `absl::c_` algorithms, `absl::Mutex`,
    `absl::StrCat`, etc.)
* We use `uv` for Python package and venv management, and to running the
  application.
* Run TUI app: `uv run tui --config=<path_to_config>`

* Do not attempt to run TUI — it messes up the Agent interface and session has
  to be killed. Ask me to check it for you manually instead.

* Do not commit unless explicitly asked.

## IMPORTANT

* NEVER add `# type: ignore` or other ways to mask/silence errors instead of
  fixing them.
* Rely on protobuf default values. DO NOT write code like
  `config.has_foo() ? config.foo() : default_value;`

## Notes and gotchas

* Training configs live in `proto/` and are set via `.textproto` at runtime;
  see `docs/index.md` and `docs/example.textproto` for usage.
* Multi-GPU data-parallel training requires batch size divisible by
  `jax.device_count()`. The training loop throws a hard error otherwise; see
  `docs/architecture.md`.
* `overfit` is single-GPU only and errors if multiple devices are visible; use
  `CUDA_VISIBLE_DEVICES` to force one GPU.
* DataLoader output must be exactly three tensors; `TrainingBatch.from_tuple`
  rejects any other arity. This is a compatibility contract with the C++ loader;
  see `docs/loader.md` and `docs/training_tuple.md`.
* `StageConfig` must set exactly one stage-specific field; anything else is a
  runtime error. See `docs/new_stage.md` for stage wiring rules.
* `ShufflingChunkPool` requires at least one initial chunk and enforces input
  arity: one input normally, two only when `cachehit_output` is configured. It
  also enforces unique output names; see `docs/loader.md`.
* Queue lifecycle is producer-driven: queues close when all `Producer` tokens
  are destroyed. `Put` fails immediately when closed; `Get` fails only after
  the queue is empty; see `docs/loader.md`.
* `StreamShuffler` bounds are monotonic (only increase); do not attempt to
  rewind ranges; see `docs/loader.md`.
* Encoder attention config invariants are enforced at runtime:
  `d_model % heads == 0` and `heads % kv_heads == 0`; see
  `docs/architecture.md`.
* Policy head embedding config is exclusive: use a shared embedding or set
  `embedding_size`, but not both; see `docs/heads.md`.
* Moves-left head outputs are ReLU-clipped to non-negative values, which
  affects interpretation of predictions and loss scaling; see `docs/heads.md`.
* Loss `metric_name` values must be unique per loss type, or the loss
  constructor raises; see `docs/index.md` for config patterns.
* Distillation only applies to policy heads, with optional per-head override of
  `kd_alpha` and `temperature`; see `docs/distillation.md`.
* Weight masking is “deny wins”: any matching selector set to false excludes a
  parameter from decay/regularization; see `docs/optimizer.md`.
* LR schedule gaps use the previous LR, and steps before any schedule use the
  first LR entry. Sparse schedules can surprise; see `docs/index.md`.
* Weight conversion has quirks: plane 109 is scaled by 99.0, and net.proto
  default activation mapping only supports MISH/RELU; see
  `docs/weights_tool.md`.
* Legacy weight protobufs are patched to add missing format fields and attention
  body migrations before JAX conversion; see `docs/weights_tool.md`.

## Documentation

* Documentation is in the `docs/` directory.
* The contents is in [The index](docs/index.md)
