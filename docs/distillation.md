# Knowledge Distillation

The training script supports knowledge distillation, where a "student" network learns from both the ground truth data and the soft targets provided by a pre-trained "teacher" network.

## Configuration

To enable distillation, you need to configure the `teacher` settings within the `training` configuration in your root configuration file. This includes the path to the teacher's weights, the distillation parameters, and the teacher's model structure.

### Teacher Settings (`training.teacher`)

This section, nested within `training`, configures the distillation process and the teacher's architecture.

```protobuf
training {
  # ... other training settings
  
  teacher {
    checkpoint_path: "/path/to/teacher/checkpoint"
    kd_alpha: 0.5       # Weight of the distillation loss (0.0 to 1.0)
    temperature: 2.0    # Softmax temperature for soft targets
    
    # Teacher model structure (same ModelConfig as the student).
    model {
      embedding { dense_size: 256 embedding_size: 256 dff: 512 }
      encoder { num_blocks: 10 dff: 512 d_model: 256 heads: 8 }
      policy_head { name: "vanilla" embedding_size: 256 d_model: 256 }
      value_head { name: "winner" num_channels: 64 }
      # Also supply defaults and any other settings from the teacher config.
    }
  }
}
```

- `checkpoint_path`: Path to the directory containing the teacher's checkpoint (Orbax format).
- `kd_alpha`: controls the balance between the standard loss and the distillation loss. KD is added to the configured ground-truth losses; setting this to `1.0` does not disable those losses. Typical values range from `0.1` to `0.5`.
- `temperature`: controls the "softness" of the teacher's probability distribution. Higher values produce softer targets, revealing more about the relationships between classes. Typical values are between `1.0` and `4.0`, with `2.0` being a common starting point.
- `model`: Defines the architecture of the teacher model. It uses the same `ModelConfig` structure as the main `model` config. This allows the teacher to have a different architecture than the student.

## Standard ONNX files

Set `training.teacher.onnx_path` to a standard `.onnx` model file. The file
is passed directly to ONNX Runtime; LC0 network protobuf files containing an
embedded ONNX model are not accepted. Omit `checkpoint_path` and `model` when
using `onnx_path`.

Map each desired ONNX output to a student head explicitly:

```protobuf
training {
  teacher {
    onnx_path: "/path/to/teacher.onnx"
    temperature: 2.0
    policy {
      head_name: "vanilla"       # Student policy head.
      output_name: "output/policy"  # Exact ONNX output name.
      kd_alpha: 0.5
    }
    # Optional WDL distillation.
    value {
      head_name: "winner"        # Student value head.
      output_name: "output/wdl"  # Exact ONNX output name.
      output_is_probabilities: true
      kd_alpha: 0.25
    }
  }
}
```

The output names above are examples; use the names from your ONNX model.
At least one mapping is required. `head_name` must identify an existing student
head, and each head can appear only once within each family. Multiple student
heads may use the same ONNX output. Unmapped ONNX outputs are ignored.

Set `kd_alpha` in every mapping: as with checkpoint teacher head overrides, a
mapping's value overrides the global weight, and zero disables KD for that head.
A positive per-head `temperature` overrides the global temperature; otherwise
the global value is used (or `1.0` if it is non-positive).

Outputs are treated as logits by default. Set `output_is_probabilities: true`
for outputs that already have softmax applied, such as exported WDL
probabilities. These are converted to log probabilities before applying the KD
temperature. Policy outputs must have shape `[N, 1858]` in LC0 move order; value
outputs must have shape `[N, 3]` in win/draw/loss order. Scalar value outputs
are not supported.

The model must accept one float32 or float16 input of shape `[N, 112, 8, 8]`
using the same LC0 input format as the training data. Input plane 109 (rule-50)
is multiplied by 99 to restore raw counts, matching ONNX evaluation in this
repository. Dynamic batch sizes are supported. For a fixed batch size,
inference splits the training batch and zero-pads the final partial chunk.

The teacher remains frozen. Inference uses a JAX callback that batches samples
under `vmap` and works inside the compiled loss used by training, daemon, and
overfit. ONNX Runtime prefers CUDA when available, with CPU fallback. The
callback transfers inputs and outputs through host memory, so throughput can
differ from a native JAX teacher.

## Tuning Distillation

When tuning `kd_alpha`, it is important to monitor the magnitude of the distillation loss relative to the other losses (policy, value, etc.). 

**Crucially, the weighted distillation loss (`unweighted_loss * kd_alpha`) should be in the same order of magnitude as the other weighted losses.** 

If the distillation loss dominates the total loss (e.g., accounting for 90-95% of the total loss), the student will focus too heavily on mimicking the teacher's policy at the expense of learning from the ground truth rewards (values) and other signals. A well-balanced setup typically sees the distillation loss contributing a significant but not overwhelming portion of the total gradient.

## How it Works

When configured:
1. The teacher is loaded from a checkpoint using `training.teacher.model`, or directly from `onnx_path`.
2. During each training step, the same input batch is fed to both the student (being trained) and the teacher (frozen).
3. Soft-target cross entropy is computed for matching policy and WDL value heads, with logits scaled by `temperature`. Policy KD masks illegal moves using the training targets. The loss is multiplied by the temperature squared.
4. This distillation loss is weighted by `kd_alpha` and added to the total loss.

**Note:** Distillation supports policy and WDL value heads. Moves-left and auxiliary value outputs are not distilled.
