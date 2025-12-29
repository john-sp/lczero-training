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
    
    # Teacher model structure (same as main model config)
    model {
      embeddings_size: 256
      encoder {
        layers: 10
        dim: 256
        heads: 8
        # ... other encoder settings
      }
      # ... other model settings
    }
  }
}
```

- `checkpoint_path`: Path to the directory containing the teacher's checkpoint (Orbax format).
- `kd_alpha`: controls the balance between the standard loss and the distillation loss. A value of `1.0` would mean only using the distillation loss (scaled). Typical values range from `0.1` to `0.5`.
- `temperature`: controls the "softness" of the teacher's probability distribution. Higher values produce softer targets, revealing more about the relationships between classes. Typical values are between `1.0` and `4.0`, with `2.0` being a common starting point.
- `model`: Defines the architecture of the teacher model. It uses the same `ModelConfig` structure as the main `model` config. This allows the teacher to have a different architecture than the student.

## Tuning Distillation

When tuning `kd_alpha`, it is important to monitor the magnitude of the distillation loss relative to the other losses (policy, value, etc.). 

**Crucially, the weighted distillation loss (`unweighted_loss * kd_alpha`) should be in the same order of magnitude as the other weighted losses.** 

If the distillation loss dominates the total loss (e.g., accounting for 90-95% of the total loss), the student will focus too heavily on mimicking the teacher's policy at the expense of learning from the ground truth rewards (values) and other signals. A well-balanced setup typically sees the distillation loss contributing a significant but not overwhelming portion of the total gradient.

## How it Works

When configured:
1. The teacher model is loaded from `checkpoint_path` using the structure defined in `training.teacher.model`.
2. During each training step, the same input batch is fed to both the student (being trained) and the teacher (frozen).
3. The Kullback-Leibler (KL) divergence is computed between the student's policy logits and the teacher's policy logits, both scaled by `temperature`.
4. This distillation loss is weighted by `kd_alpha` and added to the total loss.

**Note:** Currently, distillation is only implemented for the policy head.
