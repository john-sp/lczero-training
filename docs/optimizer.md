# Optimizer Settings

This document describes the available optimizer configurations and suggested settings.

## Nadamw

Nadamw (Nesterov Adam with Weight Decay) is a standard choice.

Suggested settings:

```text
optimizer {
  nadamw {
    beta_1: 0.9
    beta_2: 0.999
    epsilon: 1e-7
    weight_decay: 0.0001
    decay_embedding: false
    decay_biases: false
    decay_layer_norms: false
  }
}
```

## Muon

Muon is a momentum-orthogonalized optimizer.

Suggested settings:

```text
optimizer {
  muon {
    ns_steps: 5
    beta: 0.95
    epsilon: 1e-8
    weight_decay: 0.01
    nesterov: true
    adaptive: false
    adam_beta_1: 0.9
    adam_beta_2: 0.999
    adam_epsilon_root: 1e-8
    adam_weight_decay: 0.0001
    decay_embedding: false
    decay_biases: false
    decay_layer_norms: false
  }
}
```
