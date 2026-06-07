# Fock-PARFLM v2.1 Conservativity Diagnostic — Results Summary

Generated: 2026-06-07 04:45:39

## Arm 1: Structural Jacobian Symmetry


## Arm 2: Conservative Ablation

- Reverse channel scale: tanh(s) = -0.2269802987575531
- Mean R^2 (Q_i off): 0.737445
- Mean R^2 (Q_i on):  0.615444

## Arm 3: Energy Budget Decomposition

- Samples: 40
- Mean |residual|: 2.3106e+03
- Mean |Delta_H|:  2.5266e+03
- Residual ratio:  9.1449e-01

## Arm 4: Conservativity Dial

- Learned tanh(scale): -0.226980
| tanh(s) | PPL | R^2 mean |
|---------|-----|----------|
| -0.2270 | 12.17 | 0.9740 |
| -0.2000 | 11.73 | 0.9987 |
| -0.1500 | 11.46 | 0.9984 |
| -0.1000 | 11.91 | 0.9985 |
| -0.0500 | 13.29 | 0.9977 |
| 0.0000 | 15.95 | 0.9966 |

## Arm 5: Four-Way Separator

| Model | R^2 mean | Source |
|-------|----------|--------|
| GPT-2 (paper) | 0.4600 | literature |
| SPLM (paper) | 0.9570 | literature |
| FockPARFLM (Q_i=0) | 0.7374 | measured |
| FockPARFLM v2.1 | 0.6154 | measured |
