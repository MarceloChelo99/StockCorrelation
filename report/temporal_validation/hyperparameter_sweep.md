# Temporal Hyperparameter Sweep

Vanilla baseline NMI: `0.4333`
No candidate passed all validation gates; best row is ranked by diagnostic score.

Best candidate:
- `lambda_temp = 0.5`
- `alpha = 2`
- Overall pass: `False`
- AI ratio: `1.565`
- Stability improvement ratio: `0.159`
- Event pass rate: `0.00%`
- NMI ratio to vanilla: `1.144`
- Run dir: `experiments/_runtime/temporal_sweep/20260425_172812/lambda_0p5_alpha_2`

Top candidates:

| lambda_temp | alpha | overall_pass | selection_score | ai_ratio | stability_improvement_ratio | event_pass_rate | nmi_ratio_to_vanilla | clustering_nmi | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 2 | False | 2.213 | 1.565 | 0.1589 | 0 | 1.144 | 0.4958 | experiments/_runtime/temporal_sweep/20260425_172812/lambda_0p5_alpha_2 |
| 1 | 0.5 | False | 2.212 | 1.486 | 0.1821 | 0 | 1.158 | 0.5019 | experiments/_runtime/temporal_sweep/20260425_172812/lambda_1_alpha_0p5 |
| 0.25 | 2 | False | 2.211 | 1.653 | 0.1086 | 0 | 1.149 | 0.4977 | experiments/_runtime/temporal_sweep/20260425_172812/lambda_0p25_alpha_2 |
| 1 | 2 | False | 2.206 | 1.504 | 0.1639 | 0 | 1.161 | 0.5033 | experiments/_runtime/temporal_sweep/20260425_172812/lambda_1_alpha_2 |
| 0.5 | 0.5 | False | 2.178 | 1.392 | 0.2019 | 0 | 1.152 | 0.4991 | experiments/_runtime/temporal_sweep/20260425_172812/lambda_0p5_alpha_0p5 |
