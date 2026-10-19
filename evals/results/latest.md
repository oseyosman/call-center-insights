# Evaluation Report: BASELINE (HELDOUT split)

> **SIMULATED DATA NOTICE**: All metrics, transcripts, and audio files are synthetic and simulated for QA testing.

- **Evaluated Date**: 2026-10-19 18:08:48 UTC
- **Total Calls**: 20
- **Provider**: `baseline`
- **Bootstrap**: 1000 resamples, seed=42

## Summary Metrics
| Metric | Score | 95% CI | Ordinal / Extra |
| ------ | ----- | ------ | --------------- |
| Sentiment Start Accuracy | 15.0% | [5.2%, 36.0%] | Macro-F1: 0.1167 [0.0000, 0.2222] |
| Sentiment Start Within-One | 90.0% | — | OrdMAE: 1.0, QWK: 0.2553 |
| Sentiment End Accuracy | 40.0% | [21.9%, 61.3%] | Macro-F1: 0.2008 [0.1000, 0.3046] |
| Sentiment End Within-One | 100.0% | — | OrdMAE: 0.6, QWK: 0.6091 |
| Resolution Accuracy | 55.0% | [34.2%, 74.2%] | 3-class |
| Agent Handled Well Accuracy | 100.0% | [83.9%, 100.0%] | F1 (pos): 1.0 |
| Evidence Grounding Rate | 100.0% | [83.9%, 100.0%] | Verbatim transcript match |
| Resolution-Time MAE | 0.268 min | [0.1744, 0.3667] | 9 resolved calls |
| Latency (Mean / P95) | 0.0059s / 0.0086s | — | Per-call |

> **Grounding note**: The baseline's grounding rate is trivially 100% because its evidence quotes are copied verbatim from transcript segments. This number is only meaningful for LLM providers (ollama, anthropic) that must generate quotes from memory.

## Resolution Confusion Matrix
| True \ Pred | resolved | unresolved | unclear |
| ----------- | -------- | ---------- | ------- |
| **resolved** | 9 | 0 | 0 |
| **unresolved** | 6 | 0 | 3 |
| **unclear** | 0 | 0 | 2 |

## Failures Section (18 misclassified calls)

### 1. `heldout_01_partial_resolution` (Hard case: *partial_resolution*)
- **Issue**: Sentiment start mismatch: predicted 'angry', true 'upset'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from angry to satisfied.*

### 2. `heldout_02_callback_promised` (Hard case: *callback_promised*)
- **Issue**: Sentiment start mismatch: predicted 'angry', true 'upset'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from angry to satisfied.*

### 3. `heldout_03_wrong_customer_auth` (Hard case: *wrong_customer_authentication*)
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from neutral to satisfied.*

### 4. `heldout_04_language_barrier` (Hard case: *language_barrier*)
- **Issue**: Sentiment start mismatch: predicted 'angry', true 'upset'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from angry to satisfied.*

### 5. `heldout_05_hold_time_frustration` (Hard case: *hold_time_frustration*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'angry'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from upset to satisfied.*

### 6. `heldout_07_multi_issue_one_fixed` (Hard case: *multi_issue_one_fixed*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from satisfied to satisfied.*

### 7. `heldout_08_hold_time_abandoned` (Hard case: *hold_time_frustration*)
- **Issue**: Sentiment start mismatch: predicted 'neutral', true 'upset'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from neutral to satisfied.*

### 8. `heldout_09_partial_refund_dispute` (Hard case: *partial_resolution*)
- **Issue**: Sentiment start mismatch: predicted 'neutral', true 'upset'
- **Issue**: Sentiment end mismatch: predicted 'upset', true 'angry'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from neutral to upset.*

### 9. `heldout_10_callback_fulfilled` (Hard case: *callback_promised*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'upset'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from satisfied to satisfied.*

### 10. `heldout_11_multi_issue_both_fixed` (Hard case: *multi_issue_one_fixed*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from satisfied to satisfied.*

### 11. `heldout_12_agent_self_correction_wrong_plan` (Hard case: *agent_self_correction*)
- **Issue**: Sentiment start mismatch: predicted 'neutral', true 'upset'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from neutral to satisfied.*

### 12. `heldout_13_language_barrier_unresolved` (Hard case: *language_barrier*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from satisfied to neutral.*

### 13. `heldout_14_wrong_auth_lockout` (Hard case: *wrong_customer_authentication*)
- **Issue**: Sentiment start mismatch: predicted 'neutral', true 'upset'
- **Issue**: Sentiment end mismatch: predicted 'upset', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from neutral to upset.*

### 14. `heldout_15_hold_frustration_abandoned` (Hard case: *hold_time_frustration*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'angry'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from satisfied to satisfied.*

### 15. `heldout_16_multi_issue_none_fixed` (Hard case: *multi_issue_one_fixed*)
- **Issue**: Sentiment start mismatch: predicted 'neutral', true 'upset'
- **Issue**: Sentiment end mismatch: predicted 'upset', true 'angry'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from neutral to upset.*

### 16. `heldout_17_callback_agent_self_correct` (Hard case: *agent_self_correction*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'angry'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from upset to satisfied.*

### 17. `heldout_18_partial_resolution_shipping` (Hard case: *partial_resolution*)
- **Issue**: Sentiment start mismatch: predicted 'angry', true 'upset'
- **Issue**: Sentiment end mismatch: predicted 'upset', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from angry to upset.*

### 18. `heldout_20_language_barrier_partial` (Hard case: *language_barrier*)
- **Issue**: Sentiment start mismatch: predicted 'angry', true 'upset'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from angry to satisfied.*
