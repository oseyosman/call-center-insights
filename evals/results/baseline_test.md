# Evaluation Report: BASELINE (TEST split)

> **SIMULATED DATA NOTICE**: All metrics, transcripts, and audio files are synthetic and simulated for QA testing.

- **Evaluated Date**: 2026-10-19 02:33:47 UTC
- **Total Calls**: 20
- **Provider**: `baseline`

## Summary Metrics
| Metric | Score | Note |
| ------ | ----- | ---- |
| Sentiment Start Accuracy | 45.0% | Macro-F1: 0.4181 |
| Sentiment End Accuracy | 55.0% | Macro-F1: 0.3738 |
| Resolution Accuracy | 55.0% | Evaluated on 3-class problem |
| Evidence Grounding Rate | 100.0% | Verbatim transcript match |
| Resolution-Time MAE | 0.337 min | Evaluated on 9 calls |
| Latency (Mean / P95) | 0.004s / 0.0048s | Per-call latency |

## Resolution Confusion Matrix
| True \ Pred | resolved | unresolved | unclear |
| ----------- | -------- | ---------- | ------- |
| **resolved** | 9 | 0 | 2 |
| **unresolved** | 3 | 0 | 4 |
| **unclear** | 0 | 0 | 2 |

## Failures Section (15 misclassified calls)

### 1. `test_02_polite_but_unresolved` (Hard case: *polite_but_unresolved*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from satisfied to neutral.*

### 2. `test_03_resolved_still_angry` (Hard case: *resolved_still_angry*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'angry'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'resolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from upset to angry.*

### 3. `test_04_sarcasm` (Hard case: *sarcasm*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'angry'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'resolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from satisfied to neutral.*

### 4. `test_06_thanks_no_fix` (Hard case: *thanks_no_fix*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from satisfied to satisfied.*

### 5. `test_07_escalation_to_supervisor` (Hard case: *escalation_to_supervisor*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'angry'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'angry'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from upset to satisfied.*

### 6. `test_08_rude_agent` (Hard case: *rude_agent*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Sentiment end mismatch: predicted 'neutral', true 'angry'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from satisfied to neutral.*

### 7. `test_09_standard_refund_issued` (Hard case: *routine_fix*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'neutral'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from upset to satisfied.*

### 8. `test_10_call_dropped_unclear` (Hard case: *call_dropped_unclear*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'upset'
- **Issue**: Sentiment end mismatch: predicted 'neutral', true 'upset'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from satisfied to neutral.*

### 9. `test_11_sim_card_activation` (Hard case: *routine_fix*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from satisfied to satisfied.*

### 10. `test_12_denied_medical_authorization` (Hard case: *dispute_unresolved*)
- **Issue**: Sentiment end mismatch: predicted 'neutral', true 'angry'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from upset to neutral.*

### 11. `test_14_lost_wallet_card_freeze` (Hard case: *customer_calms_down*)
- **Issue**: Sentiment start mismatch: predicted 'angry', true 'upset'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from angry to satisfied.*

### 12. `test_15_wrong_number_transfer` (Hard case: *transfer_unresolved*)
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from neutral to satisfied.*

### 13. `test_16_pending_loan_underwriting` (Hard case: *in_progress_unclear*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from satisfied to satisfied.*

### 14. `test_18_discontinued_product_parts` (Hard case: *unresolved_outage*)
- **Issue**: Sentiment end mismatch: predicted 'neutral', true 'upset'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from neutral to neutral.*

### 15. `test_19_airline_baggage_delayed` (Hard case: *customer_calms_down*)
- **Issue**: Sentiment end mismatch: predicted 'neutral', true 'satisfied'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from upset to neutral.*
