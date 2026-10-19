# Evaluation Report: BASELINE (DEV split)

> **SIMULATED DATA NOTICE**: All metrics, transcripts, and audio files are synthetic and simulated for QA testing.

- **Evaluated Date**: 2026-10-19 02:30:59 UTC
- **Total Calls**: 20
- **Provider**: `baseline`

## Summary Metrics
| Metric | Score | Note |
| ------ | ----- | ---- |
| Sentiment Start Accuracy | 50.0% | Macro-F1: 0.4476 |
| Sentiment End Accuracy | 55.0% | Macro-F1: 0.4095 |
| Resolution Accuracy | 65.0% | Evaluated on 3-class problem |
| Evidence Grounding Rate | 100.0% | Verbatim transcript match |
| Resolution-Time MAE | 0.264 min | Evaluated on 9 calls |
| Latency (Mean / P95) | 0.006s / 0.009s | Per-call latency |
| Audio Subset WER | 25.09% | 10 calls transcribed via faster-whisper (base) |

## Resolution Confusion Matrix
| True \ Pred | resolved | unresolved | unclear |
| ----------- | -------- | ---------- | ------- |
| **resolved** | 9 | 0 | 1 |
| **unresolved** | 3 | 2 | 3 |
| **unclear** | 0 | 0 | 2 |

## Failures Section (14 misclassified calls)

### 1. `dev_02_polite_but_unresolved` (Hard case: *polite_but_unresolved*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from satisfied to satisfied.*

### 2. `dev_03_resolved_still_angry` (Hard case: *resolved_still_angry*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'angry'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from upset to angry.*

### 3. `dev_04_sarcasm` (Hard case: *sarcasm*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'angry'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'resolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from satisfied to satisfied.*

### 4. `dev_05_interruptions` (Hard case: *interruptions*)
- **Issue**: Sentiment start mismatch: predicted 'neutral', true 'upset'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from neutral to satisfied.*

### 5. `dev_06_thanks_no_fix` (Hard case: *thanks_no_fix*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from satisfied to satisfied.*

### 6. `dev_07_escalation_to_supervisor` (Hard case: *escalation_to_supervisor*)
- **Issue**: Sentiment end mismatch: predicted 'upset', true 'angry'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from upset to upset.*

### 7. `dev_08_rude_agent` (Hard case: *rude_agent*)
- **Issue**: Sentiment end mismatch: predicted 'upset', true 'angry'
- **Model Explanation**: *Baseline rule classified resolution as unresolved with customer sentiment shifting from neutral to upset.*

### 8. `dev_10_unclear_connection_drop` (Hard case: *call_dropped_unclear*)
- **Issue**: Sentiment start mismatch: predicted 'neutral', true 'upset'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from neutral to neutral.*

### 9. `dev_12_warranty_claim_rejected` (Hard case: *dispute_unresolved*)
- **Issue**: Sentiment start mismatch: predicted 'neutral', true 'upset'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'angry'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from neutral to satisfied.*

### 10. `dev_13_subscription_cancellation_smooth` (Hard case: *routine_cancellation*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from satisfied to satisfied.*

### 11. `dev_15_wrong_department` (Hard case: *transfer_unresolved*)
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from neutral to satisfied.*

### 12. `dev_16_disputed_charge_investigation` (Hard case: *in_progress_unclear*)
- **Issue**: Sentiment start mismatch: predicted 'neutral', true 'upset'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from neutral to neutral.*

### 13. `dev_18_system_outage_helpless` (Hard case: *unresolved_outage*)
- **Issue**: Sentiment start mismatch: predicted 'neutral', true 'angry'
- **Model Explanation**: *Baseline rule classified resolution as unresolved with customer sentiment shifting from neutral to angry.*

### 14. `dev_19_polite_cancel_flight` (Hard case: *polite_but_unresolved*)
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from neutral to satisfied.*
