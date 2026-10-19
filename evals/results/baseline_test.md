# Evaluation Report: BASELINE (TEST split)

> **SIMULATED DATA NOTICE**: All metrics, transcripts, and audio files are synthetic and simulated for QA testing.

- **Evaluated Date**: 2026-10-19 19:29:47 UTC
- **Total Calls**: 20
- **Provider**: `baseline`
- **Bootstrap**: 1000 resamples, seed=42

## Summary Metrics
| Metric | Score | 95% CI | Ordinal / Extra |
| ------ | ----- | ------ | --------------- |
| Sentiment Start Accuracy | 45.0% | [25.8%, 65.8%] | Macro-F1: 0.4181 [0.1923, 0.6342] |
| Sentiment Start Within-One | 90.0% | — | OrdMAE: 0.7, QWK: 0.4712 |
| Sentiment End Accuracy | 55.0% | [34.2%, 74.2%] | Macro-F1: 0.3738 [0.1818, 0.5534] |
| Sentiment End Within-One | 85.0% | — | OrdMAE: 0.65, QWK: 0.4773 |
| Resolution Accuracy | 55.0% | [34.2%, 74.2%] | 3-class |
| Agent Handled Well Accuracy | 95.0% | [76.4%, 99.1%] | F1 (pos): 0.973 |
| Evidence Grounding Rate | 100.0% | [83.9%, 100.0%] | Verbatim transcript match |
| Resolution-Time MAE | 0.337 min | [0.2356, 0.4389] | 9 resolved calls |
| Latency (Mean / P95) | 0.0043s / 0.0052s | — | Per-call |

> **Grounding note**: The baseline's grounding rate is trivially 100% because its evidence quotes are copied verbatim from transcript segments. This number is only meaningful for LLM providers (ollama, anthropic) that must generate quotes from memory.

## Resolution Confusion Matrix
| True \ Pred | resolved | unresolved | unclear |
| ----------- | -------- | ---------- | ------- |
| **resolved** | 9 | 0 | 2 |
| **unresolved** | 3 | 0 | 4 |
| **unclear** | 0 | 0 | 2 |

## Failures Section (16 misclassified calls)

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

### 4. `test_05_interruptions` (Hard case: *interruptions*)
- **Issue**: agent_handled_well mismatch: predicted False, true True
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from upset to satisfied.*

### 5. `test_06_thanks_no_fix` (Hard case: *thanks_no_fix*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from satisfied to satisfied.*

### 6. `test_07_escalation_to_supervisor` (Hard case: *escalation_to_supervisor*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'angry'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'angry'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from upset to satisfied.*

### 7. `test_08_rude_agent` (Hard case: *rude_agent*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Sentiment end mismatch: predicted 'neutral', true 'angry'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from satisfied to neutral.*

### 8. `test_09_standard_refund_issued` (Hard case: *routine_fix*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'neutral'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from upset to satisfied.*

### 9. `test_10_call_dropped_unclear` (Hard case: *call_dropped_unclear*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'upset'
- **Issue**: Sentiment end mismatch: predicted 'neutral', true 'upset'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from satisfied to neutral.*

### 10. `test_11_sim_card_activation` (Hard case: *routine_fix*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from satisfied to satisfied.*

### 11. `test_12_denied_medical_authorization` (Hard case: *dispute_unresolved*)
- **Issue**: Sentiment end mismatch: predicted 'neutral', true 'angry'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from upset to neutral.*

### 12. `test_14_lost_wallet_card_freeze` (Hard case: *customer_calms_down*)
- **Issue**: Sentiment start mismatch: predicted 'angry', true 'upset'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from angry to satisfied.*

### 13. `test_15_wrong_number_transfer` (Hard case: *transfer_unresolved*)
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from neutral to satisfied.*

### 14. `test_16_pending_loan_underwriting` (Hard case: *in_progress_unclear*)
- **Issue**: Sentiment start mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from satisfied to satisfied.*

### 15. `test_18_discontinued_product_parts` (Hard case: *unresolved_outage*)
- **Issue**: Sentiment end mismatch: predicted 'neutral', true 'upset'
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *Baseline rule classified resolution as unclear with customer sentiment shifting from neutral to neutral.*

### 16. `test_19_airline_baggage_delayed` (Hard case: *customer_calms_down*)
- **Issue**: Sentiment end mismatch: predicted 'neutral', true 'satisfied'
- **Model Explanation**: *Baseline rule classified resolution as resolved with customer sentiment shifting from upset to neutral.*
