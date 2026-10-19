# Evaluation Report: ANTHROPIC (TEST split)

> **SIMULATED DATA NOTICE**: All metrics, transcripts, and audio files are synthetic and simulated for QA testing.

- **Evaluated Date**: 2026-10-19 19:45:48 UTC
- **Total Calls**: 20
- **Provider**: `anthropic`
- **Bootstrap**: 1000 resamples, seed=42

## Summary Metrics
| Metric | Score | 95% CI | Ordinal / Extra |
| ------ | ----- | ------ | --------------- |
| Sentiment Start Accuracy | 80.0% | [58.4%, 91.9%] | Macro-F1: 0.7964 [0.3757, 0.9444] |
| Sentiment Start Within-One | 100.0% | — | OrdMAE: 0.2, QWK: 0.8551 |
| Sentiment End Accuracy | 80.0% | [58.4%, 91.9%] | Macro-F1: 0.7714 [0.4629, 0.9407] |
| Sentiment End Within-One | 100.0% | — | OrdMAE: 0.2, QWK: 0.9263 |
| Resolution Accuracy | 80.0% | [58.4%, 91.9%] | 3-class |
| Agent Handled Well Accuracy | 90.0% | [69.9%, 97.2%] | F1 (pos): 0.9444 |
| Evidence Grounding Rate | 100.0% | [83.9%, 100.0%] | Verbatim transcript match |
| Resolution-Time MAE | 0.226 min | [0.1420, 0.3180] | 10 resolved calls |
| Latency (Mean / P95) | 3.4462s / 5.9386s | — | Per-call |

> **Grounding note**: The baseline's grounding rate is trivially 100% because its evidence quotes are copied verbatim from transcript segments. This number is only meaningful for LLM providers (ollama, anthropic) that must generate quotes from memory.

## Resolution Confusion Matrix
| True \ Pred | resolved | unresolved | unclear |
| ----------- | -------- | ---------- | ------- |
| **resolved** | 10 | 1 | 0 |
| **unresolved** | 2 | 5 | 0 |
| **unclear** | 0 | 1 | 1 |

## Failures Section (11 misclassified calls)

### 1. `test_02_polite_but_unresolved` (Hard case: *polite_but_unresolved*)
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *The customer's return was outside the 30-day window and could not be processed, but the agent politely explained the policy and the customer accepted it graciously.*

### 2. `test_03_resolved_still_angry` (Hard case: *resolved_still_angry*)
- **Issue**: Resolution mismatch: predicted 'unresolved', true 'resolved'
- **Model Explanation**: *The agent quickly restored power and offered a $250 credit toward the $300 loss, but the customer stayed furious and said they would escalate to the state utility board.*

### 3. `test_04_sarcasm` (Hard case: *sarcasm*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'angry'
- **Model Explanation**: *The agent apologized, diagnosed the issue, and restored all the deleted files from a snapshot, which resolved the problem even though the customer stayed a bit sarcastic.*

### 4. `test_05_interruptions` (Hard case: *interruptions*)
- **Issue**: Sentiment start mismatch: predicted 'angry', true 'upset'
- **Model Explanation**: *The agent stayed calm despite the customer's interruptions, rebooked them on an 11:15 AM United non-stop that met the noon deadline, and the customer thanked them.*

### 5. `test_06_thanks_no_fix` (Hard case: *thanks_no_fix*)
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *The agent clearly explained that cosmetic rock chips under one inch are not covered, and the customer thanked her and ended the call satisfied with the answer.*

### 6. `test_07_escalation_to_supervisor` (Hard case: *escalation_to_supervisor*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'angry'
- **Issue**: Sentiment end mismatch: predicted 'upset', true 'angry'
- **Model Explanation**: *The customer disputed a $300 pet-hair cleaning fee, the agent explained the branch's photo evidence and couldn't remove the fee, then transferred the call to a supervisor at the customer's request, leaving the dispute unresolved.*

### 7. `test_09_standard_refund_issued` (Hard case: *routine_fix*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'neutral'
- **Model Explanation**: *The agent quickly identified the duplicate charge, voided it and issued an immediate refund, and the customer confirmed receipt of the refund email and thanked her.*

### 8. `test_10_call_dropped_unclear` (Hard case: *call_dropped_unclear*)
- **Issue**: Resolution mismatch: predicted 'unresolved', true 'unclear'
- **Model Explanation**: *The customer reported a blown tire on highway 101 but the line dropped before the location was confirmed, and the agent responded appropriately by asking for the mile marker, so the issue remains unresolved.*

### 9. `test_12_denied_medical_authorization` (Hard case: *dispute_unresolved*)
- **Issue**: agent_handled_well mismatch: predicted False, true True
- **Model Explanation**: *The customer's urgent MRI denial was not resolved, and the agent gave only a brief, rigid explanation without offering an expedited appeal, a way to submit the missing records, or other next steps, so the customer escalated to threatening a state complaint.*

### 10. `test_15_wrong_number_transfer` (Hard case: *transfer_unresolved*)
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *The customer dialed the wrong business, and the agent politely clarified the mistake and transferred the call to the pharmacy line, which the customer appreciated.*

### 11. `test_18_discontinued_product_parts` (Hard case: *unresolved_outage*)
- **Issue**: agent_handled_well mismatch: predicted False, true True
- **Model Explanation**: *The customer's request for a carburetor diaphragm couldn't be fulfilled because the model is discontinued, and the agent gave no alternatives (aftermarket parts, referrals, or empathy) when the customer expressed frustration.*
