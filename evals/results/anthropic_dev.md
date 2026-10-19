# Evaluation Report: ANTHROPIC (DEV split)

> **SIMULATED DATA NOTICE**: All metrics, transcripts, and audio files are synthetic and simulated for QA testing.

- **Evaluated Date**: 2026-10-19 19:44:39 UTC
- **Total Calls**: 20
- **Provider**: `anthropic`
- **Bootstrap**: 1000 resamples, seed=42

## Summary Metrics
| Metric | Score | 95% CI | Ordinal / Extra |
| ------ | ----- | ------ | --------------- |
| Sentiment Start Accuracy | 85.0% | [64.0%, 94.8%] | Macro-F1: 0.8651 [0.4874, 1.0000] |
| Sentiment Start Within-One | 100.0% | — | OrdMAE: 0.15, QWK: 0.8958 |
| Sentiment End Accuracy | 85.0% | [64.0%, 94.8%] | Macro-F1: 0.6756 [0.5654, 0.7500] |
| Sentiment End Within-One | 100.0% | — | OrdMAE: 0.15, QWK: 0.9462 |
| Resolution Accuracy | 85.0% | [64.0%, 94.8%] | 3-class |
| Agent Handled Well Accuracy | 80.0% | [58.4%, 91.9%] | F1 (pos): 0.8824 |
| Evidence Grounding Rate | 100.0% | [83.9%, 100.0%] | Verbatim transcript match |
| Resolution-Time MAE | 0.258 min | [0.1540, 0.3600] | 10 resolved calls |
| Latency (Mean / P95) | 3.3519s / 4.4841s | — | Per-call |

> **Grounding note**: The baseline's grounding rate is trivially 100% because its evidence quotes are copied verbatim from transcript segments. This number is only meaningful for LLM providers (ollama, anthropic) that must generate quotes from memory.

## Resolution Confusion Matrix
| True \ Pred | resolved | unresolved | unclear |
| ----------- | -------- | ---------- | ------- |
| **resolved** | 10 | 0 | 0 |
| **unresolved** | 0 | 7 | 1 |
| **unclear** | 0 | 2 | 0 |

## Failures Section (10 misclassified calls)

### 1. `dev_02_polite_but_unresolved` (Hard case: *polite_but_unresolved*)
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *The agent checked the policy, apologized, and courteously explained that the expired coupon couldn't be extended, so the request went unfulfilled but the customer accepted it graciously and ended the call on a positive note.*

### 2. `dev_04_sarcasm` (Hard case: *sarcasm*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'angry'
- **Model Explanation**: *The customer began sarcastic and upset over an update crash, and the agent apologized, identified the version, and rolled back to 4.1, restoring the server, though the customer ended with lingering sarcasm and a warning.*

### 3. `dev_06_thanks_no_fix` (Hard case: *thanks_no_fix*)
- **Issue**: agent_handled_well mismatch: predicted False, true True
- **Model Explanation**: *The agent checked the account and gave a clear answer, but the fee waiver was denied and the agent didn't offer alternatives or try to retain the customer, who then said they'd cancel the card.*

### 4. `dev_07_escalation_to_supervisor` (Hard case: *escalation_to_supervisor*)
- **Issue**: agent_handled_well mismatch: predicted False, true True
- **Model Explanation**: *The agent did not acknowledge or apologize for the airline-initiated cancellation, quoted a $200 change fee right away, and only transferred to a supervisor queue, so the customer's anger grew and the rebooking was not resolved.*

### 5. `dev_08_rude_agent` (Hard case: *rude_agent*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'neutral'
- **Model Explanation**: *The agent was rude and dismissive, offered no help or explanation beyond citing the throttling policy, and the customer ended the call angry and planning a formal complaint.*

### 6. `dev_10_unclear_connection_drop` (Hard case: *call_dropped_unclear*)
- **Issue**: Sentiment end mismatch: predicted 'upset', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'unresolved', true 'unclear'
- **Model Explanation**: *The call dropped due to poor signal before the caller could give the exact location of the sparking line, so the report was not completed, though the agent promptly asked for the cross street.*

### 7. `dev_12_warranty_claim_rejected` (Hard case: *dispute_unresolved*)
- **Issue**: agent_handled_well mismatch: predicted False, true True
- **Model Explanation**: *The agent denied the warranty replacement due to a water ingress exclusion without explaining it or offering alternatives, so the customer's issue went unresolved and he ended the call angry.*

### 8. `dev_15_wrong_department` (Hard case: *transfer_unresolved*)
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *The agent quickly recognized the customer had reached the wrong department and transferred them to commercial lending, but the call ended before it could be confirmed that the customer's rate question was answered.*

### 9. `dev_16_disputed_charge_investigation` (Hard case: *in_progress_unclear*)
- **Issue**: Sentiment end mismatch: predicted 'upset', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'unresolved', true 'unclear'
- **Model Explanation**: *The agent promptly froze the card and opened a fraud investigation, but the customer's refund was not issued and was pending a five-business-day review, leaving the customer's question about same-day reimbursement unmet.*

### 10. `dev_18_system_outage_helpless` (Hard case: *unresolved_outage*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'angry'
- **Issue**: agent_handled_well mismatch: predicted False, true True
- **Model Explanation**: *The customer's production database remained down due to a regional outage, and the agent offered only a brief apology with no workaround, escalation, or ETA, so the customer's frustration grew.*
