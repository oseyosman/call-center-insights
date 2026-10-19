# Evaluation Report: ANTHROPIC (HELDOUT split)

> **SIMULATED DATA NOTICE**: All metrics, transcripts, and audio files are synthetic and simulated for QA testing.

- **Evaluated Date**: 2026-10-19 19:47:05 UTC
- **Total Calls**: 20
- **Provider**: `anthropic`
- **Bootstrap**: 1000 resamples, seed=42

## Summary Metrics
| Metric | Score | 95% CI | Ordinal / Extra |
| ------ | ----- | ------ | --------------- |
| Sentiment Start Accuracy | 85.0% | [64.0%, 94.8%] | Macro-F1: 0.6257 [0.3809, 0.7500] |
| Sentiment Start Within-One | 100.0% | — | OrdMAE: 0.15, QWK: 0.8125 |
| Sentiment End Accuracy | 75.0% | [53.1%, 88.8%] | Macro-F1: 0.4164 [0.3072, 0.5000] |
| Sentiment End Within-One | 100.0% | — | OrdMAE: 0.25, QWK: 0.7967 |
| Resolution Accuracy | 80.0% | [58.4%, 91.9%] | 3-class |
| Agent Handled Well Accuracy | 100.0% | [83.9%, 100.0%] | F1 (pos): 1.0 |
| Evidence Grounding Rate | 100.0% | [83.9%, 100.0%] | Verbatim transcript match |
| Resolution-Time MAE | 0.653 min | [0.6000, 0.7278] | 9 resolved calls |
| Latency (Mean / P95) | 3.8023s / 7.04s | — | Per-call |

> **Grounding note**: The baseline's grounding rate is trivially 100% because its evidence quotes are copied verbatim from transcript segments. This number is only meaningful for LLM providers (ollama, anthropic) that must generate quotes from memory.

## Resolution Confusion Matrix
| True \ Pred | resolved | unresolved | unclear |
| ----------- | -------- | ---------- | ------- |
| **resolved** | 9 | 0 | 0 |
| **unresolved** | 1 | 6 | 2 |
| **unclear** | 0 | 1 | 1 |

## Failures Section (11 misclassified calls)

### 1. `heldout_01_partial_resolution` (Hard case: *partial_resolution*)
- **Issue**: Sentiment start mismatch: predicted 'neutral', true 'upset'
- **Model Explanation**: *The agent quickly fixed the email issue and escalated the critical static IP problem as high priority, but the static IP remains unresolved with a 24-48 hour wait, leaving the customer cautiously accepting.*

### 2. `heldout_02_callback_promised` (Hard case: *callback_promised*)
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *The agent placed a billing hold, emailed written confirmation, and scheduled a specialist callback, which calmed the customer, but the underlying invoice discrepancy is still pending review.*

### 3. `heldout_04_language_barrier` (Hard case: *language_barrier*)
- **Issue**: Sentiment start mismatch: predicted 'neutral', true 'upset'
- **Model Explanation**: *The agent patiently identified that the phone was stuck in international roaming mode, fixed it, and the customer confirmed data was working and expressed happiness.*

### 4. `heldout_05_hold_time_frustration` (Hard case: *hold_time_frustration*)
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *The agent apologized, identified the outdated code as the cause of the denial, contacted the doctor's office, and added an expedite flag, which calmed the customer, but the prescription won't be re-processed until tomorrow so the outcome isn't confirmed.*

### 5. `heldout_09_partial_refund_dispute` (Hard case: *partial_resolution*)
- **Issue**: Sentiment end mismatch: predicted 'upset', true 'angry'
- **Model Explanation**: *The agent explained the fee, acknowledged the defect argument and escalated with a timeline, but the 300 dollar refund remains pending approval and the customer ended the call still frustrated and threatening a credit card dispute.*

### 6. `heldout_12_agent_self_correction_wrong_plan` (Hard case: *agent_self_correction*)
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Model Explanation**: *The agent acknowledged the company's error, apologized, switched the customer to the agreed sixty dollar plan, issued a twenty dollar credit, and confirmed both changes, leaving the customer thankful.*

### 7. `heldout_13_language_barrier_unresolved` (Hard case: *language_barrier*)
- **Issue**: Resolution mismatch: predicted 'unresolved', true 'unclear'
- **Model Explanation**: *The agent patiently helped a non-native English speaker locate her policy by name and date of birth, but the transcript ends with the claim not yet filed because a police report number was still needed.*

### 8. `heldout_16_multi_issue_none_fixed` (Hard case: *multi_issue_one_fixed*)
- **Issue**: Sentiment end mismatch: predicted 'upset', true 'angry'
- **Model Explanation**: *The agent explained both charges clearly and offered a payment hold as a workaround, but neither the high bill nor the new fee was resolved, so the customer ended the call frustrated.*

### 9. `heldout_17_callback_agent_self_correct` (Hard case: *agent_self_correction*)
- **Issue**: Sentiment start mismatch: predicted 'upset', true 'angry'
- **Model Explanation**: *The agent apologized, found the drafts on the server, re-linked the account so all thirty-two drafts reappeared, and honestly acknowledged the previous agent's error, leaving the customer relieved and thankful.*

### 10. `heldout_18_partial_resolution_shipping` (Hard case: *partial_resolution*)
- **Issue**: Resolution mismatch: predicted 'unclear', true 'unresolved'
- **Model Explanation**: *The agent apologized, started an immediate mug replacement and a carrier trace for the missing headphones, and the customer accepted this, but the headphones issue stays open pending a 3-5 business day investigation.*

### 11. `heldout_20_language_barrier_partial` (Hard case: *language_barrier*)
- **Issue**: Sentiment end mismatch: predicted 'satisfied', true 'neutral'
- **Issue**: Resolution mismatch: predicted 'resolved', true 'unresolved'
- **Model Explanation**: *The agent recognized the urgency, located the account by address despite the missing account number, explained the known boiler fault and today's 2–5 PM repair window, and added a priority note, leaving the customer informed and thankful.*
