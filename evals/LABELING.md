# Annotation Label Definitions – Call Center Insights Evaluation Dataset

> ⚠️ **SIMULATED DATA WARNING**: All transcripts in this dataset are strictly synthetic/simulated.
> No real customer recordings or real personal data are present.
> Labels come from a **single annotator** (Osman Yildiz) and have not been adjudicated by multiple reviewers.

---

## Overview

Each call record in `dev_transcripts.json`, `test_transcripts.json`, and `heldout_transcripts.json` carries a `ground_truth` object with the following six fields:

| Field | Type | Description |
|---|---|---|
| `customer_sentiment_start` | enum | Customer emotional tone in the opening third of the call |
| `customer_sentiment_end` | enum | Customer emotional tone in the closing third of the call |
| `agent_handled_well` | bool | Whether the agent's conduct was professional and helpful |
| `resolution` | enum | Whether the customer's primary issue was resolved |
| `resolution_time_minutes` | float \| null | Minutes from first customer utterance to the resolution evidence |
| `reason` | string | A single sentence explaining the resolution and agent evaluation |
| `evidence` | list | Verbatim quote(s) anchoring the resolution judgment |

---

## Sentiment Label Definitions

Sentiment is assessed from **transcript text only** (not prosody or acoustic features). The ordinal order is:

```
satisfied(0)  <  neutral(1)  <  upset(2)  <  angry(3)
```

| Label | Definition | Examples |
|---|---|---|
| `satisfied` | Customer is content, grateful, or happy. Positive language, thanks, praise. | "Thank you so much, that was exactly what I needed!" |
| `neutral` | Customer is neither pleased nor distressed. Transactional or matter-of-fact tone. | "Okay, I understand. I will wait for the callback." |
| `upset` | Customer is frustrated or disappointed but not confrontational. Mild complaints, sighs, expressions of inconvenience. | "That's really inconvenient. I needed this today." |
| `angry` | Customer is confrontational, hostile, or using strong language. Demands, threats, explicit expressions of rage. | "This is utter extortion! I'm filing a complaint!" |

### Annotation Rules for Sentiment

- **Sarcasm is labelled by intent**: "Oh fantastic, another brilliant update that crashed everything" is labelled `angry`, not `satisfied`.
- **Polite language ≠ satisfied**: A customer who politely accepts bad news ("Thanks for checking anyway") remains `neutral` unless genuine warmth is present.
- **Ordinal continuity**: Sentiment rarely jumps two levels in a single call. When it does (e.g., `angry` → `satisfied`), the transition must be explicitly supported by the transcript (e.g., immediate issue resolution with expressed relief).
- **Start = first third**: The opening ~30–35% of customer utterance lines by count.
- **End = last third**: The closing ~30–35% of customer utterance lines by count.

---

## `agent_handled_well` Label Definition

`true` when the agent demonstrates **all** of:
1. **Professionalism**: Polite language, no rudeness, no dismissiveness.
2. **Competence**: Takes appropriate action (escalates when needed, applies correct fixes).
3. **Honesty**: Does not mislead the customer; self-corrects when wrong.
4. **Empathy**: Acknowledges the customer's frustration or inconvenience.

`false` when the agent exhibits **any** of:
- Rude, dismissive, or condescending language ("whatever", "calm down", "not my problem").
- Clearly wrong advice that harms the customer (e.g., instructing reinstall without reason).
- Failure to escalate when clearly beyond their authority (hanging up on complex disputes).
- Deliberate misinformation.

> **Important**: An agent can handle a call well even if the issue is NOT resolved. Agent conduct and call outcome are independent dimensions.

---

## `resolution` Label Definitions

| Label | Definition |
|---|---|
| `resolved` | The customer's primary issue was **fully addressed** during the call. A replacement was shipped, an account was unlocked, a fee was reversed, etc. The evidence quote must demonstrate this. |
| `unresolved` | The primary issue was **not addressed** by end of call. Includes: callback promises (issue deferred), partial fixes (one of two issues solved), escalations still pending, explicit denials. |
| `unclear` | The call ended without a definitive outcome — e.g., the call was dropped, the process was restarted but not completed, or conflicting signals make the outcome ambiguous. |

### Annotation Rules for Resolution

- **Partial resolution → `unresolved`**: When a multi-issue call fixes one problem but leaves another open, the call is labelled `unresolved` because the customer's complete need is not met.
- **Callback promised → `unresolved`**: A promised follow-up action does not count as resolved; the issue remains pending.
- **Self-service provided → `resolved`**: If the agent gives instructions that the customer completes successfully on the call ("restart the phone… it's working now"), this counts as resolved.
- **Customer accepts outcome they dislike → still `unresolved`**: "Okay fine, I'll wait" after a denial is not resolution.

---

## `resolution_time_minutes` Calculation

```
resolution_time_minutes = (timestamp of first resolution evidence segment - timestamp of first customer utterance) / 60.0
```

- `null` for `unresolved` and `unclear` calls.
- The **earliest** evidence timestamp is used when multiple evidence items are present.
- Rounded to 2 decimal places.

---

## `evidence` Field

Each evidence item contains:
- `quote`: An exact verbatim substring from one of the `segments[].text` fields.
- `timestamp`: The `start` time of that segment in seconds.

Evidence quotes must appear **verbatim** in the transcript (after punctuation and case normalization). They anchor the resolution judgment and are used by the grounding checker.

---

## Hard-Case Types

| Type | Description |
|---|---|
| `customer_calms_down` | Customer starts angry and calms after successful resolution. |
| `polite_but_unresolved` | Customer remains polite throughout despite an unresolved request. |
| `resolved_still_angry` | Issue is fixed but customer remains angry at the company. |
| `sarcasm` | Customer uses sarcasm; VADER sentiment tools frequently fail here. |
| `interruptions` | Rapid speaker turn-taking; segment boundaries may be noisy. |
| `thanks_no_fix` | Customer says "thank you" or "thanks anyway" despite no resolution. |
| `escalation_to_supervisor` | Call ends with a transfer request; issue remains unresolved. |
| `rude_agent` | Agent is dismissive or hostile; `agent_handled_well=false`. |
| `standard_resolution` / `routine_fix` | Straightforward issue resolved without complications. |
| `call_dropped_unclear` | Abrupt ending makes outcome ambiguous. |
| `dispute_unresolved` | Customer disputes a charge or decision; not resolved on call. |
| `routine_cancellation` | Customer cancels a service; smooth process. |
| `transfer_unresolved` | Agent transfers call; outcome not known from this transcript. |
| `in_progress_unclear` | Issue is under investigation; resolution pending. |
| `unresolved_outage` | System outage acknowledged but not fixed on the call. |
| `satisfied_throughout` | Customer is happy throughout; easy call. |
| **Heldout-only types** | |
| `partial_resolution` | Multiple issues; at least one resolved, at least one pending. |
| `callback_promised` | Agent promises a specialist callback; issue not resolved now. |
| `wrong_customer_authentication` | Customer fails or corrects identity verification mid-call. |
| `language_barrier` | Non-native speaker; communication difficulty affects resolution. |
| `hold_time_frustration` | Customer is angry about wait time before agent answers. |
| `agent_self_correction` | Agent catches and corrects their own error during the call. |
| `multi_issue_one_fixed` | Two or more issues; outcomes differ between them. |

---

## Limitations

- **Single annotator**: All labels were assigned by one person (Osman Yildiz). No inter-annotator agreement score is available. Label confidence is not quantified.
- **TTS audio**: Transcripts were generated synthetically and rendered with macOS `say`. Real call audio has overlapping speech, acoustic noise, and codec artifacts.
- **Small sample**: 20 calls per split (60 total). Confidence intervals are wide.
- **English only**: All calls are in English; cross-lingual calls are simulated with simplified vocabulary.
- **n=20 per split**: Accuracy confidence intervals span roughly ±20pp at 95% Wilson CI. Treat results as directional, not definitive.
