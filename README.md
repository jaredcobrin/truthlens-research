# TruthLens

AI conversation auditing system that detects trust failures (contradictions, ignored instructions, bad reasoning, source errors) across long multi-turn conversations.

## Known Limitations / Future Work

### Claim Supersession (TODO)

Deterministic keyword-based supersession (detecting when a user corrects a prior claim) has been removed because it was unreliable — it would miss corrections that didn't use specific keywords ("scratch that", "actually forget that") and falsely mark unrelated claims as outdated when benign words like "update" appeared in context.

**Planned replacement:** LLM-based supersession detection — after each chunk is processed, an LLM call identifies which prior claims have been semantically corrected or invalidated by the current chunk, regardless of the exact wording used.

In the meantime, the auditor LLM still catches contradictions at audit time since all claims remain active in memory and are passed as context.
