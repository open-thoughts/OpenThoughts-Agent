# CoderForge v3 axolotl 0% — diagnosis

**Decision: NO retrain needed.** The data, training, and weights are all fine. The bug is a stale `chat_template.jinja` in the HF repo. Deleted. Requires only an eval relaunch to verify.

## Probe results (H1-H4 ruled out, H5 confirmed)

### H1: Tokenizer mismatch — RULED OUT
Stock Qwen/Qwen3-8B tokenizer (vocab 151669, eos=151645, pad=151643) decodes `input_ids[:200]` to perfect OpenHands system-prompt English across rows 0/1/100/500. No OOV. Max id = 151666 (`</tool_response>`, within vocab).

### H2: Labels mask — RULED OUT
Row 0: 21.9% loss-active (10921/49861 tokens). Rows 1/100/500: 23-26%. **Zero label/id mismatches** on active positions. Every loss-active span starts right after `<|im_start|>assistant\n` (which is masked) and ends at `<|im_end|>` (151645). Textbook assistant-only-loss.

### H3: Special-token drift — RULED OUT
Row 0 specials: 91× `<|im_start|>`, 91× `<|im_end|>`, 47× `<tool_call>`, 47× `</tool_call>`, 44× `<tool_response>`, 44× `</tool_response>`. All within Qwen's expected range. Assistant turn content decodes to clean OpenHands-XML: `I'll help you... <tool_call><function=str_replace_editor><parameter=command>view</parameter>...</function></tool_call><|im_end|>`.

### H4: Training dynamics — RULED OUT
`cf-v3-1000-axolotl__Qwen3-8B_387546.out`: loss 0.81 → 0.27, ppl 2.25 → 1.30 over 3 epochs, grad_norm peaked 16.35 (step 2, warmup) then decayed to 0.3-0.4. No NaN, no watchdog trips, clean epoch boundaries, `train_runtime` 2822s exit 0. Weights converged.

### H5: Stale `chat_template.jinja` in HF repo — **CONFIRMED (same bug as Sera v1)**

HF repo `laion/CoderForge-Preview-v3-1000-axolotl__Qwen3-8B` file hashes vs stock `Qwen/Qwen3-8B`:

| File | CF sha256[:12] | Stock sha256[:12] | Match |
|---|---|---|---|
| `tokenizer_config.json` (9732 B) | `d5d09f07b48c` | `d5d09f07b48c` | yes |
| `tokenizer.json` (11.4 MB) | `aeb13307a71a` | `aeb13307a71a` | yes |
| `chat_template.jinja` (292 B) | **`e7a3df47d72f`** | **not present** | **STALE** |

The 292-byte `chat_template.jinja` is axolotl's bare chatml emission:
```jinja
{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}
```

HF transformers + vLLM **prioritize `chat_template.jinja` over `tokenizer_config.json["chat_template"]`**, so the stock 4168-char Qwen template (which properly renders `tool_calls` + role:tool with `<tool_response>` wrapping) gets overridden by this naive one that:
- Ignores `tool_calls` field on assistant messages (dropped entirely)
- Ignores `<tool_response>` wrapping on role:tool observations (renders as raw `<|im_start|>tool\n...`)

At eval time OpenHands sends multi-turn messages with structured tool_calls + role:tool observations → naive template strips those fields → prompt becomes OOD garbage → model emits " 8888888..." (4096 tokens, finish_reason=length) or "with with with..." word salad. Identical signature to Sera v1 pre-fix.

Note: axolotl bypasses chat_template rendering for pre-tokenized data (`_is_dataset_already_tokenized`), so **training was unaffected**. The template only bites at inference.

## Fix applied

1. **Deleted** `chat_template.jinja` from HF repos (otagent env, `HfApi.delete_file`):
   - `laion/CoderForge-Preview-v3-1000-axolotl__Qwen3-8B`
   - `laion/CoderForge-Preview-v3-316-axolotl__Qwen3-8B` (same stale file, same bug)

2. **Patched** `/Users/benjaminfeuer/Documents/OpenThoughts-Agent/baselines/coderforge/configs/template_qwen3_8b_cf_v3.yaml`: `chat_template: chatml` → `chat_template: tokenizer_default` (prevents future retrains from re-introducing the bare jinja).

## Next actions for the user

1. **Verify fix on v3-1000**: relaunch swebench-verified-random-100-folders eval on `laion/CoderForge-Preview-v3-1000-axolotl__Qwen3-8B` with hermes parser **ON** (since CF emits OpenHands XML — see `CF_DIAGNOSIS.md` for that second issue; but confirm the model is no longer emitting garbage first). If the model now emits structured content (even if tool_calls still need parser work), we know the chat_template fix landed.
2. **If still 0% with coherent output**: the `CF_DIAGNOSIS.md` XML-vs-Hermes wire-format issue is the residual bug → either switch harness to openhands w/ parser off, or retrain on Hermes-JSON tool-call format.
3. **Commit and push** the local config change:
   ```bash
   cd /Users/benjaminfeuer/Documents/OpenThoughts-Agent
   git add baselines/coderforge/configs/template_qwen3_8b_cf_v3.yaml
   git commit -m "coderforge: use tokenizer_default chat_template (fix v1-style stale jinja bug)"
   git push
   ```

No SLURM job submitted. No retrain started.
