# Stage 5: consolidate repaired evaluation results

The original evals and the first five repair waves have finished. The last late
sweep is running. Produce separate original and repaired sensitivity comparisons
before 11 AM Pacific on October23. Do not relabel changed-stack retries as original
frozen benchmark results.

Read canonical task outcomes from the existing head/tail extractor, then merge
only numeric verifier rewards from settled recovery driver overlays. Keep source
SHA256 and result provenance. Never overwrite a canonical trial or replace an
already scored original. Preserve raw fractional rewards, using the existing
reward>=1 full-solve predicate for pass@k.

Include original cumulative recovery2, fix1, and all six current repair roots:
cf2-verified-repair-20261023, cf2-image-query-repair-20261023,
cf2-remaining-repair-20261023, cf2-swe-repair-20261023,
cf2-cli-repair-20261023, cf2-late-repair-20261023.

Gate: each root's driver overlays settle and agree with its manifest, no duplicate
numeric replacement slots, all original hashes match, and no provisional result
is called final. Third/late controllers have a known derived metadata bug:
complete compares driver-count with node-count. Their RECOVERY_FINISHED status
plus all per-driver overlays is the completion signal; do not mutate their records.

Exclude git-repo-forensics from strict valid-task comparisons because the original
required Git fixture is absent even from the authoritative image. Report the
invalid task separately. All other unscored cases stay unknown, never zero.

Outputs: current completion/coverage/error census, paired pass@1/pass@3 for
Vanilla/Relay at epochs3/5/7 on each of TB2.1/SWE/TB-lite, eight-rep epoch3 pass@4
using 1-C(8-c,4)/C(8,4), task bootstrap uncertainty, and source provenance. Use the
same task panel within each comparison, disclose different panels between epochs.
Epoch5 remains an intermediate checkpoint of the seven-epoch schedule. Keep the
original reports alongside repaired reports. Refresh automatically after the last
sweep finishes, and retrieve the final small artifacts onto the Mac.

October23 03:47 PT: **GREEN / DONE for these eight recovery roots.** All roots
settled and agreed with their driver manifests. Strict valid coverage is
7958/8036 (99.03%), with575 recovered scores and78 unknown. The28 invalid Git
slots are separate. All21 original metric rows and paired bootstrap intervals
match their original reports;661 original and670 recovery result hashes match.
No originally numeric outcome changed and no duplicate numeric replacement was
accepted. Three actual automatic-retry chains contained no earlier numeric
verifier score. The current selected attempt is archived under attempts too;
that copy is not an additional retry.

Remote FINAL_READY snapshot03:45:11 PT has source SHA256
`3a8188bca6d000e29446964a73caa1bf577ea52b78412857de96c01077eb6593`.
The Mac pull verified15 result artifacts, then verified both analysis source
hashes and reproduced all14 generated output files byte-for-byte locally.
Artifacts: notes/artifacts/snowball-sft-repaired-readout-20261023/.
Report JSON SHA256:
`611923bf3e96c9faef0a660f20785dbcc8a71fd6e823062950b4399c0cdd66f4`.
Both owned watchers have finished. Later startup repair work is outside this
sealed revision and should produce a separately identified revision.
