# Discovery Clock

Film a checklist job. Type the steps. Discovery Clock says which steps happened, when each one started, and which were skipped, with a probability for each.

Built at the Gemini 4 NYC hackathon on Nov 21 2026, for Problem Statement Two ("Element 119"). Every file in this repository was written that day after hacking began at 10:30 AM ET. The commit history shows it.

It is a review aid. A person checks every flagged step and signs.

## The idea in one paragraph

A single Gemini call can already watch a video and list which steps happened. What it cannot do is say how sure it is. Discovery Clock asks four witnesses about the same video and combines them with Bayes' rule. The first witness is the Gemini Decisions API: one call asks, for every step and every time point, "had this step started by now?", which gives a calibrated curve per step in about one second. The other three are Gemini 4 reasoning models that each write their own timeline. Each witness has a reliability measured on held-out tuning clips. The output for every step is a probability that it happened and a posterior over when it started.

## How it works

1. `clock/media.py` makes a small copy of the video with ffmpeg (short side 360 px, 10 frames per second).
2. In parallel:
   - `clock/ladder.py` builds the question ladder and `clock/decide.py` sends it to `gemini-decision-1`. For 6 steps on a 25 second clip that is about 100 yes/no questions in one call. Isotonic regression makes each step's answers monotone; the half-rise point is the ladder's time estimate.
   - `clock/witness.py` asks `gemini-4-hydrogen` at 4 frames per second, `gemini-4-hydrogen` again at 2 frames per second, and `gemini-4-neon` at 4 frames per second, all at medium media resolution, for a timeline: for each step, did it happen and at what second. The prompt carries a completion rule: a step counts only if it is carried through and its result is still in place at the end, unless a later step on the list is what reverses it.
3. `clock/fusion.py` combines them. For each step it compares two hypotheses, "happened, starting at time tau" and "never happened". Each witness contributes a likelihood: its hit rate, its false alarm rate, and a Laplace error around the true start time with a small share of gross errors. The ladder contributes a Gaussian score and its own time estimate. The posterior over tau gives the start time and its range. One temperature on the log odds corrects for the fact that the witnesses watch the same video and so make related mistakes.
4. `clock/engine.py` runs all of this with a timeout on every call and a deadline for the whole read. After 15 seconds it returns as soon as two witnesses have answered. A witness that fails or is late is left out. The Decisions ladder alone is the fallback and is shown as the "first read" while the others are still watching.
5. `app/` is the web page: record on a phone or upload a file, type the checklist, read the step table. Each row shows the verdict, the start time with its range, the posterior curve on a shared time axis, what each witness said, and a 4 second clip.
6. `clock/auditor.py` is an optional third stage that runs after the full read: a Gemini Managed Agent (Antigravity, with a Google-hosted sandbox) gets still frames from the start, the end, and just before and after each reported step. It writes down whether each step's lasting result is in place in each frame, then writes and runs a Python file in its sandbox that applies fixed rules to those observations, and returns "confirm", "dispute" or "cannot tell" per step with a plain-text sign-off sheet. The file also prints a SHA-256 over a one-time code and the verdicts; a matching hash shows the sandbox run really happened. The same rules run again on this machine as a cross-check. The audit never changes the full read. It adds a flag for a person. In the app it runs on its own thread after the full read is on screen and shows as a third status line, a mark on each row, and a sign-off sheet; `CLOCK_AUDIT=0` switches it off.

Model jobs:

| Model | Job |
|-|-|
| `gemini-decision-1` (Decisions API) | The ladder: steps x time points in one call. First read in about a second, and a graded score that the yes/no witnesses lack. |
| `gemini-4-hydrogen` | Two fast timelines of the whole video, at 4 and at 2 frames per second. |
| `gemini-4-neon` | A third timeline from a stronger model. |
| `antigravity-preview-09-2026` (Managed Agent, driven by `gemini-4-hydrogen`) | The auditor: checks start and end frames against each verdict and runs the rules in a sandbox. Optional, about 50 seconds. |
| `gemini-4-carbon` | The baseline: one call at high thinking, asked for the same output. Not part of the system. |

## Does it work

### Test data

The DeepMind Perception Test (validation split, CC BY 4.0): short phone videos filmed by crowd participants, with human labels for each action and its start and end time. I did not write or record any of it.

For each clip the checklist is the clip's own labeled actions plus two steps that never happen in it, picked by a hash of the clip id. The system must say which steps happened, when each started, and flag the two planted skips.

Clips were assigned by a fixed rule before any model output existed (`eval/splits.json`): eligible clips sorted by the SHA-1 of their id; the first 30 are for tuning, the next 120 are the first test split, and the next four blocks of 100 are the second, third, fourth and fifth test splits.

The system went through three versions today. Version 1 used Hydrogen and Neon witnesses at low resolution and 1 frame per second. Version 2 gave every witness medium resolution and 2 or 4 frames per second. Version 3, the one in this repository, adds a completion rule to the witness prompt. Each version was scored once on clips it had never seen, and the tables below say which is which.

### Version 1 on the first test split (120 clips, 626 steps: 386 real, 240 planted skips)

Every parameter was fitted on the 30 tuning clips and frozen before this split was scored. "Skip F1" is for flagging steps that did not happen. Timing columns are the share of real steps whose start was placed within that distance of the label.

| System | Skip F1 | Ranking (AUROC) | Within 1 s | Within 2 s | Within 5 s | Median time per clip |
|-|-|-|-|-|-|-|
| Decisions ladder alone | 0.79 | 0.89 | 35% | 52% | 66% | 1.4 s |
| Hydrogen, one call | 0.86 | none (yes/no only) | 55% | 67% | 75% | 6.8 s |
| Hydrogen at 4 fps, one call | 0.88 | none | 55% | 68% | 76% | 7.2 s |
| Neon, one call | 0.85 | none | 55% | 67% | 73% | 11.3 s |
| Fused system | 0.87 | 0.96 | 60% | 70% | 77% | 11.7 s |

Paired bootstrap over clips, fused minus single call, 95 percent interval:
- against Hydrogen at 4 fps: skip F1 -0.006 [-0.027, +0.011]
- against Hydrogen: skip F1 +0.013 [-0.005, +0.030]
- against Neon: skip F1 +0.029 [+0.004, +0.052]

What this says, plainly:
- Fusion does not beat the best single call on yes/no accuracy. It ties it.
- Fusion gives something no single call gives: a probability per step that ranks real steps above skipped ones with an AUROC of 0.96.
- Fusion places more steps within one second (60% against 55%).
- The Decisions ladder alone is the fastest by a wide margin and the least accurate.

### What changed after seeing those results

The fused probabilities were overconfident: of the steps it gave less than 20 percent, 15 percent had really happened (expected calibration error 0.092). The witnesses watch the same video, so their mistakes are related and the model treated them as independent. I added one temperature on the log odds, fitted on the tuning clips only. It changes no ranking.

Because I had already looked at the first test split, I scored the changed system once on a second split that had never been run.

### Version 1 with the temperature, on the second test split (99 clips, 506 steps)

| System | Skip F1 | Ranking (AUROC) | Within 1 s | Within 2 s | Within 5 s | Median time per clip |
|-|-|-|-|-|-|-|
| Decisions ladder alone | 0.81 | 0.90 | 35% | 51% | 62% | 1.1 s |
| Hydrogen, one call | 0.86 | none | 60% | 70% | 76% | 6.8 s |
| Hydrogen at 4 fps, one call | 0.89 | none | 61% | 72% | 77% | 7.0 s |
| Neon, one call | 0.88 | none | 63% | 72% | 78% | 10.3 s |
| Fused system | 0.88 | 0.96 | 63% | 73% | 78% | 10.8 s |

Calibration of the fused probability on this split: expected calibration error 0.057.

| Fused probability | Steps | Mean probability | Share that really happened |
|-|-|-|-|
| 0.0 to 0.2 | 203 | 0.03 | 0.13 |
| 0.2 to 0.4 | 12 | 0.27 | 0.33 |
| 0.4 to 0.6 | 8 | 0.50 | 0.75 |
| 0.6 to 0.8 | 5 | 0.72 | 0.80 |
| 0.8 to 1.0 | 278 | 0.98 | 0.96 |

The reported time range contains the labeled start for 78 percent of timed steps.

### What changed after the field takes: version 2

I recorded a bag-packing checklist on a phone, with a known script per take (see `results/field.md`). On the take where the AirPods case was left out, version 1 was right in the live run and wrong in two re-runs of the same file: the Hydrogen witness at low resolution and 1 frame per second credited the Hello Panda box going into the bag to the AirPods case. Benchmark labels never name specific objects, so the benchmark could not show this.

A check of witness settings on two takes (3 calls each) showed the low-resolution Hydrogen witness wrong on that take in 3 of 3 calls, and every witness at medium resolution and 2 or 4 frames per second right in 6 of 6. Version 2 replaces the low-resolution witnesses. Reliabilities and the temperature were refitted on the tuning clips. Nothing else changed.

Version 2 was then scored once on a third split of 100 clips that had never been run.

### Version 2 on the third test split (100 clips, 508 steps: 308 real, 200 planted skips)

| System | Skip F1 | Ranking (AUROC) | Within 1 s | Within 2 s | Within 5 s | Median time per clip |
|-|-|-|-|-|-|-|
| Decisions ladder alone | 0.78 | 0.91 | 33% | 52% | 64% | 1.1 s |
| Hydrogen at 4 fps, one call | 0.88 | none | 61% | 70% | 80% | 7.6 s |
| Hydrogen at 2 fps, one call | 0.85 | none | 61% | 68% | 76% | 7.0 s |
| Neon at 4 fps, one call | 0.85 | none | 59% | 67% | 75% | 11.7 s |
| Fused system, version 2 | 0.88 | 0.94 | 64% | 73% | 80% | 12.1 s |

Paired bootstrap over clips, fused minus single call, 95 percent interval:
- against Hydrogen at 4 fps: skip F1 -0.002 [-0.019, +0.018]; within 2 s +0.032 [+0.006, +0.062]
- against Hydrogen at 2 fps: skip F1 +0.024 [+0.006, +0.043]; within 2 s +0.059 [+0.024, +0.099]
- against Neon at 4 fps: skip F1 +0.031 [+0.012, +0.052]; within 2 s +0.068 [+0.040, +0.097]

Expected calibration error 0.071. Of the 282 steps given 0.8 or more, 93 percent had happened; of the 195 given under 0.2, 10 percent had. The reported range contains the labeled start for 77 percent of timed steps.

### What changed after the fake-out takes: version 3

A step can be faked. In one field take the AirPods case was picked up, moved toward the bag and put back on the table. Version 2 called that step done in the live run and in 2 of 3 re-runs. A second fake-out with another object fooled it in 2 of 3 re-runs.

Three fixes were tried in parallel on the field takes:
- A completion rule in the witness prompt: a step counts only if it is carried through and its result is still in place at the end of the video, unless a later step on the list reverses it. On the fake-out take, 18 of 18 witness calls then said not done. An independent re-run agreed, also with the checklist reworded so it shared no nouns with the first.
- An end-state check by a separate call. Neon on the whole video caught the fake-out 6 of 6 times with no false vetoes, at 15 to 20 seconds a call. Too slow, so the idea went into the wording of the rule instead.
- Extra Decisions API questions about undoing and end state. No question separated a faked step from a real one.

Version 3 ships the rule. Witness reliabilities and the temperature were refitted on the tuning clips with the new prompt, and the result was scored once on a fourth split that had never been run.

### Version 3 on the fourth test split (99 clips, 537 steps: 339 real, 198 planted skips)

| System | Skip F1 | Precision | Recall | Ranking (AUROC) | Within 1 s | Within 2 s | Median time per clip |
|-|-|-|-|-|-|-|-|
| Decisions ladder alone | 0.77 | 0.73 | 0.81 | 0.90 | 35% | 52% | 1.4 s |
| Hydrogen at 4 fps, one call | 0.89 | 0.81 | 0.97 | none | 57% | 72% | 8.6 s |
| Hydrogen at 2 fps, one call | 0.86 | 0.78 | 0.96 | none | 58% | 71% | 8.5 s |
| Neon at 4 fps, one call | 0.87 | 0.79 | 0.97 | none | 56% | 70% | 14.7 s |
| Fused system, version 3 | 0.88 | 0.81 | 0.96 | 0.97 | 62% | 72% | 14.7 s |

Paired bootstrap, fused minus Hydrogen at 4 fps: skip F1 -0.005 [-0.024, +0.015]. Expected calibration error 0.040: of the 288 steps given 0.8 or more, 98 percent had happened; of the 186 given under 0.2, 10 percent had.

The rule makes every witness stricter. Against version 2 on the third split, recall of planted skips rises from 0.90 to 0.96 and precision falls from 0.86 to 0.81: it flags more, and more of the flags are real steps. For a tool that tells a person where to look, that is the side to err on. Latency went up because Neon at 4 frames per second is the slowest call; in the app the full read returns after 15 seconds with whichever witnesses have answered.

### Replication: version 3 again, on a fifth split (100 clips, 513 steps: 313 real, 200 planted skips)

The fourth split was scored once when version 3 was made. To see whether its numbers repeat, I ran the frozen version 3 once more on a fifth split that no system had touched. I wrote the plan down before running it and nothing was changed afterwards.

| System | Skip F1 | Precision | Recall | Ranking (AUROC) | Within 1 s | Within 2 s | Median time per clip |
|-|-|-|-|-|-|-|-|
| Decisions ladder alone | 0.78 | 0.73 | 0.82 | 0.90 | 32% | 50% | 1.6 s |
| Hydrogen at 4 fps, one call | 0.85 | 0.79 | 0.93 | none | 58% | 67% | 6.5 s |
| Hydrogen at 2 fps, one call | 0.85 | 0.78 | 0.93 | none | 58% | 68% | 6.6 s |
| Neon at 4 fps, one call | 0.85 | 0.77 | 0.94 | none | 58% | 69% | 13.0 s |
| Fused system, version 3 | 0.86 | 0.81 | 0.91 | 0.95 | 61% | 71% | 13.1 s |

Paired bootstrap, fused minus Hydrogen at 4 fps: skip F1 +0.004 [-0.014, +0.022]; steps within 2 seconds +0.035 [+0.007, +0.065]. Expected calibration error 0.053: of the 269 steps given 0.8 or more, 95 percent had happened; of the 197 given under 0.2, 12 percent had. With the Decisions ladder removed, the same three witnesses rank at 0.92 instead of 0.95.

Every headline number is a little lower than on the fourth split (skip F1 0.88 to 0.86, ranking 0.97 to 0.95, "98 percent" to "95 percent"), so the fourth split was the kinder of the two. The pattern repeats: a tie with the best single call on yes/no, a ranking score the single calls do not have, and timing level or slightly better.

The two version 3 splits pooled (199 clips, 1,050 steps: 652 real, 398 planted skips; `eval/pooled_v3.py`): fused skip F1 0.87 (precision 0.81, recall 0.94) against 0.87 for Hydrogen at 4 fps alone; ranking 0.96, and 0.93 with the ladder removed; 61 percent of steps within 1 second and 71 percent within 2, against 58 and 70 for Hydrogen at 4 fps. Paired bootstrap over the 199 clips, fused minus Hydrogen at 4 fps: skip F1 -0.000 [-0.014, +0.013]; steps within 2 seconds +0.017 [-0.002, +0.036]. Expected calibration error 0.041: of the 557 steps given 0.8 or more, 97 percent had happened; of the 383 given under 0.2, 11 percent had. These pooled figures are the ones to quote for version 3 (`results/v3_pooled_report.json`).

The picture is the same on all five splits (518 held-out clips): the fused system ties the best single call on yes/no accuracy, is level or a few points better on timing, and is the only one that gives a probability.

### Pushing length until it breaks

Everything above is 10 to 35 second clips. As a stress test I ran version 3 on two CaptainCook4D kitchen recordings (head-mounted camera, human labels for every recipe step), 14 to 40 times longer. Two recordings is an anecdote, not a score; the per-step output is in `results/long_video.json`.

| Recording | Length | Steps | Decisions ladder | Slowest witness | Steps done, read as done | Skipped step caught | Median start error |
|-|-|-|-|-|-|-|-|
| Microwave egg sandwich | 7.9 min | 12, one skipped | 20.7 s, 101K tokens, one time point every 12 s | 78 s, 163K tokens | 11 of 11 | 0 of 1 | 25 s |
| Second recipe | 21.5 min | 16, none skipped | 40.1 s, 206K tokens, one time point every 33 s | 142 s, 445K tokens | 16 of 16 | none to catch | 6 s |

What broke:
- The one skipped step. In the 8 minute recording every witness and the fused system said the skipped step had happened.
- Timing. Start times are off by 6 to 25 seconds at the median, against half a second on short clips.
- The ladder. One Decisions call still holds a 21 minute video (206K of its 262K tokens), but with 16 steps it can only afford a time point every 33 seconds, so its own start times are off by more than a minute. A coarse pass followed by a fine pass on each step's window is the obvious next step and is not built.
- Latency. The full read takes one to two and a half minutes, not twelve seconds.

What held: nothing hit a context limit, and no step that was done was read as missing.

### The Carbon baseline

One `gemini-4-carbon` call at high thinking, same prompt and output format. It is the most accurate single call on the clips it finished, and too slow to use live: about 110 seconds per clip at the median, and in my runs it timed out often enough that the wrapper fell back to Neon on a large share of clips. See `results/` for the exact counts at the time of the last run.

### Where it fails

- Vague labels. "Moving object(s) around" and "Holding something in a state" account for the largest share of errors on both test splits. These actions arguably happen through most of a clip, so their labeled start is hard to match. Without those two classes the fused system scores skip F1 0.90 and 79 percent within 2 seconds on the second split.
- Steps every witness misses. When all four sources miss a real step the fused probability is near zero, and 13 percent of those steps were real. A person still has to check flagged steps.
- The Decisions ladder on real phone footage. On the first field recording it scored three of five real steps low; the reasoning models carried the verdict.
- Which object did it. At low resolution a witness can see that something went into the bag and credit the wrong item. Version 2 reduced this on my takes; it is not measured at scale.
- A result that was already in place. In two field takes the bag was already open and was only touched, and "unzip the bag" was called done in every run. The system reads the state and credits the step. The optional audit stage disputes this when the starting state is visible in the first frame, and not otherwise (see the audit section below).
- An item packed out of order. In one field take the second item was packed before the first; under the completion rule the second item was wrongly flagged in 2 of 3 runs.

### Things I tried that did not help (tuning split)

- A zoom pass: Neon re-watches 7 seconds around each estimate at higher resolution. Shown a short clip, it said yes to 44 percent of steps that never happened. Left out.
- 720p input at high media resolution. About the same accuracy at four times the latency.
- Neon at medium thinking. No gain, twice the latency.
- A change-point likelihood over the raw ladder answers. Worse timing than the simple half-rise estimate.
- Shorter question wording for the ladder. Fewer tokens, clearly worse answers.
- A clock burned into the video frames. No better than plain video: the Decisions model already tracks time.
- A second, slow-motion Decisions pass (`clock/zoom.py`): cut a few seconds around each estimate, slow them down four times so one sampled frame is a quarter second of the scene, and ask the ladder again. On the tuning clips the ladder's own median timing error fell from 1.0 to 0.8 seconds and steps within 1 second rose from 48 to 56 percent. The fused result did not change (the reasoning witnesses already time more tightly), and on the 8 minute recording it made timing worse (median error 17 s to 24 s). Not used.

### Field check

Recordings made at the event on a phone, each with a known script, are in `results/field.md`: a six-step bag-packing checklist, 9 takes, 3 runs each under version 3. They are a small extra check on real footage, recorded by me, and not part of the scores above. Takes 1 and 2 were used to choose version 2 and take 6 to choose version 3; the file says which.

Under version 3: 151 of 162 step verdicts right. Of 60 steps that were not done (never touched, faked, or put in and taken back out), 51 read "not found", 3 read "unsure" and 6 were wrongly read as done. Of 102 steps that were done, 100 read "done" and 2 were wrongly read as not found. All 6 wrong "done" verdicts are "unzip the bag" in two takes where the bag was already open and only touched. The 3 "unsure" are one fake-out where the 2 frames-per-second witness kept saying yes. The 2 wrong "not found" are an item packed out of order.

That fake-out is not solved. The faked item in take 7 sits close to the "done" cut (0.51 to 0.66 across runs), and later runs of the same recording crossed it. Counting every version 3 run of the two fake-out takes (6 runs above, 4 in the audit check below, 4 with the five-line list below): the faked item read "not found" in 7 of 14, "unsure" in 5 and "done" in 2. Version 2 read it as done in 4 of 6.

The live demo uses a five-line list (the same checklist without "unzip the bag", and with the watch line shortened to "put the watch in the bag"). I ran that list on the stored takes, two runs each (`results/demo_list_check.json`). Takes 1 to 4 (all done, or one item skipped): every row right in 8 of 8 runs; the skipped item read not found at 0.04 to 0.25. Takes 6 to 9 (fakes and in-and-out): 6 of 8 runs had every row right; in take 7 the faked item read done once (0.66) and unsure once (0.52). The page shows three words from the fused probability: done at 0.65 or more, not found under 0.30, unsure between. The benchmark tables use the plain 0.5 cut.

One more take, 4 seconds, tests whether a step can name a person by what they wear. The step "shake hands with the person wearing a black jacket" read done at 0.98. On the same video, the same step reworded to a red jacket, to a white shirt, to a fist bump, and to handing over a pen each read not found in 3 of 3 runs. The Decisions ladder alone leaned yes on the fist bump (0.63); the reasoning witnesses said no.

A second handshake take, 5 seconds, with a different person in a white knit sleeve (`results/handshake_check.json`). "shake hands with a person wearing white tshirt" read done at 0.98 in the live run and in 3 of 3 re-runs. On the same video a red tshirt, a green jacket, a black tshirt (black trousers are in the picture), a fist bump and handing over a pen each read not found in 3 of 3 runs; the fist bump was closest at 0.25. Here the Decisions ladder alone said no to the true line every time, and the reasoning witnesses carried it. The garment word matters when near-synonyms compete: with "white shirt", "white top" and "white sweater" on one list, "sweater" read done in 2 of 2 runs, "top" in 1 of 2, and "shirt" in neither (0.49). A blue shirt (blue wristbands are in the picture), an orange shirt and a grey shirt read not found. That is two people and two colors, and nothing more.

A sound take, 8 seconds: clap, hold hands together, snap fingers, knock three times, with "say the word pineapple" left out. The skipped spoken word read not found at 0.09, and the snap and the knock read done. Counts are only checked when the list makes them matter: with "clap twice" as the only clap line, one clap was accepted as done; with "clap once", "clap twice" and "clap three times" all on the list, only "clap once" read done, and "knock on the table once" read not found against three knocks.

### Audit by a Managed Agent (field takes)

The one error class left on my takes is a step whose result was already in place: an open bag read as "unzipped". A frame from the start of the video shows that directly, so I gave that check to a Managed Agent (`clock/auditor.py`, step 6 above) and ran it on the nine field takes: 16 audits with `gemini-4-hydrogen` driving the agent, one fused read each. Records are in `results/audit_field.json`.

| Full read said | Truth | Steps | Audit disputes | Audit confirms | Audit cannot tell |
|-|-|-|-|-|-|
| done | done | 56 | 0 | 12 | 44 |
| done | not done | 5 | 2 | 0 | 3 |
| not found | not done | 32 | 0 | 9 | 23 |
| unsure | either | 3 | 0 | 0 | 3 |

- It never disputed a correct verdict (0 of 88).
- It disputed 2 of the 5 wrong "done" verdicts: "unzip the bag" in both audits of take 8. For the same error in take 9 it said "cannot tell" both times, and it said "cannot tell" for a faked item that one full read credited in take 7.
- Most steps come back "cannot tell". Items inside a bag and a knock leave nothing a still frame can show, and the rules say so instead of guessing.
- All 16 audits returned, all 16 carried a matching sandbox hash, and the local re-run of the rules agreed 16 of 16. Median 52 seconds, longest 67, about 81,000 tokens each at the median.
- With `gemini-4-neon` driving the agent (3 audits) it took 86 to 162 seconds and confirmed one wrong verdict. Not used.

This is 16 audits of nine recordings I made myself, so it is a check that the stage catches the case it was built for when the camera sees it. It is not a benchmark score.

To see how often the audit objects when it should not, I also ran it on the first 30 clips of the fifth test split, one full read and one audit each (`eval/audit_bench.py`, `results/audit_bench_test5.json`):

| Full read said | Truth | Steps | Audit disputes | Audit confirms | Audit cannot tell |
|-|-|-|-|-|-|
| done | done | 77 | 5 | 38 | 34 |
| not found | not done | 55 | 1 | 20 | 34 |
| not found | done | 5 | 0 | 1 | 4 |
| unsure | not done | 5 | 2 | 0 | 3 |
| unsure | done | 6 | 2 | 1 | 3 |

- On benchmark clips the audit does dispute correct verdicts: 6 of 132, about 1 in 20. So "0 of 88" above is a property of my takes, not of the stage.
- All 5 disputed "done" verdicts are actions that were undone before the clip ended: a book opened and closed again, a cloth unfolded and folded back, a backpack opened and zipped shut, a tea box opened and closed, letters shuffled back to the same order. The benchmark labels count those as done. The audit's rule asks for a result that is still in place at the end, and no later step on the list explains the reversal. Whether that is an error depends on the checklist: for "did this item end up in the box" it is the right objection.
- The one disputed "not found" is a pencil case that ends up on top of a notebook in a clip where "dropping something on top of something" was a planted step.
- No full read in these 30 clips credited a planted step, so this run says nothing about how often the audit catches a wrong "done". The field takes are the only evidence for that, and it is 2 of 5.
- On these clips more results are visible in a frame than on the bag takes, so it confirms about half of the correct "done" verdicts. All 30 audits returned with a matching sandbox hash. Median 41 seconds, longest 65, about 62,000 tokens.

## Run it

Requires Python 3.11 and a Gemini API key with access to the Gemini 4 models.

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     (Windows; use .venv/bin/python elsewhere)
echo GEMINI_API_KEY=your_key > .env
.venv/Scripts/python -m uvicorn app.server:app --port 8765
```

Open http://localhost:8765/ to record or upload. Open http://localhost:8765/?watch=1 on a second screen to show results as they arrive. A phone camera needs HTTPS, so for phone recording put the server behind an HTTPS tunnel.

Reproduce the evaluation:

```
python -m eval.pt_data fetch --split tune        (then test, test2: downloads the clips)
python -m eval.run_eval --split tune --system ladder --run 3
python -m eval.run_eval --split tune --system hydrogen4 --name v3   (then hydrogen2, neon4)
python -m eval.fit_fusion                        (fits clock/fusion_params.json on the tuning split)
python -m eval.run_eval --split test4 --system ladder --run 1
python -m eval.run_eval --split test4 --system hydrogen4 --name v3  (then hydrogen2, neon4)
python -m eval.final_report --split test4
python -m eval.audit_field                       (the Managed Agent audit on the field takes; needs the recordings)
```

Per-clip outputs of every run are in `results/`.

## What is in the repository

| Path | What it is |
|-|-|
| `clock/` | The engine: video prep, the question ladder, the Decisions API call, model wrappers, witnesses, fusion, and the Managed Agent auditor. |
| `app/` | FastAPI server and the page. |
| `eval/` | Data loader and split rule, eval runner, parameter fitting, scoring and the final report. |
| `results/` | Reports and per-clip outputs from the runs described above. |
| `probe.py` | The first go or no-go test: does the Decisions model track time inside a video. |
| `video/` | Scripts that build the one-minute demo video from a real run: animated charts, generated voice, assembly. |

## Data and licenses

- DeepMind Perception Test: videos and annotations, CC BY 4.0. Patraucean et al., "Perception Test: A Diagnostic Benchmark for Multimodal Video Models", NeurIPS 2023. https://github.com/google-deepmind/perception_test . The data is downloaded by `eval/pt_data.py` and is not stored in this repository.
- CaptainCook4D: a loader is included (`eval/cc4d.py`) but no result in this README uses it.

Libraries: google-genai (Apache 2.0), FastAPI (MIT), Uvicorn (BSD), NumPy (BSD), scikit-learn (BSD), imageio-ffmpeg (BSD, bundles an ffmpeg binary under LGPL/GPL), remotezip (MIT), python-dotenv (BSD), python-multipart (Apache 2.0), segno (BSD).

## Related work

I found no published work that asks cumulative "had it started by time t" questions per step and fits a monotone curve to get start times with a range. The search was about 17 queries, mostly arXiv, and no patents, so this is a statement about that search. Close work: per-clip yes/no scoring for temporal grounding ("Your VLM Already Knows When", arXiv 2608.08315), prefix "is the task complete" probabilities (TOPReward, arXiv 2602.19313), zero-shot procedural mistake detection (ZeProM, arXiv 2606.21579), and products that check station video against work instructions with trained or tuned models. Combining several model reports with fitted reliabilities follows the Dawid-Skene line of work on noisy annotators.

## Cost

Median input tokens per call on a 20 to 30 second clip, from `runs/calls.jsonl`: Decisions ladder 11,900; Hydrogen at 4 fps 9,300; Hydrogen at 2 fps 5,400; Neon at 4 fps 9,300; each witness adds about 700 to 1,000 output and thinking tokens. At the list prices in the event's builder guide that is about 8 cents for a full read of one clip, and about a tenth of a cent for the first read alone. I did not measure cost on longer videos.

## Limits

- All scores are on benchmark clips of 10 to 35 seconds. On two recordings of 8 and 21 minutes it missed the one skipped step and its timing was 10 to 50 times worse. Nothing here was measured on a real job site.
- It does not prove that a step was skipped. It ranks steps by how likely they are to have been skipped, and a person checks.
- Labels in the benchmark are generic ("Putting something into something"). Typed checklists with specific objects may behave differently.
