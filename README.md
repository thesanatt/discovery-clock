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
   - `clock/witness.py` asks `gemini-4-hydrogen` (at 1 frame per second), `gemini-4-hydrogen` again (4 frames per second, medium resolution) and `gemini-4-neon` for a timeline: for each step, did it happen and at what second.
3. `clock/fusion.py` combines them. For each step it compares two hypotheses, "happened, starting at time tau" and "never happened". Each witness contributes a likelihood: its hit rate, its false alarm rate, and a Laplace error around the true start time with a small share of gross errors. The ladder contributes a Gaussian score and its own time estimate. The posterior over tau gives the start time and its range. One temperature on the log odds corrects for the fact that the witnesses watch the same video and so make related mistakes.
4. `clock/engine.py` runs all of this with a timeout on every call and a deadline for the whole read. A witness that fails or is late is left out. The Decisions ladder alone is the fallback and is shown as the "first read" while the others are still watching.
5. `app/` is the web page: record on a phone or upload a file, type the checklist, read the step table. Each row shows the verdict, the start time with its range, the posterior curve on a shared time axis, what each witness said, and a 4 second clip.

Model jobs:

| Model | Job |
|-|-|
| `gemini-decision-1` (Decisions API) | The ladder: steps x time points in one call. First read in about a second, and a graded score that the yes/no witnesses lack. |
| `gemini-4-hydrogen` | Two fast timelines of the whole video, at 1 and at 4 frames per second. |
| `gemini-4-neon` | A third timeline from a stronger model. |
| `gemini-4-carbon` | The baseline: one call at high thinking, asked for the same output. Not part of the system. |

## Does it work

### Test data

The DeepMind Perception Test (validation split, CC BY 4.0): short phone videos filmed by crowd participants, with human labels for each action and its start and end time. I did not write or record any of it.

For each clip the checklist is the clip's own labeled actions plus two steps that never happen in it, picked by a hash of the clip id. The system must say which steps happened, when each started, and flag the two planted skips.

Clips were assigned by a fixed rule before any model output existed (`eval/splits.json`): eligible clips sorted by the SHA-1 of their id; the first 30 are for tuning, the next 120 are the test split, the next 100 are a second test split.

### Results on the first test split (120 clips, 626 steps: 386 real, 240 planted skips)

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

### Results on the second test split (99 clips, 506 steps)

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

### The Carbon baseline

One `gemini-4-carbon` call at high thinking, same prompt and output format. It is the most accurate single call on the clips it finished, and too slow to use live: about 110 seconds per clip at the median, and in my runs it timed out often enough that the wrapper fell back to Neon on a large share of clips. See `results/` for the exact counts at the time of the last run.

### Where it fails

- Vague labels. "Moving object(s) around" and "Holding something in a state" account for the largest share of errors on both test splits. These actions arguably happen through most of a clip, so their labeled start is hard to match. Without those two classes the fused system scores skip F1 0.90 and 79 percent within 2 seconds on the second split.
- Steps every witness misses. When all four sources miss a real step the fused probability is near zero, and 13 percent of those steps were real. A person still has to check flagged steps.
- The Decisions ladder on real phone footage. On a first field recording it scored three of five real steps low; the reasoning models carried the verdict.

### Things I tried that did not help (tuning split)

- A zoom pass: Neon re-watches 7 seconds around each estimate at higher resolution. Shown a short clip, it said yes to 44 percent of steps that never happened. Left out.
- 720p input at high media resolution. About the same accuracy at four times the latency.
- Neon at medium thinking. No gain, twice the latency.
- A change-point likelihood over the raw ladder answers. Worse timing than the simple half-rise estimate.
- Shorter question wording for the ladder. Fewer tokens, clearly worse answers.
- A clock burned into the video frames. No better than plain video: the Decisions model already tracks time.

### Field check

Recordings made at the event on a phone, with a known script, are listed in `results/field.md` when present. They are a small extra check on real footage, not part of the scores above.

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
python -m eval.run_eval --split tune --system hydrogen     (then hydrogen4, neon)
python -m eval.fit_fusion                        (fits clock/fusion_params.json on the tuning split)
python -m eval.run_eval --split test --system ladder --run 1   (then hydrogen, hydrogen4, neon, carbon)
python -m eval.final_report --split test
```

Per-clip outputs of every run are in `results/`.

## What is in the repository

| Path | What it is |
|-|-|
| `clock/` | The engine: video prep, the question ladder, the Decisions API call, model wrappers, witnesses, fusion. |
| `app/` | FastAPI server and the page. |
| `eval/` | Data loader and split rule, eval runner, parameter fitting, scoring and the final report. |
| `results/` | Reports and per-clip outputs from the runs described above. |
| `probe.py` | The first go or no-go test: does the Decisions model track time inside a video. |

## Data and licenses

- DeepMind Perception Test: videos and annotations, CC BY 4.0. Patraucean et al., "Perception Test: A Diagnostic Benchmark for Multimodal Video Models", NeurIPS 2023. https://github.com/google-deepmind/perception_test . The data is downloaded by `eval/pt_data.py` and is not stored in this repository.
- CaptainCook4D: a loader is included (`eval/cc4d.py`) but no result in this README uses it.

Libraries: google-genai (Apache 2.0), FastAPI (MIT), Uvicorn (BSD), NumPy (BSD), scikit-learn (BSD), imageio-ffmpeg (BSD, bundles an ffmpeg binary under LGPL/GPL), remotezip (MIT), python-dotenv (BSD), python-multipart (Apache 2.0), segno (BSD).

## Related work

I found no published work that asks cumulative "had it started by time t" questions per step and fits a monotone curve to get start times with a range. The search was about 17 queries, mostly arXiv, and no patents, so this is a statement about that search. Close work: per-clip yes/no scoring for temporal grounding ("Your VLM Already Knows When", arXiv 2608.08315), prefix "is the task complete" probabilities (TOPReward, arXiv 2602.19313), zero-shot procedural mistake detection (ZeProM, arXiv 2606.21579), and products that check station video against work instructions with trained or tuned models. Combining several model reports with fitted reliabilities follows the Dawid-Skene line of work on noisy annotators.

## Limits

- All scores are on benchmark clips of 10 to 35 seconds. Nothing here was measured on a real job site.
- It does not prove that a step was skipped. It ranks steps by how likely they are to have been skipped, and a person checks.
- Labels in the benchmark are generic ("Putting something into something"). Typed checklists with specific objects may behave differently.
