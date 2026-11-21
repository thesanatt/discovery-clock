"""Version 3 over both of its held-out splits (test4 and test5) pooled.

  python -m eval.pooled_v3

Version 3 was frozen before either split was scored: test4 was scored once when the version was
made, test5 once later as a replication. This prints and saves the same scores over the two
together (results/v3_pooled_report.json). No parameter is fitted here.
"""
from __future__ import annotations

import json
from pathlib import Path

from clock import fusion
from eval import final_report as fr
from eval import fit_fusion

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["test4", "test5"]


def main() -> None:
    params = dict(fusion.load_params(), ladder_mode="summary")
    fused, no_ladder, ladder, single = [], [], [], {n: [] for n in fr.FAST}
    clips = steps = 0
    for split in SPLITS:
        data = fit_fusion.load_all(split, "run1")
        clips += len(data)
        steps += sum(len(c["steps"]) for c in data.values())
        tag = lambda rows: [dict(r, clip=f"{split}:{r['clip']}") for r in rows]  # noqa: E731
        fused += tag(fr.fused_rows(data, params, True, fr.FAST))
        no_ladder += tag(fr.fused_rows(data, params, False, fr.FAST))
        ladder += tag(fr.fused_rows(data, params, True, []))
        for n in fr.FAST:
            single[n] += tag(fr.single_rows(split, n)[0])
    report = {"splits": SPLITS, "clips": clips, "steps": steps,
              "real": sum(r["present"] for r in fused), "planted": sum(not r["present"] for r in fused), "systems": {}}
    names = {"hydrogen4": "Hydrogen 4 fps, one call", "hydrogen2": "Hydrogen 2 fps, one call", "neon4": "Neon 4 fps, one call"}
    systems = [("ladder alone (Decisions API)", ladder, 0.5)] + [(names[n], single[n], None) for n in fr.FAST] + \
              [("fused v3", fused, 0.5), ("v3 witnesses without the ladder", no_ladder, 0.5)]
    print(f"{' + '.join(SPLITS)}: {clips} clips, {steps} steps ({report['real']} real, {report['planted']} planted skips)")
    for label, rows, thr in systems:
        m = fr.more(rows, thr)
        report["systems"][label] = m
        auroc = "n/a" if thr is None else f"{m['auroc']:.3f}"
        print(f"  {label:34s} F1 {m['skip_f1']:.3f}  P {m['skip_precision']:.2f}  R {m['skip_recall']:.2f}  AUROC {auroc}  "
              f"within 1 s {m['within_1s']:.3f}  within 2 s {m['within_2s']:.3f}  median {m['median_abs_err_s']} s")
    report["bootstrap"] = {}
    for n in fr.FAST:
        keep = {r["clip"] for r in single[n]}
        sub = [r for r in fused if r["clip"] in keep]
        out = {k: fr.bootstrap(sub, single[n], 0.5, None, k) for k in ("skip_f1", "within_2s")}
        report["bootstrap"][names[n]] = out
        print(f"  fused minus {names[n]}: F1 {out['skip_f1'][0]:+.3f} [{out['skip_f1'][1]:+.3f}, {out['skip_f1'][2]:+.3f}]  "
              f"within 2 s {out['within_2s'][0]:+.3f} [{out['within_2s'][1]:+.3f}, {out['within_2s'][2]:+.3f}]")
    tab, ece, brier = fr.calibration(fused)
    report["calibration"] = {"bins": tab, "ece": ece, "brier": brier}
    print("  calibration error", ece, "Brier", brier)
    for row in tab:
        print(f"    P in {row['bin']}: {row['n']:4d} steps, mean P {row['mean_p']:.2f}, share that really happened {row['share_real']:.3f}")
    json.dump(report, open(ROOT / "results" / "v3_pooled_report.json", "w", encoding="ascii"), indent=1)


if __name__ == "__main__":
    main()
