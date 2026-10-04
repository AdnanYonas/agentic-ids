# Beyond Flow-Level Accuracy: Explainable Incident-Level DDoS Detection

Code and results for the article *"Beyond Flow-Level Accuracy: Explainable Incident-Level DDoS Detection with Synthetic Incident Augmentation"* by Adnan Younas (Bahria University) and Habiba Adnan (Virtual University of Pakistan). The article is under review.

Machine-learning detectors classify DDoS flows almost perfectly on public benchmarks, but analysts act on incidents, not flows. This repository implements a two-agent pipeline that turns flow-level alerts into explainable incident-level decisions:

- **Monitoring Agent.** A residual MLP with temperature scaling. It scores every flow and emits confidence-tiered alerts.
- **Correlation Agent.** A standardised, class-balanced logistic regression on ten aggregate features. It groups alerts by service and time window and decides whether each group is a genuine attack.
- **Composition-controlled synthetic incidents.** These resample real flows together with their out-of-fold detector scores, to create hard positives (low-rate attacks) and hard negatives (benign bursts, mixed traffic).
- **Incident-level SHAP.** Exact linear SHAP explains every escalation decision.

Every incident is built from out-of-fold detector scores. Evaluation uses blocked and leave-one-attack-out folds, a cross-day test and cross-dataset transfer.

## Main results

All results in `results_local/` and `results_ddos2019_local/` were produced by `reproduce/run_all.py` on the authors' desktop: an Intel Core i5 12th gen with 16 GB of RAM, CPU only, taking about one hour in total. The run logs are in `reproduce/logs/`.

| Experiment | Result |
|---|---|
| CIC-IDS2017, flow level, 5 seeds | Monitoring Agent F1 0.9995 ± 0.0001; LightGBM 0.9999 |
| CIC-IDS2017, incident level | 128,223 alerts → 147–159 incidents; F1 0.986 (real-only agent) vs 0.676 (escalate all) |
| CIC-IDS2017, synthetic low-rate attacks | detection 41% → 95% with augmentation |
| CIC-DDoS2019, 11 unseen attack types | false incidents 30 → 11 (−63%), incident recall 95.4% |
| CIC-DDoS2019, new capture day (Portmap unseen) | false incidents 25 → 0, incident recall 98.9% |
| Transfer CIC-IDS2017 → CIC-DDoS2019 | 0 false incidents on both days |

Synthetic augmentation is a trade-off. It makes the agent robust to rare incident shapes, but it also escalates more isolated single-alert groups. The article discusses this in Section 6.

## Reproducing the results

```bash
python -m venv env
env\Scripts\activate          # Linux/macOS: source env/bin/activate
pip install -r requirements.txt
```

Place the datasets as described in [`data/README.md`](data/README.md), then run the steps in order:

```bash
python reproduce/run_all.py step1    # check environment
python reproduce/run_all.py step2    # build the CIC-DDoS2019 working set
python reproduce/run_all.py step3    # E2: CIC-IDS2017 blocked CV, out-of-fold scores
python reproduce/run_all.py step4    # E1: CIC-IDS2017, 5 random seeds
python reproduce/run_all.py step5    # E3: CIC-IDS2017 incidents, LOTO, window sensitivity, SHAP
python reproduce/run_all.py step6    # E4/E5: CIC-DDoS2019 flow level
python reproduce/run_all.py step7    # E4/E5/E6: CIC-DDoS2019 incidents, cross-dataset transfer
```

Each step writes a log to `reproduce/logs/`. Results go to `results_local/` (CIC-IDS2017) and `results_ddos2019_local/` (CIC-DDoS2019). LightGBM and the logistic-regression baselines are deterministic. The per-flow CIC-DDoS2019 scores were written as pickle files in the reported run. Here they are stored as Parquet files with identical values, to keep the repository small, and the scripts read and write Parquet. The neural Monitoring Agent can differ slightly across hardware and library versions, and so can the incident-level numbers built on its scores.

## Repository structure

```
reproduce/                 experiment code and run_all.py driver
  common.py                data loading, splits, Monitoring Agent, metrics
  flow_experiments.py      E1 (F1) and E2 (F2)
  incident_experiments.py  E3: candidates, synthetic generator, Correlation Agent, SHAP
  extra_incident.py        leave-one-type-out and alert accounting
  ddos2019_flow.py         E4 (L1), E5 (X1) and the seen-type reference (S1)
  ddos2019_incidents.py    E4/E5 incident level, window sensitivity
  ddos2019_summary.py      per-attack-type table
  crossdataset.py          E6: cross-dataset transfer
  sample_cicddos2019.py    builds the sampled CIC-DDoS2019 files
  logs/                    logs of the reported run
results_local/             CIC-IDS2017 results, incl. out-of-fold flow scores (oof_flow_probs.parquet)
results_ddos2019_local/    CIC-DDoS2019 results, incl. out-of-fold scores (oof_L1, oof_S1, pred_X1 as Parquet)
data/                      dataset instructions and the CIC-DDoS2019 sampling log
```

## Citation

If you use this code, please cite the article (details in [`CITATION.cff`](CITATION.cff); full reference to be added on publication).

## License

MIT. See [`LICENSE`](LICENSE). CIC-IDS2017 and CIC-DDoS2019 are subject to the terms of the Canadian Institute for Cybersecurity.
