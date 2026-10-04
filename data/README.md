# Data

The datasets are not redistributed here. Download them from the Canadian Institute for Cybersecurity and place them as follows.

## CIC-IDS2017

Download the machine-learning CSV release (MachineLearningCSV.zip) from https://www.unb.ca/cic/datasets/ids-2017.html and copy one file:

```
data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

## CIC-DDoS2019

Download `CSV-01-12.zip` and `CSV-03-11.zip` from https://www.unb.ca/cic/datasets/ddos-2019.html. Then build the sampled working files. The sampling keeps every benign flow and a seeded sample of attack flows (about 60,000 per attack file):

```
python reproduce/sample_cicddos2019.py <folder containing the two zips>
```

This writes 18 files to `data/raw/CICDDoS2019_sampled/` and a `sampling_log.json` with the per-file sampling rates. The log from the paper's run is included in this repository. Step 2 of `reproduce/run_all.py` then builds `data/ddos2019/all_sampled.pkl`, with 1,262,828 flows.
