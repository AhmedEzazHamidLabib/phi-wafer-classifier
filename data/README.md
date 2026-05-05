# Data

No data files are stored in this repository. Follow the steps below to set up the dataset.

## Step 1 — Download WM-811K

Go to: https://www.kaggle.com/datasets/qingyi/wm-811k-wafer-map

Download the zip file and extract `WM811K.pkl`. Place it in this `data/` folder.

The file is approximately 2GB uncompressed.

## Step 2 — Generate labeled_only.pkl

Run the following once from the project root to generate the filtered dataset used by all training scripts:

```bash
python data/prepare.py
```

Or run inline:

```python
import pandas as pd

df = pd.read_pickle("data/WM811K.pkl")

# failureType column contains mixed types
# string rows are labeled, ndarray rows are unlabeled
str_mask = df['failureType'].apply(lambda x: isinstance(x, str))
df_strings = df[str_mask].copy()

labeled = df_strings[df_strings['failureType'] != 'none'].copy()
labeled.to_pickle("data/labeled_only.pkl")
print(f"Saved {len(labeled)} labeled rows")
```

This produces `labeled_only.pkl` with 25,519 labeled failure examples across 8 classes.

## Dataset Summary

| Class | Count | % of labeled |
|-------|-------|--------------|
| Edge-Ring | 9,680 | 37.9% |
| Edge-Loc | 5,189 | 20.3% |
| Center | 4,294 | 16.8% |
| Loc | 3,593 | 14.1% |
| Scratch | 1,193 | 4.7% |
| Random | 866 | 3.4% |
| Donut | 555 | 2.2% |
| Near-full | 149 | 0.6% |

## Citation

Wang, M.-H. (2018). WM-811K Wafer Map Dataset. MIR Lab, National Taiwan University.
http://mirlab.org/dataSet/public/
