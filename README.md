# Trends in LLM inference efficiency on downstream benchmarks

Code for the data insight ["LLM inference prices have fallen rapidly but unequally across tasks"](https://epoch.ai/data-insights/llm-inference-price-trends?insight-option=All+benchmarks)

You can install required packages using

```
pip install -r requirements.txt
```

Results can be reproduced by running

- `llm_price_trends.ipynb`, for the main analysis.
- `llm_evaluation_cost_trends.ipynb`, for the evaluation cost analysis.

The default results are already stored in CSV format in the `results/default/` folder.
The notebooks specify a `results_dir` near the top, where new results will be saved.

The raw data used for analysis is in the `data/` folder.
