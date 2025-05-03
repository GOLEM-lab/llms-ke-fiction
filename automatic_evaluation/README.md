# Towards Automatic Evaluation

## Overview
* 3‑stage automatic evaluator: **Rule → Fuzzy (RapidFuzz) → Semantic (SBERT)**
* Per‑entity threshold modes: **tuned / lower / balanced**

## Validation
* Story validation set → **κ 0.94, Acc 96 %** (auto ≈ human)

## Observed Gap
* On full corpus auto scores **≈ –10 F1 pts** below manual  
  * Worst on *Mode of Demise* & summary inputs  
  * Cause: thresholds too strict → missed borderline matches

## Evaluation Tables

# Auto P/R/F1 by Category

## Category: Stories

| Model          | Victim F1   | Perpetrator F1   | Mode of Demise F1   | Victim P   | Perpetrator P   | Mode of Demise P   | Victim R   | Perpetrator R   | Mode of Demise R   |
|:---------------|:------------|:-----------------|:--------------------|:-----------|:----------------|:-------------------|:-----------|:----------------|:-------------------|
| deepseek-r170b | 42.6%       | **26.7%**        | 25.3%               | 73.1%      | 47.7%           | 41.9%              | 30.1%      | **18.6%**       | 18.1%              |
| gemma3:27b     | 35.1%       | 19.1%            | 16.3%               | 65.9%      | 39.6%           | 26.7%              | 23.9%      | 12.6%           | 11.7%              |
| llama3170b     | **46.3%**   | 25.5%            | **25.8%**           | 70.3%      | 44.1%           | 38.8%              | **34.5%**  | 18.0%           | **19.3%**          |
| qwen330b-a3b   | 22.1%       | 17.8%            | 19.3%               | **78.4%**  | **51.4%**       | **55.6%**          | 12.8%      | 10.8%           | 11.7%              |

## Category: Specialised Summary

| Model          | Victim F1   | Perpetrator F1   | Mode of Demise F1   | Victim P   | Perpetrator P   | Mode of Demise P   | Victim R   | Perpetrator R   | Mode of Demise R   |
|:---------------|:------------|:-----------------|:--------------------|:-----------|:----------------|:-------------------|:-----------|:----------------|:-------------------|
| deepseek-r170b | **47.0%**   | 29.0%            | **27.9%**           | 71.8%      | 44.4%           | 37.6%              | **35.0%**  | 21.6%           | **22.2%**          |
| llama3170b     | 46.8%       | **32.8%**        | 23.8%               | **72.9%**  | **54.9%**       | **43.8%**          | 34.5%      | **23.4%**       | 16.4%              |

## Category: Generic Summary

| Model          | Victim F1   | Perpetrator F1   | Mode of Demise F1   | Victim P   | Perpetrator P   | Mode of Demise P   | Victim R   | Perpetrator R   | Mode of Demise R   |
|:---------------|:------------|:-----------------|:--------------------|:-----------|:----------------|:-------------------|:-----------|:----------------|:-------------------|
| deepseek-r170b | 30.6%       | 18.4%            | **24.2%**           | 66.2%      | 40.0%           | **46.7%**          | 19.9%      | 12.0%           | **16.4%**          |
| llama3170b     | **33.1%**   | **24.3%**        | 21.5%               | **70.0%**  | **55.3%**       | 46.2%              | **21.7%**  | **15.6%**       | 14.0%              |



## Next Steps
* **RAG** — provide relevant story chunks during extraction  
* **Traditional KE pipelines** — high setup cost; often needs training/fine‑tuning  
* **Run same experiments on conventional datasets**
  * ACE 2005 *(LDC2006T06, paywalled)*
  * GunViolenceCorpus *(annotations largely off‑task)*
