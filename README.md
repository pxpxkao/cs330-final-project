# Emotion-Cause Pair Extraction with Hybrid Architecture Analysis

#### cs330-final-project 

We construct the hybrid models based on the original state-of-the-art models' githubs.

## RankCP+w2v
* Set up the environment using `RankCP+w2v/environment.yml`
* Run `python src/main.py`

## E2E+RankCP
* Use the same environment as `RankCP+w2v`
* Run `python main.py`
* It will train E2E_PExtE+RankCP model

## E2E+Sliding_window
* Use the same environment as `RankCP+w2v`
* Run `python E2E_sliding.py`
* It will train E2E_PExtE+Sliding Window model


## Reference
* An End-to-End Network for Emotion-Cause Pair Extraction[[link](https://aaditya-singh.github.io/data/ECPE.pdf)], [github](https://github.com/Aaditya-Singh/E2E-ECPE)
* Effective Inter-Clause Modeling for End-to-End Emotion-Cause Pair Extraction[[link](https://aclanthology.org/2020.acl-main.289.pdf)], [github](https://github.com/Determined22/Rank-Emotion-Cause)
* End-to-End Emotion-Cause Pair Extraction based on SlidingWindow Multi-Label Learning[[link](https://aclanthology.org/2020.emnlp-main.290.pdf)], [github](https://github.com/NUSTM/ECPE-MLL)
