## Original Link
```
https://github.com/ZhengL00/ReX-GoT
```

## Setup
- **Environment Setup**
```
conda create -n ReX-GoT python=3.9
conda activate ReX-GoT
pip install transformers==4.11.3
pip install numpy==1.23.3
pip install pandas
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install scikit-learn
pip install sentencepiece
```

## Usage

```
FULL TRAIN:
python main.py

DEBUG MODE:
python main.py --debug

RUNNING MULTIPLE DEBUG RUNS EXAMPLE:
python main.py --debug --mode baseline --name baseline_run1
python main.py --debug --mode rexgot --name rexgot_run1

FULL TRAINING:
python main.py --mode baseline --name baseline_run1
python main.py --mode rexgot --name rexgot_run1

...etc
```

## Output Files

Produces a results_<name>_<date>.txt file in the directory with the data from the run(s)

## BibTeX 

If you find ReX-GoT both interesting and helpful, please consider citing us in your research or publications:

```bibtex
@inproceedings{zheng2024reverse,
  title={Reverse Multi-Choice Dialogue Commonsense Inference with Graph-of-Thought},
  author={Zheng, Li and Fei, Hao and Li, Fei and Li, Bobo and Liao, Lizi and Ji, Donghong and Teng, Chong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 24)},
  year={2024}
}
```



