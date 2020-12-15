# ITU AI/ML in 5G Challenge (PS-032-KDDI, Japan)
* Theme: Analysis on route information failure in IP core networks by NFV-based test environment.

## Problem statement
You can check the details of the problem statement via [this link](https://www.ieice.org/~rising/AI-5G/#theme1).

### Dataset
The dataset used in this challenge were created in the NFV-based test envrionment simulated for IP core network [Kawasaki+20].
The dataset is provided at [ITU AI/ML in 5G Challenge Global Round in Japan](https://www.ieice.org/~rising/AI-5G/#theme1).

* `data-for-learning`: contains the dataset for training;
* `label-for-learning`: contains the label data for training;
* `data-for-evaluation`: contains the dataset for the evaluation;
* `label-for-evaluation`: contains the label data for the evaluation;

You can download these datasets by the following commands.
```bash
wget https://www.ieice.org/~rising/AI-5G/dataset/theme1-KDDI/data-for-learning.tar.gz
wget https://www.ieice.org/~rising/AI-5G/dataset/theme1-KDDI/label-for-learning.tar.gz
wget https://www.ieice.org/~rising/AI-5G/dataset/theme1-KDDI/data-for-evaluation.tar.gz
wget https://www.ieice.org/~rising/AI-5G/dataset/theme1-KDDI/label-for-evaluation.tar.gz
```

## Brief usage
### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
./retrieve_dataset.sh                 # retrieve dataset
./preprocessing.sh                    # data preprocessing
python graph_classification.py [ARGS] # train and test
[ARGS]
optional arguments:
  -h, --help  show this help message and exit
  --train     is train mode
  --test      is test mode
  --both      is both train and test mode
python plot.py                        # create the demonstration movie
```

## Report
Available [here](https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-032-KDDI-naist-lsm/blob/main/PS-032-KDDI-naist-lsm_report.pdf).

<!-- ### Comparison performance -->
### Brief demonstration
[![Brief demonstration](http://img.youtube.com/vi/HqRSd6vzLb4/0.jpg)](http://www.youtube.com/watch?v=HqRSd6vzLb4)
## References
* [ITU AI/ML in 5G Challenge](https://www.itu.int/en/ITU-T/AI/challenge/2020/Pages/default.aspx)
* [ITU AI/ML in 5G Challenge Global Round in Japan](https://www.ieice.org/~rising/AI-5G/)
* [Kawasaki+20] J. Kawasaki, G. Mouri, and Y. Suzuki, "Comparative Analysis of Network Fault Classification Using Machine Learning," in Proc. of NOMS2020, doi:10.1109/NOMS47738.2020.9110454.
