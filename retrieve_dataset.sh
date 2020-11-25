#!/usr/bin/env sh

echo "### Downloading ###"
wget -P https://www.ieice.org/~rising/AI-5G/dataset/theme1-KDDI/data-for-learning.tar.gz && sleep 10
wget -P https://www.ieice.org/~rising/AI-5G/dataset/theme1-KDDI/label-for-learning.tar.gz && sleep 10
wget -P https://www.ieice.org/~rising/AI-5G/dataset/theme1-KDDI/data-for-evaluation.tar.gz && sleep 10
wget -P /tmp https://www.ieice.org/~rising/AI-5G/dataset/theme1-KDDI/label-for-evaluation.tar.gz && sleep 10
echo "### DONE ###"

echo "### EXTRACT ###"
tar zxvf /tmp/data-for-learning.tar.gz -C /tmp
tar zxvf /tmp/label-for-learning.tar.gz -C /tmp
tar zxvf /tmp/data-for-evaluation.tar.gz -C /tmp
tar zxvf /tmp/label-for-evaluation.tar.gz -C /tmp
echo "### DONE ###"
