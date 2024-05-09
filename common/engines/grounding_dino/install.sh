# pip安装所需包
python3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 下载源码 GroundingDINO
cd source
git clone https://github.com/IDEA-Research/GroundingDINO.git
# 部署安装源码
cd GroundingDINO
apt update && apt install gcc
python3 -m pip install groundingdino-py -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 setup.py build develop
cd ..
# 下载语言模型离线部署版 到 bert-base-uncased
# https://huggingface.co/bert-base-uncased/tree/main
# config.json pytorch_model.bin tokenizer.json tokenizer_config.json vocab.txt
mkdir bert-base-uncased && cd bert-base-uncased
wget -O config.json https://huggingface.co/bert-base-uncased/resolve/main/config.json?download=true
wget -O pytorch_model.bin https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin?download=true
wget -O tokenizer.json https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json?download=true
wget -O tokenizer_config.json https://huggingface.co/bert-base-uncased/resolve/main/tokenizer_config.json?download=true
wget -O vocab.txt https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt?download=true
