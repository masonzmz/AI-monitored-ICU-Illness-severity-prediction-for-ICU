# AI‑monitored‑ICU‑Illness‑severity‑prediction‑for‑ICU
Predicting a patient's SOFA score for the next 24 or 48 hours

## MSNETs -> RNN / GRU / LSTM
![structure](img/structure.png)
# Set-up
## Operation System:
![macOS Badge](https://img.shields.io/badge/-macOS-white?style=flat-square&logo=macOS&logoColor=000000) ![Linux Badge](https://img.shields.io/badge/-Linux-white?style=flat-square&logo=Linux&logoColor=FCC624) ![Ubuntu Badge](https://img.shields.io/badge/-Ubuntu-white?style=flat-square&logo=Ubuntu&logoColor=E95420)

## Language and Additional Packages:
![Python](http://img.shields.io/badge/-3.8.13-eee?style=flat&logo=Python&logoColor=3776AB&label=Python) ![PyTorch](http://img.shields.io/badge/-1.12.0-eee?style=flat&logo=pytorch&logoColor=EE4C2C&label=PyTorch) ![Scikit-learn](http://img.shields.io/badge/-1.1.1-eee?style=flat&logo=scikit-learn&logoColor=e26d00&label=Scikit-Learn) ![NumPy](http://img.shields.io/badge/-1.22.3-eee?style=flat&logo=NumPy&logoColor=013243&label=NumPy) ![tqdm](http://img.shields.io/badge/-4.64.0-eee?style=flat&logo=tqdm&logoColor=FFC107&label=tqdm) ![pandas](http://img.shields.io/badge/-1.4.3-eee?style=flat&logo=pandas&logoColor=150458&label=pandas) ![cudatoolkit](http://img.shields.io/badge/-11.6.0-eee?style=flat&label=cudatoolkit) ![datasets](http://img.shields.io/badge/-2.4.0-eee?style=flat&label=datasets) ![matplotlib](http://img.shields.io/badge/-3.4.2-eee?style=flat&label=matplotlib)

## GPU:

![Nvidia](http://img.shields.io/badge/-RTX_A6000_48GB-eee?style=flat&logo=NVIDIA&logoColor=76B900&label=NVIDIA)

## Environment
```console
username@localhost:~$ conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
username@localhost:~$ pip install -U scikit-learn
username@localhost:~$ pip install numpy
username@localhost:~$ pip install pandas
username@localhost:~$ pip install tqdm
```

# Quick Start

```console
username@localhost:~$ python /src/run_training.py
```
