# DNLP SS24 Final Project

-   **Group name: SPEZI!**

-   **Group code:**

-   **Group repository: https://github.com/JonasA1998/NLP_G01**

-   **Tutor:** Finn Schmidt

-   **Group team leader:** Jonas Altendorf

-   **Group members:**
  * Matheus Coutinho da Silva
  * Bjarne Meyer
  * Jannis Rowold
  * Hanna Groß

# Introduction

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)

[![Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

[![Final](https://img.shields.io/badge/Status-Final-purple.svg)](https://https://img.shields.io/badge/Status-Final-blue.svg)

[![Black Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://black.readthedocs.io/en/stable/)

[![AI-Usage Card](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./AI-Usage-Card.pdf/)

**TODO: adapt to our project**

This repository contains our implementation of the DNLP Project which consists of Multitask BERT subproject and Paraphrase-Type-Task BART subproject for the Deep Learning for Natural Language Processing course (SoSe2024) at the University of Göttingen.

The first part of the project includes the completion of minBERT and the Adam Optimizer. Furthermore the project includes the baseline implementations for fine-tuning the pre-trained BERT-model on three tasks (sentiment analysis, question similarity, semantic similarity).
The second part of the project is the improvement of all five baselines.


# Setup instructions
  Explain how we can run your code in this section. We should be able to reproduce the results you've obtained.

  In addition, if you used libraries that were not included in the conda environment 'dnlp' explain the exact installation instructions or provide a ```.sh``` file for the installation.

  Which files do we have to execute to train/evaluate your models? Write down the command which you used to execute the experiments. We should be able to reproduce the experiments/results.

  _Hint_: At the end of the project you can set up a new environment and follow your setup instructions making sure they are sufficient and if you can reproduce your results.


## OLd
This is the starting code for the default final project for the Deep Learning for Natural Language Processing course at the University of Göttingen. You can find the handout [here](https://docs.google.com/document/d/1pZiPDbcUVhU9ODeMUI_lXZKQWSsxr7GO/edit?usp=sharing&ouid=112211987267179322743&rtpof=true&sd=true)

In this project, you will implement some important components of the BERT model to better understanding its architecture.
You will then use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection and semantic similarity.

After finishing the BERT implementation, you will have a simple model that simultaneously performs the three tasks.
You will then implement extensions to improve on top of this baseline.

## Setup instructions

* Follow `setup.sh` to properly setup a conda environment and install dependencies.
* There is a detailed description of the code structure in [STRUCTURE.md](./STRUCTURE.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh` (Use `setup_gwdg.sh` if you are using the GWDG clusters).
* Libraries that give you other pre-trained models or embeddings are not allowed (e.g., `transformers`).
* Use this template to create your README file of your repository: <https://github.com/gipplab/dnlp_readme_template>

## Project Description

Please refer to the project description for a through explanation of the project and its parts.

### Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

For the 2024 edition of the DNLP course at the University of Göttingen, the project was modified by [Niklas Bauer](https://github.com/ItsNiklas/), [Jonas Lührs](https://github.com/JonasLuehrs), ...
