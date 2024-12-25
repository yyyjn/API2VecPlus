# API2Vec++

This repository contains the code for the paper:
API2Vec++: Boosting API Sequence Representation for Malware Detection and Classification
In IEEE Transactions on Software Engineering  (TSE 2024).

# Overview
<img width="1360" alt="image" src="https://github.com/user-attachments/assets/6135fdd7-5347-42f0-8425-856819f739a1" />
A multi-process malware and its arbitrary API call sequences. (a) shows its execution logic. (b) shows two sequences of the same malware yet
traced at different epochs. (c) depicts our graph model, which is robust against various sequences.
<img width="1377" alt="image" src="https://github.com/user-attachments/assets/f414972a-3f56-4367-9f3e-5a5fe3196316" />
Overview of API2Vec++. For the sequence in Fig, one TPG and three TAPGs are generated with Graph Model. Then, with Path Generator on
graphs, several paths are generated. These paths are fed into API Embedding to generate embedding for each path. Finally, the DL models learn on these
embeddings for malware detection and classification.



## Usage

### Install 

```sh
git clone git@github.com:yyyjn/API2VecPlus.git
```

**Note**

- Python 3.9 is required.


## Contact
If you have any questions or suggestions, feel free to contact:

- [Junnan Yin](https://github.com/yyyjn/yyyjn.github.io) (yinjn2023@zgclab.edu.cn)

## Citation
If you find this repo useful, please cite our paper.
```bibtex
@article{cui2024api2vec++,
  title={Api2vec++: Boosting api sequence representation for malware detection and classification},
  author={Cui, Lei and Yin, Junnan and Cui, Jiancong and Ji, Yuede and Liu, Peng and Hao, Zhiyu and Yun, Xiaochun},
  journal={IEEE Transactions on Software Engineering},
  year={2024},
  publisher={IEEE}
}
```

Contributions via pull requests are welcome and appreciated.

## Acknowledgements

We would like to thank all the authors of the referenced papers.
