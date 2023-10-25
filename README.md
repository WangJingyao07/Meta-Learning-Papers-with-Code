# Meta-Learning-Papers-with-Code

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/WangJingyao07/Meta-Learning-Papers-with-Code)

This repository contains a reading list of papers with code on **Meta-Learning** and **Meta-Reinforcement-Learning**, These papers are mainly categorized according to the type of model. In addition, I will separately list papers from important conferences starting from 2023, e.g., NIPS, ICML, ICLR, CVPR etc. **This repository is still being continuously improved. If you have found any relevant papers that need to be included in this repository, please feel free to submit a pull request (PR) or open an issue.**

Each paper may be applicable to one or more types of meta-learning frameworks, including optimization-based and metric-based, and may be applicable to multiple data sources, including image, text, audio, video, and multi-modality. **These are marked in the type column**. In addition, for different tasks and different problems, **we have marked the SOTA algorithm separately**. This is submitted with reference to the leadboard at the time of submission, and will be continuously modified. **We provide a basic introduction to each paper to help you understand the work and core ideas of this article more quickly**.

### Label

🎭 **Different Frameworks**

* ![Meta-Learning](https://img.shields.io/badge/-ML-gray) Meta-Learning.
* ![Meta-Reinforcement-Learning](https://img.shields.io/badge/-MRL-white)  Meta-Reinforcement-Learning.

🎨 **Different Types**

* ![optimization-based](https://img.shields.io/badge/-Optimization-blue) Optimization-based meta-learning approaches acquire a collection of optimal initial parameters, facilitating rapid convergence of a model when adapting to novel tasks.
* ![metric-based](https://img.shields.io/badge/-Metric-red)  Metric-based meta-learning approaches acquire embedding functions that transform instances from various tasks, allowing them to be readily categorized using non-parametric methods.

✨ **Different Data Sources**

* ![Image](https://img.shields.io/badge/-CVimage-brightgreen) Meta-Learning for CV (Images)
* ![Video](https://img.shields.io/badge/-CVvideo-green) Meta-Learning for CV (Videos)
* ![Text](https://img.shields.io/badge/-NLP-pink)  Meta-Learning for NLP
* ![Audio](https://img.shields.io/badge/-Audio-orange)  Meta-Learning for Audio
* ![Multi](https://img.shields.io/badge/-MultiModal-purple)  Meta-Learning for Multi-modal

It is worth noting that the experiments of some frameworks consist of multiple data sources. Our annotations are based on the paper description.

🎁 **Notice**

* ![❗CODE](https://img.shields.io/badge/-❗CODE-yellow) The paper does not provide code, I will write it myself and supplement it later.

🚩 **I have marked some recommended papers with 🌟/🎈 (SOTA methods/Just my personal preference 😉).**

🚩 **I will maintain three hours of paper reading, code repository maintenance and entry supplement every day 😉).**

## Topics

* [Survey](#Survey)
* [Optimization](#Optimization)
* [Theory](#Theory)
* [Domain generalization](#Domain-generalization)
* [Lifelong learning](#Lifelong-learning)
* [Configuration transfer](#Configuration-transfer)
* [Model compression](#Model-compression)
* [Summary of conference papers](#Summary-of-conference-papers)
  * [CVPR23](#CVPR23)
  * [ICML23](#ICML23)
  * [ICCV23](#ICCV23)
  * [NIPS23](#NIPS23)
  * [ICLR23](#ICLR23)


## Survey.

| Date | Method                                                       | Type                                                    | Conference                             | Paper Title and Paper Interpretation            | Code |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------ | ---- |
| 2018 | [RL L2L](https://arxiv.org/abs/1812.07995) | ![Meta-Reinforcement-Learning](https://img.shields.io/badge/-MRL-white) | arXiv 2018                                   | A review of meta-reinforcement learning for deep neural networks architecture search                   | None |
| 2019 | [Book of Meta-Learning](https://library.oapen.org/bitstream/handle/20.500.12657/23012/1/1007149.pdf#page=46) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) | Book                                   | Meta-Learning (Automated Machine Learning)                   | None |
| 2019 | [Learn dynamics](https://arxiv.org/abs/1905.01320)           | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) | arXiv 2019                             | Meta-learners' learning dynamics are unlike learners'        | None |
| 2020 | [NLP🌟](https://arxiv.org/abs/2007.09604)                      | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) | arXiv 2020                             | Meta-learning for few-shot natural language processing: A survey | None |
| 2020 | [CV-classifier](https://ieeexplore.ieee.org/abstract/document/8951014) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) | IEEE Access                            | A literature survey and empirical study of meta-learning for classifier selection | None |
| 2020 | [RL DL L2L ](https://arxiv.org/abs/2004.11149) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![Meta-Reinforcement-Learning](https://img.shields.io/badge/-MRL-white) | arXiv 2020                           | A comprehensive overview and survey of recent advances in meta-learning | None |
| 2021 | [Learn 2 Learn](https://library.oapen.org/bitstream/handle/20.500.12657/23012/1/1007149.pdf#page=46) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) | arXiv 2021                             | Meta-Learning: A Survey                                      | None |
| 2021 | [Learn 2 Learn 🎈](https://arxiv.org/abs/2004.05439)          | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) | TPAMI                                  | Meta-Learning in Neural Networks: A Survey                   | None |
| 2021 | [Learn 2 Learn](https://arxiv.org/abs/2004.05439)            | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) | Artif Intell Rev                       | A survey of deep meta-learning                               | None |
| 2021 | [Learn 2 Learn](https://www.sciencedirect.com/science/article/abs/pii/S2352154621000024) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) | Current Opinion in Behavioral Sciences | Meta-learning in natural and artificial intelligence         | None |
| 2022 | [Multi-Modal🌟](https://arxiv.org/abs/2004.05439)              | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) | KBS                                    | Multimodality in meta-learning: A comprehensive survey       | None |
| 2022 | [Image Segmentation🌟](https://arxiv.org/abs/2004.05439)       | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) | PR                                     | Meta-seg: A survey of meta-learning for image segmentation   | None |
| 2022 | [Cyberspace Security](https://www.sciencedirect.com/science/article/pii/S2352864822000281) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) | Digit. Commun. Netw.                   | Application of meta-learning in cyberspace security: A survey | None |
| 2023 | [RL L2L🌟](https://arxiv.org/abs/2301.08028) | ![Meta-Reinforcement-Learning](https://img.shields.io/badge/-MRL-white) | arXiv 2023                   | A survey of meta-reinforcement learning | None |


## Optimization
| Date | Method                                                       | Type                                                    | Conference                             | Paper Title and Paper Interpretation            | Code |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------ | ---- |
| 2016 | [Reversible](https://arxiv.org/pdf/1502.03492.pdf) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) ![Image](https://img.shields.io/badge/-CVimage-brightgreen) | ICML 2016                                   | Gradient-based Hyperparameter Optimization through Reversible Learning      | [CODE](https://github.com/HIPS/hypergrad) |
| 2017 | [MRL-GPS](https://arxiv.org/abs/1606.01885) | ![Meta-Reinforcement-Learning](https://img.shields.io/badge/-MRL-white) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) | ICLR 2017 | Learning to Optimize | ![❗CODE](https://img.shields.io/badge/-❗CODE-yellow) |
| 2019 | [L2G](https://arxiv.org/pdf/1908.01457.pdf) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![metric-based](https://img.shields.io/badge/-Metric-red) ![Image](https://img.shields.io/badge/-CVimage-brightgreen) | arXiv 2019                                   | Learning to Generalize to Unseen Tasks with Bilevel Optimization   | ![❗CODE](https://img.shields.io/badge/-❗CODE-yellow) |
| 2019 | [LOIS](https://arxiv.org/pdf/1911.03787.pdf) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) | arXiv 2019      | Learning to Optimize in Swarms   | [CODE](https://paperswithcode.com/paper/learning-to-optimize-in-swarms) |
| 2019 | [iMAML🌟](https://paperswithcode.com/paper/meta-learning-with-implicit-gradients) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) | NIPS 2019  | Meta-Learning with Implicit Gradients   | [CODE](https://paperswithcode.com/paper/meta-learning-with-implicit-gradients) |
| 2019 | [Xfer🌟](https://paperswithcode.com/paper/transferring-knowledge-across-learning) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) | ICLR 2019  | Transferring Knowledge across Learning Processes  | [CODE](https://paperswithcode.com/paper/transferring-knowledge-across-learning) |
| 2019 | [MetaInit](https://paperswithcode.com/paper/metainit-initializing-learning-by-learning-to) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) | ICLR 2019  | MetaInit: Initializing learning by learning to initialize | ![❗CODE](https://img.shields.io/badge/-❗CODE-yellow) |
| 2019 | [Runge-Kutta-MAML](https://paperswithcode.com/paper/model-agnostic-meta-learning-using-runge) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) | arXiv 2019  | MetaInit: Initializing learning by learning to initialize | ![❗CODE](https://img.shields.io/badge/-❗CODE-yellow) |
| 2020 | [WarpGrad](https://paperswithcode.com/paper/meta-learning-with-warped-gradient-descent) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) | ICLR 2020  | Model-Agnostic Meta-Learning using Runge-Kutta Methods  | [CODE](https://paperswithcode.com/paper/meta-learning-with-warped-gradient-descent) |
| 2022 | [Sharp-MAML🎈](https://arxiv.org/abs/2206.03996) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) ![Image](https://img.shields.io/badge/-CVimage-brightgreen) | ICML 2022                                   | Sharp-MAML: Sharpness-Aware Model-Agnostic Meta Learning                 | [CODE](https://github.com/mominabbass/sharp-maml) |
| 2022 | [BMG🌟](https://openreview.net/forum?id=b-ny3x071E5) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) ![Image](https://img.shields.io/badge/-CVimage-brightgreen) | ICLR 2022 | Bootstrapped Meta-Learning | ![❗CODE](https://img.shields.io/badge/-❗CODE-yellow) |


## Theory
| Date | Method                                                       | Type                                                    | Conference                             | Paper Title and Paper Interpretation            | Code |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------ | ---- |
| 2018 | [MLAP](https://arxiv.org/pdf/1711.01244.pdf) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) | ICML 2018    | Meta-learning by adjusting priors based on extended PAC-Bayes theory      | [CODE](https://github.com/ron-amit/meta-learning-adjusting-priors) |
| 2018 | [learning algorithm approximation ](https://arxiv.org/pdf/1710.11622.pdf) |  ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) | ICLR 2018 | Meta-Learning and Universality: Deep Representations and Gradient Descent can Approximate any Learning Algorithm | ![❗CODE](https://img.shields.io/badge/-❗CODE-yellow) |
| 2018 | [ConsiderMRL](https://paperswithcode.com/paper/some-considerations-on-learning-to-explore#code) |  ![Meta-Reinforcement-Learning](https://img.shields.io/badge/-MRL-white) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) | ICLR 2018 | Some Considerations on Learning to Explore via Meta-Reinforcement Learning | [CODE](https://paperswithcode.com/paper/some-considerations-on-learning-to-explore#code) |

## Domain generalization
| Date | Method                                                       | Type                                                    | Conference                             | Paper Title and Paper Interpretation            | Code |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------ | ---- |
| 2018 | [L2G](https://arxiv.org/pdf/1711.01244.pdf) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![Meta-Reinforcement-Learning](https://img.shields.io/badge/-MRL-white) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) | AAAI 2018    | Learning to Generalize: Meta-Learning for Domain Generalization  | [CODE](https://paperswithcode.com/paper/learning-to-generalize-meta-learning-for) |
| 2019 | [MASF](https://paperswithcode.com/paper/domain-generalization-via-model-agnostic) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) ![metric-based](https://img.shields.io/badge/-Metric-red) | NIPS 2019    | Domain Generalization via Model-Agnostic Learning of Semantic Features  | [CODE](https://paperswithcode.com/paper/domain-generalization-via-model-agnostic) |
| 2020 | [MLCA](https://paperswithcode.com/paper/meta-learning-curiosity-algorithms-1) | ![Meta-Reinforcement-Learning](https://img.shields.io/badge/-MRL-white) | ICLR 2020    | Meta-learning curiosity algorithms      | [CODE](https://paperswithcode.com/paper/meta-learning-curiosity-algorithms-1) |


## Lifelong learning
| Date | Method                                                       | Type                                                    | Conference                             | Paper Title and Paper Interpretation            | Code |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------ | ---- |
| 2018 | [IL2L](https://paperswithcode.com/paper/incremental-learning-to-learn-with) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) | arXiv 2018 | Incremental Learning-to-Learn with Statistical Guarantees  | ![❗CODE](https://img.shields.io/badge/-❗CODE-yellow) |
| 2019 | [VividNet](https://paperswithcode.com/paper/a-neural-symbolic-architecture-for-inverse) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![Graph](https://img.shields.io/badge/-Graph-blue) | arXiv 2019 | A Neural-Symbolic Architecture for Inverse Graphics Improved by Lifelong Meta-Learning  | [CODE](https://paperswithcode.com/paper/a-neural-symbolic-architecture-for-inverse) |
| 2019 | [HSML](https://paperswithcode.com/paper/hierarchically-structured-meta-learning) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) ![Image](https://img.shields.io/badge/-CVimage-brightgreen) ![Text](https://img.shields.io/badge/-NLP-pink) | ICML 2019 | Hierarchically Structured Meta-learning  | [CODE](https://paperswithcode.com/paper/hierarchically-structured-meta-learning) |
| 2019 | [Online-ML](https://paperswithcode.com/paper/online-meta-learning) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray) ![optimization-based](https://img.shields.io/badge/-Optimization-blue) | ICML 2019 | Online Meta-Learning  | ![❗CODE](https://img.shields.io/badge/-❗CODE-yellow) |



## Configuration transfer

## Model compression


## Summary of conference papers

### CVPR23

### ICML23

### ICCV23

### NIPS23

### ICLR23








