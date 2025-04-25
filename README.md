[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![codecov][codecov-shield]][codecov-url]

![Banner](assets/ffossils-logo-text.png)

![lifecycle](https://img.shields.io/badge/lifecycle-active-green.svg)

# **Neotoma Article Relevance Tool (NeotomaART): Finding Fossils in the Literature**

This project is forked from the larger Neotoma [Meta-Review project](https://github.com/NeotomaDB/metareview), as a stand-alone relevance ML project.

NeotomaART aims to extract identify research articles which are relevant to the [_Neotoma Paleoecological Database_](http://www.neotomadb.org) (Neotoma), and extract article metadata (title, journal, contributing authors) to pass that information to relevant data stewards at Neotoma. This will allow Neotoma to solicit data submissions from a broader range of authors, and, potentially, reduce spatial and disciplinary biases in datasets.

Significant work on this project was performed as part of the _University of British Columbia (UBC)_ [_Masters of Data Science (MDS)_](https://masterdatascience.ubc.ca/) program in partnership with the [_Neotoma Paleoecological Database_](http://neotomadb.org).

**Table of Contents**

- [**Neotoma Article Relevance Tool (NeotomaART): Finding Fossils in the Literature**](#neotoma-article-relevance-tool-neotomaart-finding-fossils-in-the-literature)
  - [**About**](#about)
    - [**Article Relevance Prediction**](#article-relevance-prediction)
  - [How to use this repository](#how-to-use-this-repository)
    - [**Article Relevance**](#article-relevance)
    - [**Data Requirements**](#data-requirements)
      - [**Article Relevance Prediction**](#article-relevance-prediction-1)
    - [**System Requirements**](#system-requirements)
  - [**Directory Structure and Description**](#directory-structure-and-description)
  - [**Contributors**](#contributors)
    - [Tips for Contributing](#tips-for-contributing)

There are 3 primary components to this project:

1. **Article Relevance Prediction** - get the latest articles published, predict which ones are relevant to Neotoma and submit for processing.

<p align="center">
   <img src="assets/project-flow-diagram.png"  width="800">  
</p>

## **About**

Information on each component is outlined below.

### **Article Relevance Prediction**

The goal of this component is to monitor and identify new articles that are relevant to Neotoma. This is done by using the public [xDD API](https://geodeepdive.org/) to regularly get recently published articles. Article metadata is queried from the [CrossRef API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) to obtain data such as journal name, title, abstract and more. The article metadata is then used to predict whether the article is relevant to Neotoma or not.

The model was trained on ~900 positive examples (a sample of articles currently contributing to Neotoma) and ~3500 negative examples (a sample of articles unrrelated or closely related to Neotoma). Logistic regression model was chosen for its outstanding performance and interpretability.

Articles predicted to be relevant will then be submitted to the Data Extraction Pipeline for processing.

<p align="center">
   <img src="assets/article_prediction_flow.png"  width="800">  
</p>

To run the Docker image for article relevance prediction pipeline, please refer to the instructions [here](docker/article-relevance/README.md)

The model could be retrained using reviewed article data. Please refer to [here](docker/article-relevance-retrain/README.md) for the instructions.

## How to use this repository

First, begin by installing the requirements.

For pip:

```bash
pip install -r requirements.txt
```

For conda:
```bash
conda env create -f environment.yml
```

### **Article Relevance**

Please refer to the project wiki for the development and analysis workflow details: [article-relevance Wiki](https://github.com/NeotomaDB/article-relevance/wiki)

### **Data Requirements**

Each of the components of this project have different data requirements. The data requirements for each component are outlined below.

#### **Article Relevance Prediction**

The article relevance prediction component requires a list of journals that are relevant to Neotoma. This dataset used to train and develop the model is available for download [HERE](https://drive.google.com/drive/folders/1NpOO7vSnVY0Wi0rvkuwNiSo3sqq-5AkY?usp=sharing). Download all files and extract the contents into `article-relevance/data/article-relevance/raw/`.

The prediction pipeline requires the trained model object. The model is available [HERE](https://drive.google.com/drive/folders/1NpOO7vSnVY0Wi0rvkuwNiSo3sqq-5AkY?usp=sharing). Download the model file and put the .joblib file in `article-relevance/models/article-relevance/`.

### **System Requirements**

The project has been developed and tested on the following system:

- macOS Monterey 12.5.1
- Windows 11 Pro Version: 22H2
- Ubuntu 22.04.2 LTS

The pre-built Docker images were built using Docker version 4.20.0 but should work with any version of Docker since 4.

## **Directory Structure and Description**

```
├── .github/                            <- Directory for GitHub files
│   ├── workflows/                      <- Directory for workflows
├── assets/                             <- Directory for assets
├── data/                               <- Directory for data
│   ├── article-relevance/              <- Directory for data related to article relevance prediction
│   │   ├── raw/                        <- Raw unprocessed data
│   │   ├── processed/                  <- Processed data
│   │   └── interim/                    <- Temporary data location
├── results/                            <- Directory for results
│   ├── article-relevance/              <- Directory for results related to article relevance prediction
│   ├── ner/                            <- Directory for results related to named entity recognition
│   └── data-review-tool/               <- Directory for results related to data review tool
├── models/                             <- Directory for models
│   ├── article-relevance/              <- Directory for article relevance prediction models
├── notebooks/                          <- Directory for notebooks
├── src/                                <- Directory for source code
│   ├── entity_extraction/              <- Directory for named entity recognition code
│   ├── article_relevance/              <- Directory for article relevance prediction code
│   └── data_review_tool/               <- Directory for data review tool code
├── reports/                            <- Directory for reports
├── tests/                              <- Directory for tests
├── Makefile                            <- Makefile with commands to perform analysis
└── README.md                           <- The top-level README for developers using this project.
```

## **Contributors**

This project is an open project, and contributions are welcome from any individual. All contributors to this project are bound by a [code of conduct](https://github.com/NeotomaDB/article-relevance/blob/main/CODE_OF_CONDUCT.md). Please review and follow this code of conduct as part of your contribution.

The UBC MDS project team consists of:

- [![ORCID](https://img.shields.io/badge/orcid-0009--0003--0699--5838-brightgreen.svg)](https://orcid.org/0009-0003-0699-5838) [Ty Andrews](http://www.ty-andrews.com)
- [![ORCID](https://img.shields.io/badge/orcid-0009--0004--2508--4746-brightgreen.svg)](https://orcid.org/0009-0004-2508-4746) Kelly Wu
- [![ORCID](https://img.shields.io/badge/orcid-0009--0007--1998--3392-brightgreen.svg)](https://orcid.org/0009-0007-1998-3392) Shaun Hutchinson
- [![ORCID](https://img.shields.io/badge/orcid-0009--0007--8913--2403-brightgreen.svg)](https://orcid.org/0009-0007-8913-2403) [Jenit Jain](https://www.linkedin.com/in/jenit-jain-0b31b0160/)

Sponsors from Neotoma supporting the project are:

- [![ORCID](https://img.shields.io/badge/orcid-0000--0002--7926--4935-brightgreen.svg)](https://orcid.org/0000-0002-7926-4935) [Socorro Dominguez Vidana](https://ht-data.com/)
- [![ORCID](https://img.shields.io/badge/orcid-0000--0002--2700--4605-brightgreen.svg)](https://orcid.org/0000-0002-2700-4605) [Simon Goring](http://www.goring.org)

### Tips for Contributing

Issues and bug reports are always welcome. Code clean-up, and feature additions can be done either through pull requests to [project forks](https://github.com/NeotomaDB/article-relevance/network/members) or [project branches](https://github.com/NeotomaDB/article-relevance/branches).

All products of the Neotoma Paleoecology Database are licensed under an [MIT License](LICENSE) unless otherwise noted.

[contributors-shield]: https://img.shields.io/github/contributors/NeotomaDB/article-relevance.svg?style=for-the-badge
[contributors-url]: https://github.com/NeotomaDB/article-relevance/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/NeotomaDB/article-relevance.svg?style=for-the-badge
[forks-url]: https://github.com/NeotomaDB/article-relevance/network/members
[stars-shield]: https://img.shields.io/github/stars/NeotomaDB/article-relevance.svg?style=for-the-badge
[stars-url]: https://github.com/NeotomaDB/article-relevance/stargazers
[issues-shield]: https://img.shields.io/github/issues/NeotomaDB/article-relevance.svg?style=for-the-badge
[issues-url]: https://github.com/NeotomaDB/article-relevance/issues
[license-shield]: https://img.shields.io/github/license/NeotomaDB/article-relevance.svg?style=for-the-badge
[license-url]: https://github.com/NeotomaDB/article-relevance/blob/master/LICENSE.txt
[codecov-shield]: https://img.shields.io/codecov/c/github/NeotomaDB/article-relevance?style=for-the-badge
[codecov-url]: https://codecov.io/gh/NeotomaDB/article-relevance
