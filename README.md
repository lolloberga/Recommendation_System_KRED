# KRED: Considerations over Attention and Embeddings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Digital news are arguably ones of the most consumed products by millions of people every day 
and the importance of a proper recommendation system has always been fundamental for news and press websites.
KRED deals with a knowledge graph system, taking in account information from entity’s neighborhood, and refines the
entity embedding with dynamic insights depending on the context of the news.
This repository proposes two main improvements to the original model: a Multi-Head Attention module 
and an enrichment of side information encodings, such as category. 
The dataset used is MIND dataset, created on a huge amount of Microsoft News interactions between users and news.

--------------------------------------------------------------

## Table of content
- [Description](#description)
- [Setup](#setup)
  - [Requirements](#requirements)
  - [Enviroment](#enviroment)
  - [Dataset and Features](#dataset-and-features)
- [Getting started](#getting-started)
- [Contacts](#contacts)

--------------------------------------------------------------

## Description
KRED is a knowledge enhanced framework which enhance a document embedding with knowledge information for multiple news recommendation tasks. 
The framework mainly contains two part: representation enhancement part and multi-task training part.

![](./framework.PNG)

Two extensions to this model have been implemented:
1. in the context embedding layer, we enriched the entity emdedding with the news category, consisting of a first general category and a second more specific category
2. in the information distillation layer, we replaced the self-attention module with a multi-head attention module

The backbone resources of this project are listed in this table, including papers and corresponding GitHub repos.

| Paper | Title | Implementation source |
| ----- | ----- | --------------------- |
|[1910.11494](https://arxiv.org/abs/1910.11494)|KRED: Knowledge-Aware Document Representation for News Recommendations|[KRED](https://github.com/danyang-liu/KRED)|
|[2020.acl-main.331](https://aclanthology.org/2020.acl-main.331/)|MIND: A Large-scale Dataset for News Recommendation|[MIND](https://msnews.github.io/)|

--------------------------------------------------------------

## Setup

### Requirements
1. The project has been tested on Linux (Ubuntu 18.04), MacOS and Windows (10)
2. Python 3.8+ version is needed
3. Multiple NVIDIA GPU are mandatory to run the model in feasible time but, if the dataset is undersampled, the number of GPUs needed can be reduced; we could run the model with Colab's GPU (Tesla T4 16GB), after sampling properly MIND

### Enviroment
Once the repo is cloned, some python libraries are required to properly setup your (virtual) enviroment.


They can be installed via pip:
```bash
    pip install -r requirements.txt
```

or via conda:
```bash
    conda create --name <env_name> --file requirements.txt
```

### Dataset and Features
The dataset explored by KRED is MIcrosoft News Dataset (MIND), a large-scale dataset collected by anonymized behavior logs of Microsoft News website. 
Constructed from the user click logs, MIND contains 1 million users and more than 160k English news articles; 
in addition, Microsoft has released a smaller version of the original dataset, by randomly sampling 50k users and their behavior logs.

After preprocessing the dataset with BERT, each Document Vector should be structured in:
- News ID
- Category and Sub-Category
- Title and a brief Abstract
- URL of the news
- List of entities in the Title
- List of entities in the Abstract

In turn, each entity should be characterized by:
- Label, which the entity is associated to
- Type of entity, such as person, object ecc.
- WikidataId of the label
- Confidence of label
- Position of first char in the text
- Entity itself

--------------------------------------------------------------

## Getting started
Before running the code, you should check `config.yaml` and setup all the parameters correctly; 
once done, you can perfom the training and testing phases in `main.py`.

Alternatively, you can follow the instructions in `kred_example.ipynb` for a guided experience.

--------------------------------------------------------------

## Contacts

| Author | Mail | GitHub | 
| ------ | ---- | ------ |
| **Lorenzo Bergadano** | s304415@studenti.polito.it | [lolloberga](https://github.com/lolloberga) |
| **Stefano Gioda** | s294781@studenti.polito.it | [Giost](https://github.com/Giost) |
| **Paolo Rizzo** | paolo.rizzo@studenti.polito.it | [polrizzo](https://github.com/polrizzo) |
