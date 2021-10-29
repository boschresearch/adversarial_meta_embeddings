<!---

    Copyright (c) 2021 Robert Bosch GmbH and its subsidiaries.

-->

# FAME: Feature-based Adversarial Meta-Embeddings

This is the companion code for the experiments reported in the paper

> "FAME: Feature-Based Adversarial Meta-Embeddings
for Robust Input Representations"  by Lukas Lange, Heike Adel, Jannik Strötgen and Dietrich Klakow published at EMNLP 2021.

The paper can be found [here](TODO). 
The code allows the users to reproduce the results reported in the paper and extend the model to new datasets and embedding configurations. 
Please cite the above paper when reporting, reproducing or extending the results as:

## Citation

```
@inproceedings{lange-etal-2021-fame,
    title = "FAME: Feature-Based Adversarial Meta-Embeddings for Robust Input Representations",
    author = {Lange, Lukas  and
      Adel, Heike  and
      Str{\"o}tgen, Jannik and
      Klakow, Dietrich},
    booktitle = "EMNLP",
    month = nov,
    year = "2021",
}
```

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication "FAME: Feature-Based Adversarial Meta-Embeddings for Robust Input Representations". 
It will neither be maintained nor monitored in any way.

## Setup

* Install flair and transformers (Tested with flair=0.8, transformers=3.3.1, pytorch=1.6.0 and python=3.7.9)
* Download pre-trained word embeddings (using flair or your own).
* Prepare corpora in BIO format.
* Train a [sequence-labeling](Sequence_Labeling.ipynb) or [text-classification](Text_Classification.ipynb) model as described in the example notebooks. 

## Data

We do not ship the corpora used in the experiments from the paper. 

## License

FAME is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in Joint-Anonymization-NER, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

The software including its dependencies may be covered by third party rights, including patents. You should not execute this code unless you have obtained the appropriate rights, which the authors are not purporting to give.