# 🖋️ Low-Cost Synthetic Data Generation for HTR Training

## Evaluating a Multimodal Strategy for Historical Manuscript Processing

This repository contains the data-augmentation workflow developed within
the **ERC PRIMA --- Manuscripts in the Age of Print (1575--1800)**
project for experiments on Handwritten Text Recognition (HTR) of
premodern Italian manuscripts.

The workflow generates synthetic training data from historical document
images and their corresponding **ALTO XML** files. It combines image
preprocessing, line-level transformations, synthetic handwriting
variation, page reconstruction, and controlled perturbation of layout
information while preserving the correspondence between images and
annotations.

The code was developed by **Serena Carlamaria Crespi** and
**Carlos-Emiliano González-Gallardo**.

------------------------------------------------------------------------

## Features

The workflow includes:

-   **Sauvola binarization** for document preprocessing
-   Morphological transformations including **dilation, erosion, and
    blurring**
-   Simulation of **ink bleeding**
-   Random **shadows and smudges**
-   Controlled perturbation of **baseline coordinates** in ALTO XML
-   Preservation of the correspondence between augmented images and XML
    annotations
-   Generation of script variations using **Bézier curves**
-   Reconstruction of manuscript pages from synthetic line-level data
-   Generation of multiple augmented versions of each source document

------------------------------------------------------------------------

## Workflow

The augmentation pipeline processes historical manuscript images
together with their ALTO XML annotations.

The general workflow is:

1.  Load manuscript images and corresponding ALTO XML files.
2.  Extract and preprocess text lines.
3.  Apply Sauvola binarization and image cleaning.
4.  Generate line-level graphical variations.
5.  Apply additional visual augmentation.
6.  Reconstruct synthetic manuscript pages.
7.  Update the corresponding ALTO XML information.
8.  Export paired synthetic images and annotations for HTR training.

This makes it possible to increase the size and graphical diversity of
an HTR training corpus while retaining the structural information
required by the recognition pipeline.

------------------------------------------------------------------------

## How to Use

### 1. Prepare the input data

Place the manuscript images (`.png` or `.jpg`) and their corresponding
ALTO XML (`.xml`) files in the input directory.

Each image must have a corresponding XML file with the same base
filename.

Example:

``` text
page_001.png
page_001.xml
page_002.png
page_002.xml
```

### 2. Create the environment

Create the Conda environment using:

``` bash
conda env create -f augmentation_htr.yaml
```

Then activate the environment according to the name defined in the YAML
file.

### 3. Run the workflow

``` bash
python data_generation.py
```

Generated files are written to the directories specified in
`config.json`.

By default, multiple augmented versions can be generated for each input
image.

------------------------------------------------------------------------

## Configuration

The main parameters can be configured in `config.json`.

  -----------------------------------------------------------------------------------
  Parameter                      Default value                  Description
  ------------------------------ ------------------------------ ---------------------
  `src`                          `data/src`                     Input directory

  `binarized_lines`              `data/binarized_lines`         Output directory for
                                                                binarized text lines

  `binarized_lines_clean`        `data/binarized_lines_clean`   Output directory for
                                                                cleaned binarized
                                                                lines

  `augmented_lines`              `data/augmented_lines`         Output directory for
                                                                generated line
                                                                variations

  `rebuilt_pages`                `data/rebuilt_pages`           Output directory for
                                                                reconstructed pages

  `augmented_pages`              `data/augmented_pages`         Output directory for
                                                                augmented pages

  `baseline_noise`               `4`                            Baseline perturbation
                                                                used during
                                                                augmentation

  `crop_margin_v`                `5`                            Vertical crop margin

  `crop_margin_h`                `15`                           Horizontal crop
                                                                margin

  `sauvola_window_size`          `35`                           Window size for
                                                                Sauvola binarization

  `sauvola_k`                    `0.3`                          Sauvola parameter *k*

  `cleaner_min_area`             `15`                           Minimum
                                                                connected-component
                                                                area retained during
                                                                cleaning

  `augmentation_k1`              `0.1`                          Bézier transformation
                                                                parameter *k1*

  `augmentation_k2`              `0.2`                          Bézier transformation
                                                                parameter *k2*

  `augmentation_stroke_radius`   `4`                            Stroke radius used
                                                                during graphical
                                                                augmentation
  -----------------------------------------------------------------------------------

------------------------------------------------------------------------

## Requirements

Dependencies are defined in `augmentation_htr.yaml`.

Install them with:

``` bash
conda env create -f augmentation_htr.yaml
```

------------------------------------------------------------------------

## Notes

-   The augmentation workflow is non-destructive: the original source
    data are not modified.
-   Several transformations include stochastic operations; consequently,
    different runs may produce different synthetic outputs.
-   Parameters can be adjusted in `config.json` to adapt the workflow to
    different manuscript corpora.
-   The workflow was developed and tested in the context of HTR
    experiments on premodern Italian manuscript material.

------------------------------------------------------------------------

## Acknowledgements and Third-Party Code

The Bézier-based augmentation component builds upon
[`script-level_aug_ICFHR2022`](https://github.com/IMU-MachineLearningSXD/script-level_aug_ICFHR2022),
which was adapted for use with historical manuscript material.

Please refer to the original project for information concerning its
implementation, licensing, and attribution requirements.

------------------------------------------------------------------------

## How to Cite

If you use this software in your research, please cite:

> Crespi, Serena Carlamaria, and Carlos-Emiliano González-Gallardo.\
> *PRIMA HTR Augmentation Code*. Version 1.0, 2025.\
> ERC PRIMA, Centre d'Études Supérieures de la Renaissance, Université
> de Tours.

Citation metadata are also available in the [CITATION.cff](CITATION.cff)
file.

------------------------------------------------------------------------

## Authors

**Serena Carlamaria Crespi**\
ERC PRIMA · Centre d'Études Supérieures de la Renaissance · Université de Tours\
ORCID: https://orcid.org/0000-0001-6747-3257

**Carlos-Emiliano González-Gallardo**\
LIFAT — Laboratoire d'Informatique Fondamentale et Appliquée de Tours · Université de Tours\
ERC PRIMA External Collaborator · Centre d'Études Supérieures de la Renaissance\
ORCID: https://orcid.org/0000-0002-0787-2990

------------------------------------------------------------------------

## Project

This software was developed within:

**ERC PRIMA --- Manuscripts in the Age of Print (1575--1800)**\
Grant Agreement No. **101142242**

Centre d'Études Supérieures de la Renaissance\
Université de Tours

------------------------------------------------------------------------

## License

This repository contains research software developed within the ERC
PRIMA project.

See the `LICENSE` file for the terms governing reuse and distribution.
