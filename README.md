# EndoOHCs
Repository for [Endometriosis Online Communities: A Quantitative Analysis](https://doi.org/10.1101/2024.02.27.24303445)

**Table of Contents**

1. [Project Overview](#project-overviewverview)
2. [Repository Overview](#repository-overview)
3. [License](#license)
4. [Citation](#citation)

## Project Overview

### Background
Endometriosis is a chronic condition that affects 10% of people with a uterus. Due to the complex social and psychological impacts caused by the condition, people with endometriosis often turn to online health communities (OHCs) for support.

### Objective
Prior work identifies a lack of large-scale analyses of endometriosis patient experiences and of OHCs. Our study fills this gap by investigating aspects of the condition and aggregate user needs that emerge from two endometriosis OHCs, r/Endo and r/endometriosis.

### Methods
We leverage topic modeling and supervised machine learning to identify associations between a post’s subject matter (“topics”), the people and relationships (“personas”) mentioned, and the type of support the post seeks (“intent”).

### Results
The most discussed topics in posts are medical stories, medical appointments, sharing symptoms, menstruation, and empathy. In addition, when discussing medical appointments, users are more likely to mention the endometriosis OHCs than medical professionals. Furthermore, medical professional is the least likely of any persona to be associated with empathy. Posts that mention partner or family are likely to discuss topics from the life issues category, in particular fertility. Lastly, we find that while users seek experiential knowledge regarding treatments and healthcare processes, they also wish to vent and to establish emotional connections about the life-altering aspects of the condition.

### Conclusions
Endometriosis OHCs provide members a space where they can discuss care pathways, learn to manage symptoms, and receive validation. Our results emphasize the need for greater empathy within clinical settings, easier access to appointments, more information on care pathways, and further support for patient loved ones. In addition, this study demonstrates the value of quantitative analyses of OHCs: they can support and extend findings from small-scale studies about patient experiences and provide insight into hard-to-reach groups. Lastly, analyses of OHCs can help design interventions to improve care, as argued in previous studies.

## Repository Overview

### `Code` folder
The following are the most important folders and files:
- `analyze_results` contains the code used to conduct analysis of our topic modeling and DistilBERT results: calculate inter-annotator agreement, conduct statistical significance testing to assess the difference in ratings between general and endometriosis answers, and create figures.
- `bert` contains the code used to fine-tune DistilBERT models for the persona and intent classification tasks, as well as to predict persona and intent labels in paragraphs and posts.
- `data_processing` contains code to do silly data wrangling.
- `output` contains classification reports detailing the performance of the DistilBERT model, as well as their config file.
- `community_fw.ipynb` contains code to compare r/Endo, r/endometriosis, and r/pcos using Fightin' Words.
- `topic_modeling.ipynb` contains code to conduct topic modeling with different settings on a dataset of choice.

### `Data` folder
contains graphs showing statistics about the source dataset.

### `Labeling` folder
contains functions and files to set up the annotation tool Prodigy and conduct annotations for the persona and intent labels.
- More to come!

### `Output` folder
The following are the most important folders and files:
- `fightin-words` contains the Fightin' Words results in LaTeX format from comparing r/Endo, r/endometriosis, and r/pcos.
- `topic-modeling` contains the result of the topic model we chose for our analysis after human evaluation.
- `predictions` contains the model predictions for the persona and intent labels -> BRB!

## License
This repository is released under [ODC-BY](https://opendatacommons.org/licenses/by/1.0/) license. 
By downloading this data you acknowledge that you have read and agreed to all the terms in this license.

## Citation
```
@misc{bologna_endometriosis_2024,
	title = {Endometriosis {Online} {Communities}: {A} {Quantitative} {Analysis}},
	author = {Bologna, Federica and Thalken, Rosamond and Pepin, Kristen and Wilkens, Matthew},
	shorttitle = {Endometriosis {Online} {Communities}},
	url = {https://www.medrxiv.org/content/10.1101/2024.02.27.24303445v1},
	doi = {10.1101/2024.02.27.24303445},
	language = {en},
	publisher = {medRxiv},
	month = feb,
	year = {2024},
}
```