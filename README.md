# Benchmarking-KE

This repository contains a benchmark evaluation of Knowledge Editing using logical rules. Our methodology includes multi-hop questions generated using logical rules to evaluate the effectiveness of knowledge editing methods. We conducted experiments on the popular approaches ROME, FT and KN and, the results show a considerable performance gap of up to 24% between evaluations on directly edited knowledge and on entailed knowledge particularly on ROME and FT.

<p align="center">
  <img src="images/approach.pdf"/>
</p>

### Installation
To start, install the required packages:

```
cd evaluate_rules
pip install torch
pip install -r requirements.txt
```
Ensure that all dependencies are correctly installed

### Sparql querry
To get triples from the KG, we extracted entities in [MLaKE](https://github.com/Hi-archers/MLaKE/blob/main/dataset/single_hop/en_qa.json) and [MQuAKE](https://github.com/princeton-nlp/MQuAKE) which we use to query the DICE Dbpedia endpoint and construct our sub-knowledge graph. To get the triples files in the corrected form, run:
```
cd evaluate_rules/
python SparqlQuery.py
```


### Running AMIE
To get the rules, run the following commands:

```
cd evaluate_rules/amie
java -jar amie-dev.jar  -mins 1 ../all_triples/processed_triples3.txt > ../all_triples/output_file.txt
```
  
  Make sure that you have the latest version [Java] installed to run AMIE, Download an AMIE executable jar file [AMIE-JAR], and run the commands above.
  
  
  #### Generating multihop questions and answers
  To generate the questions and answer run:
  
  ```python generateQA.py```
  
 The outputs print the question and the answer related to the given rules and facts and save the QA dict into a file.
 
 The datasets used in the experiments, all triples, rules and multihop_qa_pairs for each dataset are found in /evaluate_rules/all_triples .
 
 
### Running KE methods
We first run KE methods over the selected datasets (MLaKE and MQuAKE) and save the models weights
To do so, you will need to clone rome repository into your local folder, and  and run the following commands :
```
git clone https://github.com/kmeng01/rome.git
cd rome/rome  or cd rome/baselines for others KE

python rome_main.py --model_name openai-community/gpt2-large --dataset_path ../evaluate_rules/all_triples/MLaKE/new_en-qa.json --config ../hparams/ROME/gpt2-large.json --save_dir edited_models  #Feel free to change the model and the datasets path
```

Config files for each KE can be found in /hparams and others KE are placed in /baselines. Some examples of python code used to run each KE are found in /examples folder.
 
### Evaluating KE methods

To evaluate existing KE techniques on directly edited or correlated knowledge after saving models weights, run the followings commands .

```
python evaluate_rules/rome_eval.py  #for correlated knowledge 
python evaluate_rules/rome_eval_direct   #for directly edited knowledge
```
This will save the evaluation results in /results


### Maintenance

In the future, we will conduct the following experiments by adding other KE methods such as:
- `MEND`
- `KE`
- `MeLLo`
Including the following knowledge graphs:
- `Wikidata`
- `YAGO`

<table>
  <tr><th align="left">Model</th><th>Status</th></tr>
  <tr><th align="left">LLama2</th><td>Upcoming</td></tr>
  <tr><th align="left">GPT-j</th><td>In progress</td></tr>
  <tr><th align="left">GPT-3-based architectures</td></tr>In progress</td></tr>
  <tr><th align="left">Mistral</th><td>Upcoming</td></tr>
</table>
 
### Citation
```
 Moteu Ngoli, T. (2025). Benchmarking Knowledge Editing using Logical Rules (1.0.0) [Data set]. THE 24th INTERNATIONAL SEMANTIC WEB CONFERENCE (ISWC 2025), Nara, Japan. Zenodo. https://doi.org/10.5281/zenodo.15697400
 ```
#### DOI
[https://doi.org/10.5281/zenodo.15697400](https://doi.org/10.5281/zenodo.15697400)
 
### Contact
Feel free to contact us at tatianam@mail.uni-paderborn.de if you have any questions.
