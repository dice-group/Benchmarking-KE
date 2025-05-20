# Benchmarking-KE

#### Sparql querry
To get triples from the KG, we extracted entities in [MLaKE](https://github.com/Hi-archers/MLaKE/blob/main/dataset/single_hop/en_qa.json) and [MQuAKE](https://github.com/princeton-nlp/MQuAKE) which we use to query the DICE Dbpedia endpoint and construct our sub-knowledge graph. To get the triples files in the corrected form, run:

```
cd evaluate_rules/
python SparqlQuery.py

```


#### Running AMIE
To get the rules, run the following commands:

```
cd evaluate_rules/amie
java -jar amie-dev.jar  -mins 1 ../all_triples/processed_triples3.txt > ../all_triples/output_file.txt

```
  
  Make sure that you have the lates version [Java] installed to run AMIE, Download an AMIE executable jar file [AMIE-JAR], and run the commands above.
  
  
  #### Generating multihop questions and answers
  To generate the questions and answer run:
  
  ```python generateQA.py```
  
 The outputs print the question and the answer related to the given rules and facts and save the QA dict into a file.
 
 
#### Running KE methods
We first run KE methods over the selected datasets (MLaKE and MQuAKE) and save the models weights
To do so, you will need to clone rome repository into your local folder, and  and run the following commands :
```
git clone https://github.com/kmeng01/rome.git
cd rome/rome  or cd rome/baselines for others KE

python rome_main.py --model_name openai-community/gpt2-large --dataset_path ../evaluate_rules/all_triples/MLaKE/new_en-qa.json --config ../hparams/ROME/gpt2-large.json --save_dir edited_models  #Feel free to change the model and the datasets path

```

Config files for each KE can be found in /hparams and others KE are placed in /baselines. An example of python code use to run can be found in /examples folder.
 
#### Evaluating KE methods

To evauale existing KE techniques on directly edited or correleated knowledge after saving models weights, run the followings commands .

```
python evaluate_rules/rome_eval.py  #for correlated knowledge 
python evaluate_rules/rome_eval_direct   #for directly edited knowledge

```
This will save the evaluation results in /results
 
