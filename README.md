# Benchmarking-KE

#### Sparql querry
To get triples from the KG, we extracted entities in [MLaKE](https://github.com/Hi-archers/MLaKE/blob/main/dataset/single_hop/en_qa.json) and [MQuAKE](https://github.com/princeton-nlp/MQuAKE) which we use to query the DICE Dbpedia endpoint and construct our sub-knowledge graph. To get the triples files in the corrected form, run:

```
cd evaluate_rules/
python SparqlQuery.py

```


#### Running AMIE
To get the rules run the following cmd:

```
cd evaluate_rules/amie
java -jar amie-dev.jar  -mins 1 ../all_triples/processed_triples3.txt > ../all_triples/output_file.txt

```
  
  Make sure that you have the lates version [Java] installed to run AMIE, Download an AMIE executable jar file [AMIE-JAR], and run the commands above.
  
  
  #### Generating multihop questions and answers
  To generate the questions and answer run:
  
  ```python generateQA.py```
  
 The outputs print the question and the answer related to the given rules and facts and save the QA dict into a file.
