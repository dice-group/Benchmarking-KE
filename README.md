# Benchmarking-KE

#### Sparql querry
To get triples from the KG, we extracted entities in [MLaKE](https://github.com/Hi-archers/MLaKE/blob/main/dataset/single_hop/en_qa.json) which we use to query the DICE Dbpedia endpoint and get triples. To get the triples files in the corrected form, type:

```python SparqlQuery.py```


#### Running AMIE
To get the rules first clone the amie repository as follow:

```git clone https://github.com/dig-team/amie.git```
```cd amie```
```java -jar amie-dev.jar  -mins 1 ../all_triples/processed_triples.txt > ../all_triples/output_file.txt```
  
  Make sure that you have the lates version [Java] installed to run AMIE, Download an AMIE executable jar file [AMIE-JAR], and run the commands above.
  
  
  #### Generating multihop questions and answers
  To generate the questions and answer run:
  
  ```python generateQA.py```
  
  You will need to have your Hugginface Face API key configured. The outputs print the question and the answer related to the given rules and facts.
