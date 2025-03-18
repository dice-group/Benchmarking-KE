# Benchmarking-KE

#### Sparql querry
To get triples from the KG, we extracted entities in [MLaKE](https://github.com/Hi-archers/MLaKE/blob/main/dataset/single_hop/en_qa.json) which we use to query the DICE Dbpedia endpoint and get triples. To get the triples files in the corrected form, type:

```python SparqlQuery.py```


#### Running AMIE
To get the rules firt clone the amie repository as follow:

```git clone https://github.com/dig-team/amie.git
   cd amie
  java -jar amie-dev.jar  -mins 1 ../all_triples/processed_triples.txt > ../all_triples/output_file.txt```
