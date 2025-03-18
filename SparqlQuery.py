#import librairies
import spacy
import random
import csv
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.parse import quote
import json
import re
import subprocess
import os
import openai
import pandas as pd
import requests

 # Added for handling JSON output

try:
    nlp = spacy.load("en_core_web_sm")  # Or "en_core_web_trf" or any other model
except OSError:
    print("Please download a spaCy model (e.g., 'python -m spacy download en_core_web_sm')")
    exit()


endpoint = "https://dbpedia.data.dice-research.org/sparql"
    
data = pd.read_csv("all_triples/en_qa.csv")

#Extract entities from our dataset(MLaKE/en_qa.csv)
def extract_entities(question):
    """
    Extracts entities from a question using spaCy.

    Args:
        question (str): The question string.

    Returns:
        list: A list of extracted entities (strings).
    """
    doc = nlp(question)
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)
    return entities

#query the KG given an entity taken fron the dataset
def query_sparql_endpoint(endpoint, entity):
    """Queries a SPARQL endpoint for triples involving an entity."""

    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)

    query = """SELECT?s?p?o 
    WHERE {
      { <"""+entity+""">?p?o }
      UNION
      {?s?p <"""+entity+"""> }
    
    } LIMIT 100"""
    
    query = """SELECT ?s ?p ?o 
    WHERE {
      {
        { SELECT ?s ?p ?o WHERE { VALUES ?s { <"""+entity+"""> } ?s ?p ?o FILTER(?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> && ?p!= <http://www.w3.org/2000/01/rdf-schema#label> && ?p!= <http://www.w3.org/2002/07/owl#sameAs> && ?p!= <http://dbpedia.org/property/wikiPageUsesTemplate> && ?p!= <hhttp://dbpedia.org/ontology/wikiPageRedirects> && ?p!= <http://dbpedia.org/ontology/almaMater> && ?p!= <http://dbpedia.org/ontology/wikiPageExternalLink> && ?p!=<http://dbpedia.org/ontology/wikiPageWikiLink> && ?p!= <http://www.w3.org/2000/01/rdf-schema#comment>) } LIMIT 50 }
      } UNION 
      { 
        { SELECT ?s ?p ?o WHERE { VALUES ?o { <"""+entity+"""> } ?s ?p ?o FILTER(?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> && ?p!= <http://www.w3.org/2000/01/rdf-schema#label> && ?p!= <http://www.w3.org/2002/07/owl#sameAs> && ?p!= <http://dbpedia.org/property/wikiPageUsesTemplate> && ?p!= <hhttp://dbpedia.org/ontology/wikiPageRedirects> && ?p!= <http://dbpedia.org/ontology/almaMater> && ?p!= <http://dbpedia.org/ontology/wikiPageExternalLink> && ?p!=<http://dbpedia.org/ontology/wikiPageWikiLink> && ?p!= <http://www.w3.org/2000/01/rdf-schema#comment>) } LIMIT 50 }
      } 
    } """

    #print(query)  # Debugging: Print the query
    sparql.setQuery(query)
    triples = []
    try:
        results = sparql.query().convert()
        for result in results["results"]["bindings"]:
            if "s" in result and "p" in result and "o" in result:
                triples.append(f'{result["s"]["value"]}\t{result["p"]["value"]}\t{result["o"]["value"]}')
        return triples
    except Exception as e:
        print(f"Error querying SPARQL endpoint: {e}")
        return []



#get the uri of a given entity
def find_dbpedia_uri(keyword):
    """
    Finds the URI of a DBpedia entity based on a keyword search.

    Args:
        keyword: The keyword to search for.

    Returns:
        The URI of the DBpedia entity, or None if no matching entity is found.
    """

    endpoint = "https://dbpedia.data.dice-research.org/sparql" #We choose DICE sparql dbpedia
    query = f"""
        SELECT DISTINCT ?s 
        WHERE {{ 
            {{ ?s <http://www.w3.org/2000/01/rdf-schema#label> "{keyword}" }} 
            UNION 
            {{ ?s <http://www.w3.org/2000/01/rdf-schema#label> "{keyword}"@en }} 
        FILTER (!CONTAINS(STR(?s), "Category:")) 
        }}
    """
    sparql = SPARQLWrapper(endpoint)
    
    sparql.setReturnFormat(JSON)

    sparql.setQuery(query)

    try:
        results = sparql.query().convert()
        if results["results"]["bindings"]:  # Check if any results were found
            return results["results"]["bindings"][0]["s"]["value"]  # Return the URI of the first result
        else:
            return None  # No matching entity found
    except Exception as e:
        print(f"Error: {e}")
        return None

#Here we save all the triples_dict in json file, the rows that do not correspond to any entity and the triples in a txt file
def save_triples(endpoint, data):

    all_triples = {} 
    remove_rows = []
    
    for i, keyword in enumerate(data["question"]):
        if keyword not in all_triples:
            entity = extract_entities(keyword)
            for ent in entity:
                entities = find_dbpedia_uri(ent)
                if entities is None:
                    remove_rows.append(i)
                    continue
                
                if entities and keyword not in all_triples:
                    result = query_sparql_endpoint(endpoint, entities)
                    all_triples[keyword] = result
    
        
    with open("./all_triples/"+"all_triples_dict.json", "w") as f:
        json.dump(all_triples, f)
    
    with open("./all_triples/"+"removed_rows.json", "w") as f:
        json.dump(remove_rows, f)
    #save the triples
    with open( "./all_triples/triples.txt",'w', encoding='utf-8') as r: 
        for keyword, triples in all_triples.items():
            for t in triples:
                r.write(t + "\n")

    return all_triples


#process the triples
def process_triples_file(all_triples, output_file):
    """
    Reads a file containing triples (s\tp\to), extracts the last part of each link,
    and writes the processed triples to a new file.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile: # \
             #open(output_file, 'w', encoding='utf-8') as outfile:

            for keyword, triples in all_triples.items():
                for line in triples:  
                    line = line.strip()
                    if not line:
                        continue
            #for line in infile:
                #line = line.strip()  # Remove leading/trailing whitespace
                #if not line:  # Skip empty lines
                    #continue

                # Split by tab or space using regular expressions
                parts = re.split(r'\t| |    ', line)
                #parts = re.split(r'\t| ', line)

                if len(parts) != 3:
                    print(f"Warning: Invalid triple format in line: '{line}'")
                    continue  # Skip lines with incorrect number of parts

                s, p, o = parts
                # Extract the last part of each link
                s_last = s.split('/')[-1]
                p_last = p.split('/')[-1]
                o_last = o.split('/')[-1]

                # Write the processed triple to the output file
                outfile.write(f"{s_last}\t{p_last}\t{o_last}\n")

        print(f"Processed triples saved to '{output_file}'.")

    #except FileNotFoundError:
        #print(f"Error: Input file '{input_file}' not found.")
    except ValueError:
        print(f"Error: Invalid triple format in '{all_triples}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

saved_triples = save_triples(endpoint, data)
output_file = 'all_triples/processed_triples.txt' # Replace with your desired output file name

process_triples_file(saved_triples, output_file)
