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
#import openai
import pandas as pd
import requests
from rdflib import Graph, URIRef
import warnings

 # Added for handling JSON output

try:
    nlp = spacy.load("en_core_web_sm")  # Or "en_core_web_trf" or any other model
except OSError:
    print("Please download a spaCy model (e.g., 'python -m spacy download en_core_web_sm')")
    exit()


def extract_entities(question):
    """
    Extracts entities from a question using spaCy.

    Args:
        question (str): The question string.

    Returns:
        list: A list of extracted entities (strings).
    """
    if not isinstance(question, str):
        print(f"Warning: Skipping invalid input: {question}")
        return []
    doc = nlp(question)
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)
    return entities


def query_sparql_endpoint(endpoint, entity_uris):
#    """Queries a SPARQL endpoint for triples involving a list of entities."""

    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)
    all_triples = []

    #for entity_uri in entity_uris:
    #print(entity_uris)
    query = f"""SELECT ?s ?p ?o
    WHERE {{
      {{
        SELECT ?s ?p ?o WHERE {{
          VALUES ?s {{ <{entity_uris}> }}
          ?s ?p ?o
          FILTER (
            ?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> &&
            ?p != <http://www.w3.org/2000/01/rdf-schema#label> &&
            ?p != <http://www.w3.org/2002/07/owl#sameAs> &&
            ?p != <http://dbpedia.org/property/wikiPageUsesTemplate> &&
            ?p != <http://dbpedia.org/ontology/wikiPageRedirects> &&
            ?p != <http://dbpedia.org/ontology/almaMater> &&
            ?p != <http://dbpedia.org/ontology/wikiPageExternalLink> &&
            ?p != <http://dbpedia.org/ontology/wikiPageWikiLink> &&
            ?p != <http://www.w3.org/2000/01/rdf-schema#comment>
          )
        }}
        LIMIT 100
      }} UNION
      {{
        SELECT ?s ?p ?o WHERE {{
          VALUES ?o {{ <{entity_uris}> }}
          ?s ?p ?o
          FILTER (
            ?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> &&
            ?p != <http://www.w3.org/2000/01/rdf-schema#label> &&
            ?p != <http://www.w3.org/2002/07/owl#sameAs> &&
            ?p != <http://dbpedia.org/property/wikiPageUsesTemplate> &&
            ?p != <http://dbpedia.org/ontology/wikiPageRedirects> &&
            ?p != <http://dbpedia.org/ontology/almaMater> &&
            ?p != <http://dbpedia.org/ontology/wikiPageExternalLink> &&
            ?p != <http://dbpedia.org/ontology/wikiPageWikiLink> &&
            ?p != <http://www.w3.org/2000/01/rdf-schema#comment>
          )
        }}
        LIMIT 100
      }}
    }}"""

    # print(query)  # Debugging: Print the query
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        for result in results["results"]["bindings"]:
            if "s" in result and "p" in result and "o" in result:
                all_triples.append(f'{result["s"]["value"]}\t{result["p"]["value"]}\t{result["o"]["value"]}')
    except Exception as e:
        print(f"Error querying SPARQL endpoint for URI '{entity_uris}': {e}")

    return list(set(all_triples))


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
        #print(keyword)
        if keyword not in all_triples:
            entity = extract_entities(keyword)
            for ent in entity:
                entities = find_dbpedia_uri(ent)
                #print(entities)
                #break
                if entities is None:
                    remove_rows.append(i)
                    continue
                
                if entities and keyword not in all_triples:
                    result = query_sparql_endpoint(endpoint, entities)
                    #print(result)
                    #break
                    all_triples[keyword] = result
    
        
    with open("./all_triples/"+"all_triples_dict_mquake.json", "w") as f:
        json.dump(all_triples, f)
    
    with open("./all_triples/"+"removed_rows_mquake.json", "w") as f:
        json.dump(remove_rows, f)
    #save the triples
    with open( "./all_triples/triples.txt_mquake",'w', encoding='utf-8') as r: 
        for keyword, triples in all_triples.items():
            for t in triples:
                r.write(t + "\n")

    return all_triples


#process the triples
def process_triples_file(all_triples, output_file):
    """
    Reads a dictionary of questions and their triples, extracts the last part of each link,
    and writes the processed triples to a new file.

    Args:
        all_triples (dict): A dictionary where keys are questions and values are lists of triple strings (s\tp\to).
        output_file (str): Path to the output file to write processed triples.
    """
    processed_all_triples = {}
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for question, triples in all_triples.items():
                processed_triples_for_question = []
                for line in triples:
                    line = line.strip()
                    if not line:
                        continue

                    parts = re.split(r'\t| |    ', line)

                    if len(parts) != 3:
                        warnings.warn(f"Invalid triple format in line: '{line}'")
                        continue  # Skip lines with incorrect number of parts

                    s, p, o = parts
                    # Extract the last part of each link
                    s_last = s.split('/')[-1] if '/' in s else s
                    p_last = p.split('/')[-1] if '/' in p else p
                    o_last = o.split('/')[-1] if '/' in o else o

                    # Write the processed triple to the output file
                    processed_triple = f"{s_last}\t{p_last}\t{o_last}"
                    processed_triples_for_question.append(processed_triple)
                    outfile.write(processed_triple + '\n')
                processed_all_triples[question] = processed_triples_for_question
    except Exception as e:
        print(f"An error occurred in process_triples_file: {e}")
    return processed_all_triples

#saved_triples = save_triples(endpoint, data)
#print(saved_triples)
#output_file = 'all_triples/processed_triples_mquake.txt' # Replace with your desired output file name

#process_triples_file(saved_triples, output_file)
