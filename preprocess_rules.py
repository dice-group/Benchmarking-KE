from SparqlQuery import *
from SparqlQuery import extract_entities, process_triples_file



data = pd.read_csv("datasets/MLaKE/en_qa.csv")

triples = 'all_triples/processed_triples.txt'

rules = 'all_triples/triples_mined-rules1.txt'

#we save the relations inthe triples where entities appear in the graph
def extract_relations(data, triples_file):
    with open(triples_file, 'r', encoding='utf-8') as infile:
        relations = []
        for i, keyword in enumerate(data["question"]):
            if keyword not in relations:
                entity1 = extract_entities(keyword)
                for ent in entity1:
                    for elt in infile:
                        elt = elt.strip().split('\t')
                        #elt = elt.split('\t')
                        relations.append(elt[1])
    return relations

#filter rules
def filter_rules_from_file_by_entity(file_path, entity):
    """
    Filters rules from a file where an entity appears in the subject or object.

    Args:
        file_path: The path to the file containing the rules.
        entity: The entity to filter for.

    Returns:
        A list of rules that contain the specified entity.
    """
    filtered_rules = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                rule = line.strip()  # Remove leading/trailing whitespace
                parts = rule.split("=>")
                if len(parts) != 2:
                    continue  # Skip invalid rules

                antecedent = parts[0].strip()
                consequent = parts[1].strip()

                antecedent_parts = antecedent.split()
                consequent_parts = consequent.split()

                if entity in antecedent_parts or entity in consequent_parts:
                    filtered_rules.append(rule)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return filtered_rules


#get the list of rules
def rules_list(relations, rule_file ):
    filtered_rules_list = []
    for re in relations:
        filtered_rules = filter_rules_from_file_by_entity(rule_file, re)
    
    for f in filtered_rules:
        #print(f)
        parts = f.strip().split('=>')
        #parts =  [item.replace("?", "") for item in parts]
        head = parts[1].split('\t')
        Head = head[0].split('  ')
        antecedent = parts[0]
        consequent = head[0]
        filtered_rules_list.append((antecedent, '=>', consequent))

    return filtered_rules_list


