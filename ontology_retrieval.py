
import spacy
import rdflib
import yaml
import json

from rdflib import Graph, RDFS, URIRef, XSD
from urllib.request import Request, urlopen
from urllib.parse import urlencode


# Disable selected automatic datatype conversions
for dt in (XSD.date, XSD.dateTime, XSD.decimal):
    rdflib.term._toPythonMapping.pop(dt, None)

nlp = spacy.load("en_core_web_sm")

graph = Graph()
graph.parse("digitrubber-edit.owl", format="xml")

# Define ontology predicates
LABEL = RDFS.label
SYNONYM = URIRef("http://purl.obolibrary.org/obo/IAO_0000118")


# Sample Questions 

questions = ["Who is the contributor to silane?", "What is the definition of rubber?"]

# Load the YAML configuration

with open("config.yaml", "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)


DATABASE_CROSS_REFERENCE = URIRef("http://www.geneontology.org/formats/oboInOwl#hasDbXref")

# Extract entity and relationship using spaCy

for question in questions:

    doc = nlp(question)

    entity = None
    relationship = None

    metadata_value = None
    entity_label_value = None
    entity_wikidata_id = None
    label_value = None
    description_value = None

    for token in doc:

        if token.dep_ == "nsubj" and token.pos_ == "NOUN":
            relationship = token.text

        if token.dep_ in ["pobj", "xcomp"]:
            entity = token.text


    # Normalize the extracted entity against ontology

    normalized_entity = None
    entity_uri = None
    match_method = None

    if entity:

        for subject, label in graph.subject_objects(LABEL):

            if str(label).strip().lower() == entity.strip().lower():
                normalized_entity = str(label)
                entity_uri = subject
                match_method = "label"
                break

        if normalized_entity is None:

            for subject, synonym in graph.subject_objects(SYNONYM):

                if str(synonym).strip().lower() == entity.strip().lower():
                    normalized_entity = str(synonym)
                    entity_uri = subject
                    match_method = "synonym"
                    break

    # Normalize the extracted relationship

    normalized_relationship = None

    if relationship:
        normalized_relationship = relationship.strip().lower()



    print("\nQuestion:", question)
    print("Extracted entity:", entity)
    print("Normalized entity:", normalized_entity)
    print("Entity URI:", entity_uri)
    print("Entity match method:", match_method)
    print("Extracted relationship:", relationship)
    print("Normalized extracted relationship:", normalized_relationship)


    # Extract metadata and match properties

    matched_property_uri = None

    if entity_uri and normalized_relationship:

        print("\nMatching extracted relationship with ontology metadata:")

        property_found = False

        for predicate, obj in graph.predicate_objects(entity_uri):

            property_label = graph.value(predicate, LABEL)

            if property_label:
                property_name = str(property_label)
            else:
                property_name = str(predicate).split("/")[-1].split("#")[-1]

            normalized_property = property_name.strip().lower()

            if normalized_relationship == normalized_property:

                property_found = True
                matched_property_uri = predicate

                print("\nMatch found!")
                print("Normalized extracted relationship:", normalized_relationship)
                print("Normalized ontology property:", normalized_property)
                print("Property URI:", predicate)

                print("-" * 50)

                break

        if not property_found:
            print("\nNo matching metadata property found.")

    elif not entity_uri:
        print("\nEntity not found in Digital Rubber ontology.")

    elif not normalized_relationship:
        print("\nNo relationship extracted from the question.")

    # 9. SPARQL query template for dynamic metadata retrieval

    if entity_uri and matched_property_uri:

        sparql_template = """
        SELECT ?value
        WHERE {
            ENTITY_URI PROPERTY_URI ?value .
        }
        """

        query = sparql_template.replace("ENTITY_URI", URIRef(entity_uri).n3()).replace("PROPERTY_URI", URIRef(matched_property_uri).n3())

        results = graph.query(query)

        print("\nSPARQL query results:")

        result_found = False

        for result in results:

            result_found = True
            value = result.value

            value_label = graph.value(value, LABEL)

            if value_label:
                metadata_value = str(value_label)
            else:
                metadata_value = str(value)

            print("Retrieved metadata:", metadata_value)

        if not result_found:
            print("No metadata value found.")

    # Dynamic metadata values retrieval and Wikidata interoperability

    #  Retrieve configured metadata for the matched entity

    if entity_uri:

        print("\nDynamic entity metadata retrieval:")

        entity_metadata = [
            item for item in config.get("entity_metadata", [])
            if item.get("enabled", False)
            and item.get("default", False)
        ]

        entity_query_parts = []
        selected_variables = []

        for item in entity_metadata:

            if item["name"] == "label":

                entity_query_parts.append(
                    """
                    OPTIONAL {
                        ENTITY_URI rdfs:label ?entity_label .
                        FILTER(LANG(?entity_label) = "LANGUAGE")
                    }
                    """.replace("ENTITY_URI", URIRef(entity_uri).n3()).replace("LANGUAGE", item.get("language", "en")))

                selected_variables.append("?entity_label")

            elif item["name"] == "wikidata_id":

                entity_query_parts.append(
                    """
                    OPTIONAL {
                        ENTITY_URI oboInOwl:hasDbXref ?entity_wikidata_id .
                    }
                    """.replace("ENTITY_URI", URIRef(entity_uri).n3()))

                selected_variables.append("?entity_wikidata_id")

        if entity_query_parts:

            entity_sparql_template = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>

            SELECT SELECTED_VARIABLES
            WHERE {
                METADATA_PATTERNS
            }
            """

            entity_query = entity_sparql_template.replace("SELECTED_VARIABLES"," ".join(selected_variables)).replace("METADATA_PATTERNS","\n".join(entity_query_parts))

            print("\nEntity SPARQL query:")
            print(entity_query)

            entity_results = graph.query(entity_query)

            for result in entity_results:

                if result.entity_label:
                    entity_label_value = str(result.entity_label)

                    print("English entity label:", entity_label_value)

                if result.entity_wikidata_id:

                    entity_wikidata_id = str(result.entity_wikidata_id).strip()

                    print("Entity Wikidata ID:", entity_wikidata_id)

        else:

            print("No default entity metadata configured.")

    else:

        print("\nNo matched entity available for metadata retrieval.")


    # Retrieve configured metadata from Wikidata

    wikidata_metadata = [
        item for item in config.get("wikidata_metadata", [])
        if item.get("enabled", False)
        and item.get("default", False)
    ]

    if entity_wikidata_id and wikidata_metadata:

        wikidata_fields = []

        for item in wikidata_metadata:

            if item["name"] == "label":
                wikidata_fields.append("labels")

            elif item["name"] == "description":
                wikidata_fields.append("descriptions")

        if wikidata_fields:

            print("\nWikidata interoperability:")
            print("Connecting to Wikidata entity:", entity_wikidata_id)

            params = {"action": "wbgetentities", "ids": entity_wikidata_id, "props": "|".join(wikidata_fields), "languages": "en", "format": "json"}

            api_url = ("https://www.wikidata.org/w/api.php?" + urlencode(params))

            request = Request(api_url, headers={ "User-Agent": "DigitalRubberOntologyPrototype/1.0"})

            with urlopen(request, timeout=20) as response:

                wikidata_data = json.loads(response.read().decode("utf-8"))

            entity_data = wikidata_data.get("entities", {}).get(entity_wikidata_id, {})

            print("\nRetrieved Wikidata metadata:")

            for item in wikidata_metadata:

                language = item.get("language", "en")

                if item["name"] == "label":

                    label_value = entity_data.get("labels", {}).get(language, {}).get("value")

                    print("Wikidata label:", label_value)

                elif item["name"] == "description":

                    description_value = entity_data.get("descriptions", {}).get(language, {}).get("value")

                    print("Wikidata description:", description_value)

        else:

            print("No supported Wikidata metadata fields configured.")

    elif not entity_wikidata_id:

        print("\nNo Wikidata ID retrieved for the matched entity.")

    else:

        print("\nNo default Wikidata metadata configured.")

    print("\nSection 10 completed.")


    # Linearize retrieved information for MAT AI

    linearized_output = f"""
QUESTION: {question}

Digital Rubber answer: "{metadata_value if metadata_value is not None else 'Not available'}"
Digital Rubber label: "{entity_label_value if entity_label_value is not None else 'Not available'}"

Additional Wikidata information:
Label: "{label_value if label_value is not None else 'Not available'}"
Description: "{description_value if description_value is not None else 'Not available'}"

INSTRUCTION TO MAT AI: Use the Digital Rubber answer as the primary answer and include the Wikidata label and description as additional information in your final answer."""

    print("\nLinearized output for MAT AI:")
    print(linearized_output)

print("\nExperiment completed.")
