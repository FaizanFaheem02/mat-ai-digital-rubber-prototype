

import json

import yaml

import rdflib

import logging
import re
from pathlib import Path



from gliner import GLiNER

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util



from rdflib import Graph, RDF, RDFS, OWL, URIRef, XSD



# 1. CONFIGURATION


GLINER_MODEL_NAME = "knowledgator/gliner-decoder-small-v1.0"

SENTENCE_TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MIN_ENTITY_SIMILARITY = 0.75
MIN_PROPERTY_SIMILARITY = 0.35



PROJECT_DIR = Path(__file__).resolve().parent

QWEN_MODEL_PATH = str(PROJECT_DIR / "models" / "qwen" / "qwen2.5-3b-instruct-q4_k_m.gguf")


ONTOLOGY_PATH = "digitrubber-edit.owl"

CONFIG_PATH = "config.yaml"


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# 2. RDFLIB CONFIGURATION


for dt in (XSD.date, XSD.dateTime, XSD.decimal):

    rdflib.term._toPythonMapping.pop(dt, None)


# 3. LOAD GLiNER


logging.info("Loading GLiNER Decoder...")


entity_model = GLiNER.from_pretrained(GLINER_MODEL_NAME)


logging.info("GLiNER Decoder loaded successfully.")


# 4. LOAD QWEN

logging.info("Loading Qwen2.5-3B...")

qwen_model = Llama(model_path=QWEN_MODEL_PATH, n_ctx=2048, n_threads=8, verbose=False)



logging.info("Qwen2.5-3B loaded successfully.")
logging.info("Loading Sentence Transformer...")
relationship_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
logging.info("Sentence Transformer loaded successfully.")
property_embedding_cache = {}


# 5. LOAD DIGITAL RUBBER ONTOLOGY


graph = Graph()


graph.parse(ONTOLOGY_PATH, format="xml")


logging.info("Digital Rubber ontology loaded successfully.")



LABEL = RDFS.label


SYNONYM = URIRef("http://purl.obolibrary.org/obo/IAO_0000118")


DATABASE_CROSS_REFERENCE = URIRef("http://www.geneontology.org/formats/oboInOwl#hasDbXref")


# 6. LOAD YAML CONFIGURATION


with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:

    config = yaml.safe_load(config_file)

# 7. GLiNER ENTITY EXTRACTION


def extract_entities(question):



    return entity_model.predict_entities(question, labels=["label"], threshold=0.3, num_gen_sequences=1)


# 8. QWEN RELATIONSHIP EXTRACTION


def extract_relationships(question, extracted_entities):

    entity_list = [
        item["text"]
        for item in extracted_entities
    ]

    prompt = f"""
You are a scientific relationship extraction system
specialized in rubber science and materials research.

You receive an original research question and entities
already extracted by GLiNER Decoder.

ORIGINAL QUESTION:
{question}

ENTITIES EXTRACTED BY GLiNER:
{json.dumps(entity_list, ensure_ascii=False)}

TASK:

Extract the relationship or property requested in the original question for each supplied entity.

INSTRUCTIONS:

1. Analyse the original question to identify the relationship or property being requested.
2. Return only the relationships, not the complete question or sentence.
3. Do not include the entity name in the relationship.
4. Remove unnecessary grammatical words when they are not part of the relationship meaning.
5. Preserve the original scientific meaning.
6. Use concise relationship suitable for semantic matching against ontology property labels.
7. Do not restrict relationships to predefined categories or a fixed vocabulary.
8. Do not generate answers.
9. Do not invent relationships.
10. If multiple relationships are requested, return each separately.
11. If no relationship can be identified, return an empty list.

OUTPUT FORMAT:

Return ONLY valid JSON:

{{"relationships": [{{"entity": "entity extracted by GLiNER", "relationship": "extracted relationship only" }}]}}
"""

    response = qwen_model.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise scientific relationship extraction model. "
                    "Extract all requested relationships from the question "
                    "as one independent list and return valid JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"}
    )

    content = response["choices"][0]["message"]["content"]
    logging.info("Qwen raw response: %s", content)

    try:
        result = json.loads(content)
        relationships = result.get("relationships", [])

        if not isinstance(relationships, list):
            return []

        validated_relationships = []
        seen_relationships = set()

        for item in relationships:

            if isinstance(item, dict):
                relationship = item.get("relationship", "")
            elif isinstance(item, str):
                relationship = item
            else:
                continue

            if not isinstance(relationship, str):
                continue

            relationship = relationship.strip()
            relationship_key = relationship.casefold()

            if relationship and relationship_key not in seen_relationships:
                validated_relationships.append(relationship)
                seen_relationships.add(relationship_key)

        return validated_relationships

    except (json.JSONDecodeError, AttributeError, TypeError):
        logging.exception("Qwen returned invalid JSON.")
        return []


# 9. ONTOLOGY ENTITY NORMALIZATION


def normalize_entity(entity):
    """Match an extracted entity to the closest ontology label or synonym."""
    if not entity or not entity.strip():
        return None, None, None

    candidates = []
    seen = set()

    for subject, label in graph.subject_objects(LABEL):
        text = str(label).strip()
        key = (str(subject), text.lower())
        if text and key not in seen:
            candidates.append({"uri": subject, "text": text, "source": "label"})
            seen.add(key)

    for subject, synonym in graph.subject_objects(SYNONYM):
        text = str(synonym).strip()
        key = (str(subject), text.lower())
        if text and key not in seen:
            candidates.append({"uri": subject, "text": text, "source": "synonym"})
            seen.add(key)

    if not candidates:
        return None, None, None

    entity_key = entity.strip().casefold()
    for candidate in candidates:
        if candidate["text"].casefold() == entity_key:
            return candidate["text"], candidate["uri"], candidate["source"]

    candidate_texts = [item["text"] for item in candidates]
    embeddings = relationship_model.encode(
        candidate_texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    query_embedding = relationship_model.encode(
        entity.strip(),
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    scores = util.cos_sim(query_embedding, embeddings)[0]
    best_index = int(scores.argmax().item())
    best = candidates[best_index]
    best_score = float(scores[best_index].item())

    logging.info(
        "Best ontology entity candidate: '%s' -> '%s' (%s; cosine similarity: %.4f)",
        entity, best["text"], best["source"], best_score
    )
    if best_score < MIN_ENTITY_SIMILARITY:
        logging.warning(
            "Entity match rejected: similarity %.4f is below the minimum %.2f.",
            best_score, MIN_ENTITY_SIMILARITY
        )
        return None, None, None

    return best["text"], best["uri"], "semantic_" + best["source"]


# 10. RETRIEVE ONTOLOGY SYNONYMS



def retrieve_synonyms(entity_uri):



    synonyms = []



    for synonym in graph.objects(entity_uri, SYNONYM):



        synonyms.append(str(synonym))



    return synonyms


# 11. MATCH ONTOLOGY PROPERTY GLOBALLY


def match_ontology_property(relationship):
    
    if not isinstance(relationship, str) or not relationship.strip():
        return None

    relationship_text = relationship.strip()
    relationship_key = relationship_text.casefold()

    excluded_predicates = {
        RDF.type,
        RDFS.label,
        RDFS.comment,
        RDFS.subClassOf,
        RDFS.subPropertyOf,
        OWL.equivalentClass,
        OWL.disjointWith,
        OWL.inverseOf,
        OWL.imports,
    }

    cache_key = "global_ontology_property_vocabulary"

    if cache_key not in property_embedding_cache:
        candidates = []
        seen_predicates = set()

        for predicate in set(graph.predicates()):
            if not isinstance(predicate, URIRef) or predicate in excluded_predicates:
                continue
            if predicate in seen_predicates:
                continue
            seen_predicates.add(predicate)

            predicate_uri = str(predicate)
            local_name = predicate_uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1].strip()

            label_values = [
                str(value).strip()
                for value in graph.objects(predicate, LABEL)
                if str(value).strip()
            ]
            synonym_values = [
                str(value).strip()
                for value in graph.objects(predicate, SYNONYM)
                if str(value).strip()
            ]

            aliases = []
            for alias in label_values + synonym_values + ([local_name] if local_name else []):
                if alias and alias.casefold() not in {item.casefold() for item in aliases}:
                    aliases.append(alias)

            if not aliases:
                continue

            display_label = label_values[0] if label_values else (
                synonym_values[0] if synonym_values else local_name
            )

            candidates.append({
                "uri": predicate,
                "label": display_label,
                "aliases": aliases,
            })

        # Exact label/synonym/local-name matches are checked before embeddings.
        property_embedding_cache[cache_key] = candidates

    candidates = property_embedding_cache[cache_key]

    if not candidates:
        logging.warning("No global ontology property candidates were found.")
        return None

    # First: exact match against labels, synonyms, or the predicate local name.
    for candidate in candidates:
        for alias in candidate["aliases"]:
            if relationship_key == alias.casefold():
                logging.info(
                    "Exact global ontology property match: '%s' -> '%s'",
                    relationship_text,
                    candidate["label"],
                )
                return {
                    "uri": str(candidate["uri"]),
                    "label": candidate["label"],
                }

    alias_records = []
    for candidate in candidates:
        for alias in candidate["aliases"]:
            alias_records.append((candidate, alias))

    alias_embeddings = relationship_model.encode(
        [alias for _, alias in alias_records],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    relationship_embedding = relationship_model.encode(
        relationship_text,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    similarity_scores = util.cos_sim(
        relationship_embedding,
        alias_embeddings,
    )[0]

    best_index = int(similarity_scores.argmax().item())
    best_score = float(similarity_scores[best_index].item())
    best_candidate, best_alias = alias_records[best_index]

    logging.info(
        "Best global ontology property candidate: '%s' -> '%s' "
        "(matched alias: '%s'; cosine similarity: %.4f)",
        relationship_text,
        best_candidate["label"],
        best_alias,
        best_score,
    )

    if best_score < MIN_PROPERTY_SIMILARITY:
        logging.warning(
            "Property match rejected: similarity %.4f is below the minimum %.2f.",
            best_score,
            MIN_PROPERTY_SIMILARITY,
        )
        return None

    return {
        "uri": str(best_candidate["uri"]),
        "label": best_candidate["label"],
    }


# 12. RETRIEVE PROPERTY VALUES USING SPARQL



def retrieve_metadata(entity_uri, property_uri):



    values = []



    sparql_template = """

    SELECT ?value

    WHERE {

        ENTITY_URI PROPERTY_URI ?value .

    }

    """



    query = (sparql_template.replace("ENTITY_URI", URIRef(entity_uri).n3()).replace("PROPERTY_URI", URIRef(property_uri).n3()))


    results = graph.query(query)

    for result in results:


        value = result.value



        value_label = graph.value(value, LABEL)



        if value_label:

            values.append(str(value_label))



        else:

            values.append(str(value))



    return values



# 13. RETRIEVE CONFIGURED ENTITY METADATA



def retrieve_entity_metadata(entity_uri):



    metadata = {}



    entity_metadata = [

        item

        for item in config.get("entity_metadata", [])

        if item.get("enabled", False)

        and item.get("default", False)

    ]



    entity_query_parts = []

    selected_variables = []



    for item in entity_metadata:



        if item["name"] == "label":



            entity_query_parts.append(""" OPTIONAL {ENTITY_URI rdfs:label ?entity_label .FILTER(LANG(?entity_label) = "LANGUAGE")}""".replace("ENTITY_URI", URIRef(entity_uri).n3()).replace("LANGUAGE", item.get("language", "en")))


            selected_variables.append("?entity_label")



        elif item["name"] == "wikidata_id":



            entity_query_parts.append("""OPTIONAL {

                    ENTITY_URI oboInOwl:hasDbXref

                    ?entity_wikidata_id .

                }

                """.replace(

                    "ENTITY_URI",

                    URIRef(entity_uri).n3()

                )

            )



            selected_variables.append("?entity_wikidata_id")



    if not entity_query_parts:

        return metadata



    entity_sparql_template = """

    PREFIX rdfs:

    <http://www.w3.org/2000/01/rdf-schema#>



    PREFIX oboInOwl:

    <http://www.geneontology.org/formats/oboInOwl#>



    SELECT SELECTED_VARIABLES

    WHERE {

        METADATA_PATTERNS

    }

    """



    entity_query = (

        entity_sparql_template

        .replace(

            "SELECTED_VARIABLES",

            " ".join(selected_variables)

        )

        .replace(

            "METADATA_PATTERNS",

            "\n".join(entity_query_parts)

        )

    )



    entity_results = graph.query(entity_query)



    for result in entity_results:



        if result.entity_label:



            metadata["label"] = str(result.entity_label)



        if result.entity_wikidata_id:



            metadata["wikidata_id"] = (

                str(result.entity_wikidata_id).strip()

            )



    return metadata



# 15. PROCESS ONE QUESTION AND BUILD JSON



def process_question(question):
    output = {"classes": {}, "properties": {}}

    logging.info("=" * 70)
    logging.info("STARTING QUESTION: %s", question)
    logging.info("=" * 70)

    logging.info("STEP 1/5: Extracting entities with GLiNER...")
    extracted_entities = extract_entities(question)
    entity_list = [
        item["text"].strip()
        for item in extracted_entities
        if isinstance(item.get("text"), str) and item["text"].strip()
    ]

    if entity_list:
        for item in extracted_entities:
            logging.info(
                "GLiNER entity: %s (confidence: %.4f)",
                item.get("text", ""),
                float(item.get("score", 0.0)),
            )
    else:
        logging.warning("GLiNER did not extract any entities.")
        return output

    logging.info("GLiNER entity list: %s", entity_list)

    logging.info("STEP 2/5: Extracting relationships with Qwen...")
    relationship_list = extract_relationships(question, extracted_entities)

    if relationship_list:
        for relationship in relationship_list:
            logging.info("Qwen relationship: %s", relationship)
    else:
        logging.warning("Qwen did not extract any relationships.")

    logging.info("Qwen relationship list: %s", relationship_list)

    # STEP 3: Match and retrieve metadata for each entity independently.
    for original_entity in entity_list:
        logging.info("=" * 60)
        logging.info("PROCESSING ENTITY: %s", original_entity)

        normalized_entity, entity_uri, match_method = normalize_entity(
            original_entity
        )
        class_metadata = {}

        if entity_uri:
            logging.info(
                "STEP 3/5: Ontology entity matched: '%s' -> '%s' (method: %s)",
                original_entity,
                normalized_entity,
                match_method,
            )

            class_metadata.update(retrieve_entity_metadata(entity_uri))
            synonyms = retrieve_synonyms(entity_uri)
            if synonyms:
                class_metadata["synonym"] = synonyms
        else:
            logging.warning(
                "STEP 3/5: Entity not found in DigiRubber ontology: %s",
                original_entity,
            )

        output["classes"][original_entity] = class_metadata

    # STEP 4: Match each extracted relationship independently against
    # the global DigiRubber property vocabulary.
    for relationship in relationship_list:
        logging.info(
            "Matching Qwen relationship '%s' against all global ontology properties.",
            relationship,
        )
        output["properties"][relationship] = {}
        property_match = match_ontology_property(relationship)
        if property_match:
            output["properties"][relationship] = property_match

    logging.info("Question processing completed.")
    return output


# 16. TEST QUESTIONS



questions = ["What should we change in the material structure (polymer Mooney, MSR, cis/vinyl/styrene content) in order to increase rebound, but at the same time, have good processability?"]


# 17. EXECUTE PIPELINE AND OUTPUT JSON


try:



    for question in questions:



        result = process_question(question)



        # Print only valid JSON to standard output.

        print(

            json.dumps(

                result,

                indent=4,

                ensure_ascii=False

            )

        )



finally:



    qwen_model.close()



    logging.info("Qwen model resources released.")