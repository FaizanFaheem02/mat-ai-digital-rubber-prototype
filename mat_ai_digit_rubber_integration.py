

import json

import rdflib

import logging
import re
from pathlib import Path



from gliner import GLiNER

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util



from rdflib import Graph, RDF, RDFS, OWL, URIRef, XSD
import requests



# 1. CONFIGURATION


GLINER_MODEL_NAME = "knowledgator/gliner-decoder-small-v1.0"

SENTENCE_TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MIN_ENTITY_SIMILARITY = 0.75
MIN_PROPERTY_SIMILARITY = 0.35



PROJECT_DIR = Path(__file__).resolve().parent

QWEN_MODEL_PATH = str(PROJECT_DIR / "models" / "qwen" / "qwen2.5-3b-instruct-q4_k_m.gguf")


ONTOLOGY_PATH = PROJECT_DIR / "digitrubber-edit.owl"



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


WIKIDATA_ID = URIRef("http://purl.obolibrary.org/obo/DIGITRUBBER_1000063")


# 6. GLiNER ENTITY EXTRACTION


def extract_entities(question):

    return entity_model.predict_entities(question, labels=["label"], threshold=0.3, num_gen_sequences=1)


# 7. QWEN RELATIONSHIP EXTRACTION


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



# 8. ONTOLOGY ENTITY NORMALIZATION


def normalize_entity(entity):
    """Match an extracted entity to the closest ontology label or synonym."""
    if not entity or not entity.strip():
        return None, None, None

    candidates = []
    seen = set()

    class_types = {OWL.Class, RDFS.Class}
    ontology_classes = {
        subject
        for class_type in class_types
        for subject in graph.subjects(RDF.type, class_type)
        if isinstance(subject, URIRef)
    }

    for subject in ontology_classes:
        for label in graph.objects(subject, LABEL):
            text = str(label).strip()
            key = (str(subject), text.casefold())
            if text and key not in seen:
                candidates.append({"uri": subject, "text": text, "source": "label"})
                seen.add(key)

        for synonym in graph.objects(subject, SYNONYM):
            text = str(synonym).strip()
            key = (str(subject), text.casefold())
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


# 9. RETRIEVE ONTOLOGY SYNONYMS



def retrieve_synonyms(entity_uri):



    synonyms = []



    for synonym in graph.objects(entity_uri, SYNONYM):



        synonyms.append(str(synonym))



    return synonyms


# 10. MATCH ONTOLOGY PROPERTY GLOBALLY


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

        property_types = {
            RDF.Property,
            OWL.ObjectProperty,
            OWL.DatatypeProperty,
            OWL.AnnotationProperty,
            OWL.FunctionalProperty,
            OWL.InverseFunctionalProperty,
            OWL.TransitiveProperty,
            OWL.SymmetricProperty,
            OWL.AsymmetricProperty,
            OWL.ReflexiveProperty,
            OWL.IrreflexiveProperty,
        }
        declared_properties = {
            subject
            for property_type in property_types
            for subject in graph.subjects(RDF.type, property_type)
            if isinstance(subject, URIRef)
        }

        for predicate in declared_properties:
            if predicate in excluded_predicates:
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

        property_embedding_cache[cache_key] = candidates

    candidates = property_embedding_cache[cache_key]

    if not candidates:
        logging.warning("No global ontology property candidates were found.")
        return None

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


# 11. RETRIEVE PROPERTY VALUES USING SPARQL



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



# 12. RETRIEVE WIKIDATA ID INDEPENDENTLY

def retrieve_wikidata_id(entity_uri):
    """Retrieve the Wikidata ID annotation for an ontology entity."""
    entity_uri = URIRef(entity_uri)
    wikidata_id = graph.value(entity_uri, WIKIDATA_ID)

    if wikidata_id is None:
        return None

    return str(wikidata_id).strip()

# 13. RETRIEVE METADATA FROM WIKIDATA

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"



def retrieve_wikidata_metadata(wikidata_id):
    """Retrieve the IDs and English labels of properties used by a Wikidata entity.

    Property values are deliberately not retrieved. MAT AI can decide later
    which properties are relevant and whether their values need to be fetched.
    """

    if not wikidata_id:
        return {"properties": []}

    wikidata_id = str(wikidata_id).strip().rstrip("/").split("/")[-1]

    if not re.fullmatch(r"Q[1-9]\d*", wikidata_id):
        logging.warning("Invalid Wikidata ID: %s", wikidata_id)
        return {"properties": []}

    sparql_query = """
    SELECT DISTINCT ?propertyId ?propertyLabel
    WHERE {
        BIND(wd:WIKIDATA_ID AS ?entity)

        # Find direct claims made by this entity.
        ?entity ?claimPredicate ?value .

        # Map each direct-claim predicate to its Wikidata property entity.
        ?property wikibase:directClaim ?claimPredicate .

        # Keep English property labels only.
        ?property rdfs:label ?propertyLabel .
        FILTER(LANG(?propertyLabel) = "en")

        BIND(
            STRAFTER(STR(?property), "http://www.wikidata.org/entity/")
            AS ?propertyId
        )
    }
    ORDER BY ?propertyId
    """

    query = sparql_query.replace("WIKIDATA_ID", wikidata_id)

    try:
        response = requests.get(
            WIKIDATA_ENDPOINT,
            params={
                "query": query,
                "format": "json"
            },
            headers={
                "User-Agent": "DigiRubber/1.0 (research application)"
            },
            timeout=30
        )

        response.raise_for_status()
        bindings = response.json()["results"]["bindings"]

        properties = []
        seen_property_ids = set()

        for item in bindings:
            property_id = item.get("propertyId", {}).get("value", "").strip()
            property_label = item.get("propertyLabel", {}).get("value", "").strip()

            if not property_id or not property_label:
                continue

            if not re.fullmatch(r"P[1-9]\d*", property_id):
                continue

            if property_id in seen_property_ids:
                continue

            properties.append({
                "property_id": property_id,
                "property_label": property_label
            })
            seen_property_ids.add(property_id)

        metadata = {"properties": properties}

        logging.info(
            "Retrieved Wikidata property catalogue for %s: %d properties",
            wikidata_id,
            len(properties)
        )

        return metadata

    except requests.RequestException:
        logging.exception(
            "Wikidata request failed for %s",
            wikidata_id
        )
        return {"properties": []}

    except (ValueError, KeyError, TypeError):
        logging.exception(
            "Could not process Wikidata response for %s",
            wikidata_id
        )
        return {"properties": []}



# 14. RETRIEVE CONFIGURED ENTITY METADATA

def retrieve_entity_metadata(entity_uri):
    metadata = {}

    entity_uri = URIRef(entity_uri)

    annotation_property_type = URIRef(
        "http://www.w3.org/2002/07/owl#AnnotationProperty"
    )

    annotation_values = {}

    for predicate, value in graph.predicate_objects(entity_uri):

        if predicate == WIKIDATA_ID:
            continue

        if (predicate, RDF.type, annotation_property_type) not in graph:
            continue

        property_labels = list(graph.objects(predicate, RDFS.label))

        english_labels = [
            label for label in property_labels
            if getattr(label, "language", None) == "en"
        ]

        if english_labels:
            property_name = str(english_labels[0])
        elif property_labels:
            property_name = str(property_labels[0])
        else:
            property_name = predicate.split("/")[-1].split("#")[-1]

        key = "".join(
            character.lower() if character.isalnum() else "_"
            for character in property_name
        ).strip("_")

        while "__" in key:
            key = key.replace("__", "_")

        language = getattr(value, "language", None)
        if language and language.lower() != "en":
            key = f"{key}_{language.lower()}"

        annotation_values.setdefault(key, []).append(str(value))

    for key, values in annotation_values.items():
        metadata[key] = values[0] if len(values) == 1 else values

    return metadata



# 15. PROCESS ONE QUESTION AND BUILD JSON



def process_question(question):
    """
    Process the question against the local Digital Rubber ontology only.

    Returns:
        ontology_output: JSON-ready local ontology metadata.
        matched_entities: Mapping of extracted entity text to matched ontology URI.
    """
    ontology_output = {"classes": {}, "properties": {}}

    logging.info("=" * 70)
    logging.info("STARTING QUESTION: %s", question)
    logging.info("=" * 70)

    logging.info("STEP 1/4: Extracting entities with GLiNER...")
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
        return ontology_output, {}

    logging.info("GLiNER entity list: %s", entity_list)

    logging.info("STEP 2/4: Extracting relationships with Qwen...")
    relationship_list = extract_relationships(question, extracted_entities)

    if relationship_list:
        for relationship in relationship_list:
            logging.info("Qwen relationship: %s", relationship)
    else:
        logging.warning("Qwen did not extract any relationships.")

    logging.info("Qwen relationship list: %s", relationship_list)

    matched_entities = {}

    for original_entity in entity_list:
        logging.info("=" * 60)
        logging.info("PROCESSING ENTITY: %s", original_entity)

        normalized_entity, entity_uri, match_method = normalize_entity(
            original_entity
        )
        class_metadata = {}

        if entity_uri:
            matched_entities[original_entity] = entity_uri

            logging.info(
                "Ontology entity matched: '%s' -> '%s' (method: %s)",
                original_entity,
                normalized_entity,
                match_method,
            )

            class_metadata.update(retrieve_entity_metadata(entity_uri))

            # The Wikidata ID is an annotation stored in the local ontology.
            # Keep the mapping ID here, but do not retrieve Wikidata metadata here.
            wikidata_id = retrieve_wikidata_id(entity_uri)
            if wikidata_id:
                class_metadata["wikidata_id"] = wikidata_id

        else:
            logging.warning(
                "Entity not found in Digital Rubber ontology: %s",
                original_entity,
            )

        ontology_output["classes"][original_entity] = class_metadata

    for relationship in relationship_list:
        logging.info(
            "Matching Qwen relationship '%s' against ontology properties.",
            relationship,
        )

        ontology_output["properties"][relationship] = {}

        property_match = match_ontology_property(relationship)

        if property_match:
            property_metadata = property_match.copy()
            property_values = {}

            for entity_name, entity_uri in matched_entities.items():
                values = retrieve_metadata(
                    entity_uri,
                    property_match["uri"],
                )

                if values:
                    property_values[entity_name] = values

            if property_values:
                property_metadata["values"] = property_values

            ontology_output["properties"][relationship] = property_metadata

    logging.info("Local ontology processing completed.")
    return ontology_output, matched_entities


def process_wikidata(matched_entities):
    """
    Retrieve Wikidata metadata independently of process_question().

    Args:
        matched_entities: Mapping of extracted entity text to local ontology URI.

    Returns:
        A separate JSON-ready object containing Wikidata results.
    """
    wikidata_output = {"entities": {}}

    for entity_name, entity_uri in matched_entities.items():
        wikidata_id = retrieve_wikidata_id(entity_uri)

        if not wikidata_id:
            logging.info(
                "No Wikidata ID is assigned to matched entity '%s'. Skipping.",
                entity_name,
            )
            continue

        logging.info(
            "Independently retrieving Wikidata properties for %s (%s).",
            entity_name,
            wikidata_id,
        )

        wikidata_metadata = retrieve_wikidata_metadata(wikidata_id)

        wikidata_output["entities"][entity_name] = {
            "wikidata_id": wikidata_id,
            "label": entity_name,
            "properties": wikidata_metadata.get("properties", []),
        }

    logging.info("Wikidata processing completed.")
    return wikidata_output



# 16. TEST QUESTIONS AND SCRIPT EXECUTION

if __name__ == "__main__":

    questions = [
        "Who is the contributor to silane?"
    ]

    try:
        for index, question in enumerate(questions, start=1):

            ontology_result, matched_entities = process_question(question)

            wikidata_result = process_wikidata(matched_entities)

            ontology_file = PROJECT_DIR / f"ontology_output_{index}.json"
            wikidata_file = PROJECT_DIR / f"wikidata_output_{index}.json"

            ontology_file.write_text(
                json.dumps(ontology_result, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
            wikidata_file.write_text(
                json.dumps(wikidata_result, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )

            print("\\n== LOCAL ONTOLOGY JSON ==\\n")
            print(json.dumps(ontology_result, indent=4, ensure_ascii=False))

            print("\\n== WIKIDATA JSON ==\\n")
            print(json.dumps(wikidata_result, indent=4, ensure_ascii=False))

            logging.info("Saved local ontology JSON to: %s", ontology_file)
            logging.info("Saved Wikidata JSON to: %s", wikidata_file)

    finally:
        qwen_model.close()
        logging.info("Qwen model resources released.")
