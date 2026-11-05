# MAT AI – Digital Rubber Ontology Prototype

## Overview

This project contains a Python prototype that enables MAT AI to retrieve relevant information from the Digital Rubber Ontology based on natural-language questions.

The system processes research questions using the following components:

- **GLiNER Decoder:** Extracts entities from natural-language questions.
- **Qwen2.5-3B:** Extracts relationships and properties requested in the questions.
- **Sentence Transformer:** Performs semantic similarity matching between extracted entities and relationships and the concepts and properties available in the ontology.
- **DigiRubber Ontology:** Provides the scientific knowledge base for entity matching and information retrieval.
- **SPARQL:** Retrieves corresponding information from the ontology.

The extracted entities and relationships are processed independently. The system matches the extracted entities to ontology concepts, compares the extracted relationships with available ontology properties, and retrieves corresponding values.

The results are returned as structured JSON, which can be used to provide relevant context to MAT AI.


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/FaizanFaheem02/mat-ai-digital-rubber-prototype.git

cd mat-ai-digital-rubber-prototype
```

### 2. Set up Python

Python 3.11.9 is recommended for the current implementation.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Configure the Qwen model

The prototype uses the Qwen2.5-3B-Instruct model in GGUF format through `llama-cpp-python`.

Download or obtain the model file:

`qwen2.5-3b-instruct-q4_k_m.gguf`

Update the `QWEN_MODEL_PATH` variable in `ontology_retrieval.py` to point to the location of the downloaded model file.

### 5. Configure the ontology

Ensure that the following files are available in the project directory:

- `digitrubber-edit.owl`
- `config.yaml`

The configuration file defines the entity metadata used by the retrieval pipeline.

## Running the Prototype

Run the following command:

```bash
python ontology_retrieval.py
```