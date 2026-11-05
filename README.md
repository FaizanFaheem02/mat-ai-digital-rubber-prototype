# MAT AI – Digital Rubber Ontology Prototype

## 1. Overview

This project contains a Python prototype designed to integrate the MAT AI partner chatbot with the DigiRubber ontology. It processes natural-language questions, extracts entities and relationships, matches them against the ontology, and returns structured JSON output for MAT AI.

## 2. Technologies Used

- **GLiNER Decoder:** Entity extraction.
- **Qwen2.5-3B-Instruct:** Relationship extraction.
- **Sentence Transformers:** Semantic similarity matching.
- **RDFLib:** Loading and querying the DigiRubber ontology.
- **SPARQL:** Ontology data retrieval.

## 3. Installation

Clone the repository:

```bash
git clone https://github.com/FaizanFaheem02/mat-ai-digital-rubber-prototype.git
cd mat-ai-digital-rubber-prototype
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 4. Configure the Qwen Model

The prototype uses the Qwen2.5-3B-Instruct model in GGUF format through `llama-cpp-python`.

Download or obtain the following model file:

`qwen2.5-3b-instruct-q4_k_m.gguf`

Create the following directory structure inside the project folder:

```text
models/
└── qwen/
    └── qwen2.5-3b-instruct-q4_k_m.gguf
```

Place the downloaded model file inside the `models/qwen/` directory. The Python script automatically locates the model from this directory, so no manual path configuration is required.

**Note:** Please note that GGUF model file is not included in the GitHub repository and must be downloaded separately.

## 5. Run the Prototype

Run the main Python script:

```bash
python mat_ai_digit_rubber_integration.py
```