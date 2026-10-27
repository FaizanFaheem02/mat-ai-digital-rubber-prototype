# MAT AI – Digital Rubber Ontology Prototype

## Overview

This project contains a Python prototype for integrating the MAT AI chatbot with the Digital Rubber Ontology. It uses spaCy for entity and relationship extraction, Sentence Transformers for semantic matching, and SPARQL queries to retrieve relevant ontology information. It also retrieves configured metadata, including Wikidata identifiers, and prepares a linearised output for MAT AI.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/FaizanFaheem02/mat-ai-digital-rubber-prototype.git
   cd mat-ai-digital-rubber-prototype
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows:

   ```powershell
   .venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the spaCy English language model:

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. Run the prototype:

   ```bash
   python ontology_retrieval.py
   ```