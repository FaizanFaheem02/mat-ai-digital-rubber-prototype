# MAT AI – Digital Rubber Ontology Prototype

## Overview

This project contains a Python prototype that enables MAT AI to retrieve relevant information from the Digital Rubber Ontology based on natural-language questions. It identifies entities and relationships, retrieves corresponding information from the Digital Rubber Ontology using SPARQL, and prepares a linearized output to provide context for MAT AI.

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