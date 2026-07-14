# Explainable AI Candidate–Job Matching Prototype

An explainable AI decision-support prototype that matches candidate CVs with job descriptions using Sentence-BERT semantic similarity and structured skill matching.

## Project Overview

Traditional Applicant Tracking Systems often depend on exact keyword matching. This may overlook suitable candidates who describe equivalent skills using different terminology.

This prototype combines semantic understanding, structured skill matching and transparent explanations to support initial recruitment screening.

## Main Features

- Sentence-BERT semantic similarity
- Cosine similarity scoring
- Structured skill matching
- Weighted candidate scoring
- Explainable candidate recommendations
- High, Medium and Low suitability classification
- BM25 baseline comparison
- Protected-attribute removal
- Fairness stability testing
- PDF and DOCX CV uploading
- New CV ranking against existing candidates
- Interactive Streamlit interface

## Prototype Scoring

The final candidate score is calculated using:

- 70% Sentence-BERT semantic similarity
- 30% structured skill matching

These weights are prototype settings and may be adjusted following further evaluation.

## Dataset

The evaluation dataset contains 20 anonymised synthetic CVs:

- 7 High-suitability candidates
- 7 Medium-suitability candidates
- 6 Low-suitability candidates

Synthetic data were used to support controlled testing, reproducibility and applicant privacy.

## Technology Stack

- Python
- Streamlit
- Sentence Transformers
- Sentence-BERT
- Pandas
- NumPy
- Scikit-learn
- Rank-BM25
- PyPDF
- python-docx

## Repository Structure

    explainable-ai-cv-matcher/
    ├── app.py
    ├── README.md
    ├── requirements.txt
    ├── ground_truth_labels.csv
    ├── software_engineer_job_description.txt
    ├── synthetic_cvs/
    └── outputs/

## Installation

Clone the repository:

    git clone https://github.com/rahikhan7879-source/explainable-ai-cv-matcher.git

Open the project directory:

    cd explainable-ai-cv-matcher

Install the required libraries:

    pip install -r requirements.txt

## Run the Application

Start the Streamlit application:

    streamlit run app.py

The application will normally open at:

    http://localhost:8501

## Application Sections

### Existing 20-CV Ranking

Displays the final ranking of the synthetic candidates, including:

- Semantic score
- Skill-match score
- Final score
- Predicted suitability
- Matched skills
- Missing skills
- Explainable recommendation

### Check a New CV

A recruiter can:

1. Enter or modify a job description.
2. Define the required skills.
3. Upload a PDF or DOCX CV.
4. Analyse the CV.
5. View the final score and suitability.
6. Review matched and missing skills.
7. See the CV's position among existing candidates.

## Evaluation

The proposed approach is compared with a BM25 keyword-matching baseline using:

- NDCG@5
- NDCG@10
- Precision@5
- Recall@5
- Precision@10
- Recall@10

## Fairness and Explainability

Clearly labelled protected information is removed before scoring, including:

- Name
- Age
- Gender
- Nationality
- Religion
- Ethnicity
- Marital status
- Date of birth

The system provides explanations showing the semantic score, skill score, matched skills, missing skills and final recommendation.

## Ethical Notice

This prototype is intended only for recruitment decision support.

It must not:

- Make final hiring decisions
- Automatically reject applicants
- Replace human recruiter judgement
- Score candidates using protected characteristics

Final employment decisions must remain under meaningful human control.

## Limitations

- The evaluation uses a small synthetic dataset.
- Skill extraction is primarily rule-based.
- Image-based scanned PDFs may require OCR.
- The scoring weights require wider validation.
- Fairness testing does not prove the absence of every possible bias.
- The prototype requires further security, legal and usability testing before production use.

## Academic Purpose

Developed as part of an MSc Artificial Intelligence research project on explainable AI-based candidate–job matching.

## Author

Aziz ul Hassan  
MSc Artificial Intelligence  
University of Roehampton London
