
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Explainable AI Candidate Matcher",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_FOLDER = Path("/content/cv_job_matcher")
OUTPUT_FOLDER = BASE_FOLDER / "outputs"

JOB_DESCRIPTION_FILE = (
    BASE_FOLDER /
    "software_engineer_job_description.txt"
)

FINAL_RESULTS_FILE = (
    OUTPUT_FOLDER /
    "final_explainable_candidate_ranking.csv"
)


# =========================================================
# SYSTEM SETTINGS
# =========================================================

MODEL_NAME = "paraphrase-MiniLM-L3-v2"

SEMANTIC_WEIGHT = 0.70
SKILL_WEIGHT = 0.30


# =========================================================
# REQUIRED SKILLS
# =========================================================

REQUIRED_SKILLS = [
    "Python",
    "SQL",
    "Object-Oriented Programming",
    "REST APIs",
    "Git",
    "Docker",
    "AWS",
    "Machine Learning",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "Software Testing",
    "Agile"
]


SKILL_ALIASES = {
    "Python": [
        "python"
    ],

    "SQL": [
        "sql",
        "mysql",
        "postgresql",
        "structured query language"
    ],

    "Object-Oriented Programming": [
        "object-oriented programming",
        "object oriented programming",
        "oop"
    ],

    "REST APIs": [
        "rest api",
        "rest apis",
        "restful api",
        "restful apis",
        "api development"
    ],

    "Git": [
        "git",
        "github",
        "gitlab",
        "version control"
    ],

    "Docker": [
        "docker",
        "containerisation",
        "containerization",
        "containerised",
        "containerized"
    ],

    "AWS": [
        "aws",
        "amazon web services"
    ],

    "Machine Learning": [
        "machine learning",
        "predictive modelling",
        "predictive modeling"
    ],

    "Pandas": [
        "pandas"
    ],

    "NumPy": [
        "numpy"
    ],

    "Scikit-learn": [
        "scikit-learn",
        "scikit learn",
        "sklearn"
    ],

    "Software Testing": [
        "software testing",
        "unit testing",
        "integration testing",
        "test automation",
        "testing and debugging"
    ],

    "Agile": [
        "agile",
        "scrum",
        "sprint planning"
    ]
}


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )


# =========================================================
# LOAD EXISTING RESULTS
# =========================================================

@st.cache_data
def load_existing_results():
    if FINAL_RESULTS_FILE.exists():
        return pd.read_csv(FINAL_RESULTS_FILE)

    return pd.DataFrame()


# =========================================================
# LOAD DEFAULT JOB DESCRIPTION
# =========================================================

@st.cache_data
def load_default_job_description():
    if JOB_DESCRIPTION_FILE.exists():
        return JOB_DESCRIPTION_FILE.read_text(
            encoding="utf-8"
        )

    return ""


# =========================================================
# TEXT EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):
    uploaded_file.seek(0)

    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        page_text = page.extract_text() or ""

        if page_text.strip():
            pages.append(page_text.strip())

    return "\n".join(pages)


def extract_docx_text(uploaded_file):
    uploaded_file.seek(0)

    document = Document(uploaded_file)

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)


def extract_uploaded_cv_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        text = extract_pdf_text(uploaded_file)

    elif filename.endswith(".docx"):
        text = extract_docx_text(uploaded_file)

    else:
        raise ValueError(
            "Only PDF and DOCX files are supported."
        )

    if len(text.strip()) < 50:
        raise ValueError(
            "The CV contains insufficient readable text. "
            "It may be a scanned image-based PDF."
        )

    return text.strip()


# =========================================================
# PROTECTED-ATTRIBUTE REMOVAL
# =========================================================

PROTECTED_ATTRIBUTE_PATTERNS = [
    r"^\s*name\s*:",
    r"^\s*full name\s*:",
    r"^\s*gender\s*:",
    r"^\s*sex\s*:",
    r"^\s*age\s*:",
    r"^\s*date of birth\s*:",
    r"^\s*dob\s*:",
    r"^\s*nationality\s*:",
    r"^\s*ethnicity\s*:",
    r"^\s*religion\s*:",
    r"^\s*marital status\s*:"
]


def remove_protected_attributes(text):
    remaining_lines = []

    for line in text.splitlines():

        is_protected = any(
            re.match(
                pattern,
                line,
                flags=re.IGNORECASE
            )
            for pattern in PROTECTED_ATTRIBUTE_PATTERNS
        )

        if not is_protected:
            remaining_lines.append(line)

    return "\n".join(remaining_lines).strip()


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"\+?\d[\d\s().-]{7,}\d",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9+#.\-/\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# SKILL MATCHING
# =========================================================

def skill_is_present(cv_text, skill):
    aliases = SKILL_ALIASES.get(
        skill,
        [skill.lower()]
    )

    for alias in aliases:

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(alias.lower())
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, cv_text):
            return True

    return False


def calculate_skill_match(
    cv_text,
    required_skills
):
    matched_skills = []
    missing_skills = []

    for skill in required_skills:

        if skill_is_present(cv_text, skill):
            matched_skills.append(skill)

        else:
            missing_skills.append(skill)

    if required_skills:
        skill_score = (
            len(matched_skills)
            / len(required_skills)
        ) * 100

    else:
        skill_score = 0.0

    return {
        "skill_score": skill_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }


# =========================================================
# SUITABILITY CLASSIFICATION
# =========================================================

def classify_suitability(final_score):
    if final_score >= 70:
        return "High"

    if final_score >= 50:
        return "Medium"

    return "Low"


def create_recommendation(label):
    if label == "High":
        return (
            "Strong candidate for initial recruiter review."
        )

    if label == "Medium":
        return (
            "Potential candidate, but missing requirements "
            "should be reviewed."
        )

    return (
        "Limited match for this role based on the current "
        "job description."
    )


# =========================================================
# SCORE A NEW CV
# =========================================================

def score_new_cv(
    cv_text,
    job_description,
    required_skills
):
    protected_removed_text = (
        remove_protected_attributes(cv_text)
    )

    clean_cv_text = preprocess_text(
        protected_removed_text
    )

    clean_job_description = preprocess_text(
        job_description
    )

    if not clean_cv_text:
        raise ValueError(
            "The CV contains no readable text after processing."
        )

    if not clean_job_description:
        raise ValueError(
            "The job description is empty."
        )

    model = load_sbert_model()

    embeddings = model.encode(
        [
            clean_job_description,
            clean_cv_text
        ],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    semantic_similarity = float(
        np.dot(
            embeddings[0],
            embeddings[1]
        )
    )

    semantic_score = float(
        np.clip(
            semantic_similarity,
            0,
            1
        ) * 100
    )

    skill_result = calculate_skill_match(
        clean_cv_text,
        required_skills
    )

    skill_score = skill_result["skill_score"]

    final_score = (
        SEMANTIC_WEIGHT * semantic_score
        +
        SKILL_WEIGHT * skill_score
    )

    final_score = float(
        np.clip(
            final_score,
            0,
            100
        )
    )

    predicted_suitability = (
        classify_suitability(final_score)
    )

    recommendation = create_recommendation(
        predicted_suitability
    )

    explanation = (
        f"The CV received a semantic similarity score of "
        f"{semantic_score:.2f}% and a structured skill-match "
        f"score of {skill_score:.2f}%. The final score uses "
        f"70% semantic similarity and 30% structured skill "
        f"matching. The predicted suitability is "
        f"{predicted_suitability}."
    )

    return {
        "semantic_score": semantic_score,
        "skill_score": skill_score,
        "final_score": final_score,
        "predicted_suitability":
            predicted_suitability,
        "matched_skills":
            skill_result["matched_skills"],
        "missing_skills":
            skill_result["missing_skills"],
        "recommendation": recommendation,
        "explanation": explanation,
        "processed_text": clean_cv_text
    }


# =========================================================
# WEBSITE HEADER
# =========================================================

st.title(
    "Explainable AI Candidate–Job Matching Prototype"
)

st.write(
    "This prototype combines Sentence-BERT semantic "
    "similarity with structured skill matching to support "
    "initial CV screening."
)

st.warning(
    "This system provides decision support only. "
    "Final recruitment decisions must be made by humans."
)


# =========================================================
# WEBSITE TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "Existing 20-CV Ranking",
    "Check a New CV",
    "System Information"
])


# =========================================================
# TAB 1: EXISTING RANKING
# =========================================================

with tab1:

    st.header("Existing Synthetic Candidate Ranking")

    existing_results_df = load_existing_results()

    if existing_results_df.empty:

        st.error(
            "The final ranking file was not found. "
            "Please make sure Step 6 was completed."
        )

    else:

        total_candidates = len(
            existing_results_df
        )

        high_candidates = (
            existing_results_df[
                "Predicted Suitability"
            ]
            .eq("High")
            .sum()
        )

        medium_candidates = (
            existing_results_df[
                "Predicted Suitability"
            ]
            .eq("Medium")
            .sum()
        )

        low_candidates = (
            existing_results_df[
                "Predicted Suitability"
            ]
            .eq("Low")
            .sum()
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Candidates",
            total_candidates
        )

        col2.metric(
            "High Suitability",
            high_candidates
        )

        col3.metric(
            "Medium Suitability",
            medium_candidates
        )

        col4.metric(
            "Low Suitability",
            low_candidates
        )

        display_columns = [
            column
            for column in [
                "Final Rank",
                "Candidate ID",
                "Ground Truth Label",
                "Semantic Score (%)",
                "Skill Score (%)",
                "Final Score (%)",
                "Predicted Suitability",
                "Recommendation"
            ]
            if column in existing_results_df.columns
        ]

        st.dataframe(
            existing_results_df[
                display_columns
            ],
            use_container_width=True,
            hide_index=True
        )

        candidate_options = (
            existing_results_df[
                "Candidate ID"
            ]
            .astype(str)
            .tolist()
        )

        selected_candidate = st.selectbox(
            "Select a candidate to view the explanation",
            candidate_options
        )

        selected_row = existing_results_df[
            existing_results_df["Candidate ID"]
            == selected_candidate
        ].iloc[0]

        st.subheader(
            f"Explanation for {selected_candidate}"
        )

        if "Explanation" in selected_row.index:
            st.info(
                selected_row["Explanation"]
            )

        if "Matched Skills" in selected_row.index:
            st.write(
                "**Matched skills:**",
                selected_row["Matched Skills"]
            )

        if "Missing Skills" in selected_row.index:
            st.write(
                "**Missing skills:**",
                selected_row["Missing Skills"]
            )


# =========================================================
# TAB 2: CHECK A NEW CV
# =========================================================

with tab2:

    st.header("Check a New Candidate CV")

    default_job_description = (
        load_default_job_description()
    )

    job_description_input = st.text_area(
        "Job description",
        value=default_job_description,
        height=280
    )

    skills_input = st.text_area(
        "Required skills — one skill per line",
        value="\n".join(REQUIRED_SKILLS),
        height=220
    )

    new_required_skills = [
        skill.strip()
        for skill in skills_input.splitlines()
        if skill.strip()
    ]

    uploaded_cv = st.file_uploader(
        "Upload a CV",
        type=[
            "pdf",
            "docx"
        ],
        accept_multiple_files=False
    )

    analyse_button = st.button(
        "Analyse New CV",
        type="primary"
    )

    if analyse_button:

        if uploaded_cv is None:

            st.error(
                "Please upload a PDF or DOCX CV."
            )

        elif not job_description_input.strip():

            st.error(
                "Please provide a job description."
            )

        elif not new_required_skills:

            st.error(
                "Please provide at least one required skill."
            )

        else:

            try:

                with st.spinner(
                    "Extracting and analysing the CV..."
                ):

                    extracted_cv_text = (
                        extract_uploaded_cv_text(
                            uploaded_cv
                        )
                    )

                    new_result = score_new_cv(
                        cv_text=extracted_cv_text,
                        job_description=
                            job_description_input,
                        required_skills=
                            new_required_skills
                    )

                st.success(
                    "CV analysis completed successfully."
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Semantic Similarity",
                    f"{new_result['semantic_score']:.2f}%"
                )

                col2.metric(
                    "Skill Match",
                    f"{new_result['skill_score']:.2f}%"
                )

                col3.metric(
                    "Final Score",
                    f"{new_result['final_score']:.2f}%"
                )

                st.subheader(
                    "Predicted Suitability"
                )

                if (
                    new_result[
                        "predicted_suitability"
                    ]
                    == "High"
                ):
                    st.success("High suitability")

                elif (
                    new_result[
                        "predicted_suitability"
                    ]
                    == "Medium"
                ):
                    st.warning("Medium suitability")

                else:
                    st.error("Low suitability")

                st.info(
                    new_result["recommendation"]
                )

                left_column, right_column = (
                    st.columns(2)
                )

                with left_column:

                    st.subheader("Matched Skills")

                    if new_result["matched_skills"]:

                        for skill in (
                            new_result[
                                "matched_skills"
                            ]
                        ):
                            st.write(f"✅ {skill}")

                    else:
                        st.write(
                            "No required skills were identified."
                        )

                with right_column:

                    st.subheader("Missing Skills")

                    if new_result["missing_skills"]:

                        for skill in (
                            new_result[
                                "missing_skills"
                            ]
                        ):
                            st.write(f"❌ {skill}")

                    else:
                        st.write(
                            "No required skills are missing."
                        )

                st.subheader(
                    "Explainable Recommendation"
                )

                st.write(
                    new_result["explanation"]
                )

                existing_results_df = (
                    load_existing_results()
                )

                if not existing_results_df.empty:

                    new_candidate_row = pd.DataFrame([
                        {
                            "Candidate ID":
                                uploaded_cv.name,

                            "Ground Truth Label":
                                "New CV",

                            "Semantic Score (%)":
                                round(
                                    new_result[
                                        "semantic_score"
                                    ],
                                    2
                                ),

                            "Skill Score (%)":
                                round(
                                    new_result[
                                        "skill_score"
                                    ],
                                    2
                                ),

                            "Final Score (%)":
                                round(
                                    new_result[
                                        "final_score"
                                    ],
                                    2
                                ),

                            "Predicted Suitability":
                                new_result[
                                    "predicted_suitability"
                                ],

                            "Recommendation":
                                new_result[
                                    "recommendation"
                                ]
                        }
                    ])

                    ranking_columns = [
                        "Candidate ID",
                        "Ground Truth Label",
                        "Semantic Score (%)",
                        "Skill Score (%)",
                        "Final Score (%)",
                        "Predicted Suitability",
                        "Recommendation"
                    ]

                    existing_for_ranking = (
                        existing_results_df[
                            ranking_columns
                        ].copy()
                    )

                    updated_ranking = pd.concat(
                        [
                            existing_for_ranking,
                            new_candidate_row
                        ],
                        ignore_index=True
                    )

                    updated_ranking = (
                        updated_ranking
                        .sort_values(
                            by="Final Score (%)",
                            ascending=False
                        )
                        .reset_index(drop=True)
                    )

                    updated_ranking.insert(
                        0,
                        "Rank",
                        range(
                            1,
                            len(updated_ranking) + 1
                        )
                    )

                    new_cv_rank = int(
                        updated_ranking.loc[
                            updated_ranking[
                                "Candidate ID"
                            ]
                            == uploaded_cv.name,
                            "Rank"
                        ].iloc[0]
                    )

                    st.subheader(
                        "Position Among Existing Candidates"
                    )

                    st.metric(
                        "New CV Rank",
                        f"{new_cv_rank} of "
                        f"{len(updated_ranking)}"
                    )

                    st.dataframe(
                        updated_ranking,
                        use_container_width=True,
                        hide_index=True
                    )

                with st.expander(
                    "View extracted CV text"
                ):
                    st.text(
                        extracted_cv_text[:15000]
                    )

            except Exception as error:

                st.error(
                    f"CV analysis failed: {error}"
                )


# =========================================================
# TAB 3: SYSTEM INFORMATION
# =========================================================

with tab3:

    st.header("System Information")

    st.write(
        "**Semantic model:** "
        f"{MODEL_NAME}"
    )

    st.write(
        "**Semantic similarity weight:** "
        f"{SEMANTIC_WEIGHT * 100:.0f}%"
    )

    st.write(
        "**Structured skill weight:** "
        f"{SKILL_WEIGHT * 100:.0f}%"
    )

    st.write(
        "**Supported CV formats:** PDF and DOCX"
    )

    st.write(
        "**Protected information:** Clearly labelled age, "
        "gender, nationality, religion and similar fields are "
        "removed before scoring."
    )

    st.write(
        "**Purpose:** Initial candidate-screening decision "
        "support with mandatory human oversight."
    )
