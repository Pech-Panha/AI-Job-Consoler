import sys
import sqlite3
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QScrollArea, QFrame, QMessageBox,
    QTextEdit, QStackedWidget, QRadioButton, QButtonGroup, QSpacerItem, QSizePolicy,
    QListWidget, QGridLayout
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor, QPalette, QBrush, QLinearGradient, QPixmap

import json
from collections import defaultdict

# --- Import for plotting ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Scikit-learn imports ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --- Database Setup ---
DATABASE_NAME = 'career_data.db'

def init_db():
    """Initializes the SQLite database and creates the survey_responses table."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS survey_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            raw_survey_responses TEXT,
            preferred_industry TEXT,
            recommended_career TEXT,
            recommendation_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

# --- JOB DETAILS (Incorporating all data from Ai_Jobs.docx) ---
JOB_DETAILS = {
    "Doctor": {
        "description": "Diagnose and treat illnesses, perform check-ups, prescribe medicine, and guide patients to recovery.",
        "salary_range": "$11,000 - $40,000 per year (Annual)",
        "skills": [
            "Clinical Diagnosis",
            "Treatment Planning",
            "Patient Care",
            "Medical Procedures",
            "Communication",
            "Problem-solving"
        ],
        "schools": [
            "University of Health Sciences",
            "International University",
            "Norton University",
            "University of Puthisastra",
            "Cambodian University for Specialties"
        ],
        "companies": [
            "Royal Phnom Penh Hospital",
            "Calmette Hospital",
            "Sunrise Japan Hospital Phnom Penh",
            "SenSok International University Hospital",
            "Raffles Medical Phnom Penh"
        ],
        "image_path": "img/doctor.jpg"
    },
    "Project Manager": {
        "description": "leads and oversees projects from start to finish—planning, coordinating resources, ensuring timelines and budgets are met, managing teams, and communicating with stakeholders.",
        "salary_range": "$5,000 - $19,000 per year (Annual)",
        "skills": [
            "Planning",
            "Budget Management",
            "Team Leadership",
            "Risk Management",
            "Communication"
        ],
        "schools": [
            "CamEd Business School",
            "University of Puthisatra",
            "National Institute of Business",
            "National University of Management",
            "Norton University"
        ],
        "companies": [
            "Vattanac Bank Cambodia",
            "ACLEDA Company",
            "Oddatelier Company"
        ],
        "image_path": "img/Project manager.png"
    },
    "Researcher": {
        "description": "Researchers conduct systematic investigations to establish facts, develop new theories, or revise existing ones. They often work in academic institutions, government agencies, or private companies, designing experiments, collecting and analyzing data, and reporting their findings. Strong analytical skills, attention to detail, and a commitment to scientific integrity are crucial.",
        "salary_range": "$5,000 - $18,000 per year (Annual)",
        "skills": [
            "Research Design",
            "Data Collection",
            "Statistical Analysis",
            "Report Writing",
            "Critical Thinking"
        ],
        "schools": [
            "Royal University of Phnom Penh (RUPP)",
            "University of Health Sciences (UHS)",
            "Institute of Technology of Cambodia (ITC)",
            "American University of Phnom Penh (AUPP)"
        ],
        "companies": [
            "Universities",
            "Government Labs",
            "Pharmaceutical Companies",
            "R&D Departments",
            "Innovative Research Firms"
        ],
        "image_path": "img/Researcher.png"
    },
    "UX/UI Designer": {
        "description": "focuses on creating intuitive, efficient, and enjoyable user experiences for websites, apps, and software. They research user needs, design interfaces, and test prototypes to ensure products are user-friendly.",
        "salary_range": "$6,000 - $20,000 per year (Annual)",
        "skills": [
            "User Research",
            "Wireframing",
            "Prototyping",
            "Usability Testing",
            "Figma/Sketch/Adobe XD",
            "Communication"
        ],
        "schools": [
            "Limkokwing University of Creative Technology",
            "Royal University of Phnom Penh (RUPP)",
            "Pannasastra University of Cambodia (PUC)",
            "Cambodia Academy of Digital Technology (CADT)"
        ],
        "companies": [
            "Tech Startups",
            "Digital Agencies",
            "E-commerce Companies",
            "Software Development Firms",
            "Banks"
        ],
        "image_path": "img/ux ui.png"
    },
    "Data Scientist": {
        "description": "Data Scientists analyze complex datasets to extract insights and knowledge. They use statistical analysis, machine learning, and programming to build predictive models and inform business decisions. A strong background in mathematics and statistics is beneficial.",
        "salary_range": "$7,000 - $24,000 per year (Annual)",
        "skills": [
            "Programming (Python, R, SQL)",
            "Data Analysis Tools",
            "Machine Learning",
            "Data Visualization",
            "Problem-solving",
            "Critical thinking",
            "Communication"
        ],
        "schools": [
            "American University of Phnom Penh (AUPP)",
            "Institute of Technology of Cambodia (ITC)",
            "Royal University of Phnom Penh (RUPP)",
            "Step IT Academy",
            "Cambodia Academy of Digital Technology (CADT)"
        ],
        "companies": [
            "Banks & Microfinance Company: ABA Bank, Acleda Bank, Wing Bank",
            "Telecom: Smart Axiata, Metfone, Cellcard",
            "Tech Companies / Startups: Codingate, Pathmazing, Slash"
        ],
        "image_path": "img/Data Scientist.png"
    },
    "Software Engineer": {
        "description": "design, develop, and maintain software applications. They apply engineering principles to build robust, scalable, and efficient systems.",
        "salary_range": "$6,000 - $22,000 per year (Annual)",
        "skills": [
            "Programming (Java, Python, C++, JavaScript)",
            "Data Structures & Algorithms",
            "Software Development Life Cycle (SDLC)",
            "Database Management",
            "Problem-solving",
            "Teamwork"
        ],
        "schools": [
            "Royal University of Phnom Penh (RUPP)",
            "Institute of Technology of Cambodia (ITC)",
            "National University of Management (NUM)",
            "American University of Phnom Penh (AUPP)",
            "SETEC Institute"
        ],
        "companies": [
            "Tech Companies (e.g., Agoda, Pruksa)",
            "Banks & FinTech",
            "Telecoms",
            "Software Outsourcing Firms",
            "E-commerce Platforms"
        ],
        "image_path": "img/Software enginee.png"
    },
    "Fire Fighter": {
        "description": "Firefighters respond to emergencies, extinguish fires, rescue people from dangerous situations, and provide first aid. They also educate the public on fire safety.",
        "salary_range": "$3,000 - $8,000 per year (Annual)",
        "skills": [
            "Emergency Response",
            "First Aid/CPR",
            "Physical Fitness",
            "Teamwork",
            "Stress Management"
        ],
        "schools": [
            "National Police Academy of Cambodia (specific firefighter training programs)",
            "Various provincial training centers"
        ],
        "companies": [
            "Fire and Rescue Department (under Ministry of Interior)",
            "Airport Fire Services",
            "Industrial Fire Brigades (large factories, complexes)"
        ],
        "image_path": "img/firefigher.jpg"
    },
    "Lawyer": {
        "description": "Lawyers provide legal advice, represent clients in court, and prepare legal documents. They specialize in various fields like criminal law, civil law, or corporate law.",
        "salary_range": "$8,000 - $30,000 per year (Annual)",
        "skills": [
            "Legal Research",
            "Advocacy",
            "Negotiation",
            "Contract Drafting",
            "Communication",
            "Analytical Thinking"
        ],
        "schools": [
            "Royal University of Law and Economics (RULE)",
            "Pannasastra University of Cambodia (PUC)",
            "National University of Management (NUM)",
            "University of Cambodia (UC)"
        ],
        "companies": [
            "Law Firms",
            "Corporate Legal Departments",
            "Government Ministries",
            "NGOs",
            "International Organizations"
        ],
        "image_path": "img/lawyer.jpg"
    },
    "High School Teacher": {
        "description": "High school teachers educate students in various subjects, prepare lesson plans, assess student progress, and foster a positive learning environment.",
        "salary_range": "$3,000 - $10,000 per year (Annual)",
        "skills": [
            "Lesson Planning",
            "Classroom Management",
            "Subject Matter Expertise",
            "Communication",
            "Student Assessment",
            "Adaptability"
        ],
        "schools": [
            "National Institute of Education (NIE)",
            "Royal University of Phnom Penh (RUPP) - Education Dept.",
            "Phnom Penh International University (PPIU) - Education Dept."
        ],
        "companies": [
            "Public High Schools (Ministry of Education, Youth and Sport)",
            "Private International Schools",
            "Community Learning Centers"
        ],
        "image_path": "img/teacher.jpg"
    },
    "Accountant": {
        "description": "Accountants prepare and examine financial records, ensure financial statements are accurate, and help individuals and businesses manage their finances and comply with tax laws.",
        "salary_range": "$4,000 - $15,000 per year (Annual)",
        "skills": [
            "Financial Reporting",
            "Tax Preparation",
            "Auditing",
            "Bookkeeping",
            "Data Analysis",
            "Attention to Detail"
        ],
        "schools": [
            "CamEd Business School",
            "National University of Management (NUM)",
            "Royal University of Law and Economics (RULE)",
            "University of Cambodia (UC)"
        ],
        "companies": [
            "Accounting Firms",
            "Banks & Financial Institutions",
            "Manufacturing Companies",
            "NGOs",
            "Government Agencies"
        ],
        "image_path": "img/accountant.jpg"
    },
    "Civil Site Engineer": {
        "description": "Civil Site Engineers plan, design, and manage construction projects such as buildings, roads, bridges, and infrastructure, ensuring they are built safely and efficiently.",
        "salary_range": "$5,000 - $18,000 per year (Annual)",
        "skills": [
            "Project Management",
            "Structural Analysis",
            "AutoCAD/Design Software",
            "Site Supervision",
            "Problem-solving",
            "Safety Regulations"
        ],
        "schools": [
            "Institute of Technology of Cambodia (ITC)",
            "National University of Management (NUM)",
            "Norton University",
            "Royal University of Phnom Penh (RUPP) - Engineering Dept."
        ],
        "companies": [
            "Construction Companies",
            "Real Estate Developers",
            "Consulting Engineering Firms",
            "Government Public Works Departments",
            "Infrastructure Development Companies"
        ],
        "image_path": "img/enginee.jpg"
    },
    "Architecture": {
        "description": "Architects design buildings and other physical structures. They blend aesthetics with functionality, considering safety, sustainability, and client needs.",
        "salary_range": "$5,000 - $17,000 per year (Annual)",
        "skills": [
            "Architectural Design",
            "AutoCAD/Revit",
            "Sketching & Rendering",
            "Building Codes",
            "Project Management",
            "Creativity"
        ],
        "schools": [
            "Royal University of Phnom Penh (RUPP) - Dept. of Architecture",
            "Limkokwing University of Creative Technology",
            "Pannasastra University of Cambodia (PUC) - Architecture"
        ],
        "companies": [
            "Architectural Firms",
            "Construction Companies",
            "Real Estate Development Firms",
            "Interior Design Companies",
            "Government Urban Planning Departments"
        ],
        "image_path": "img/architect.jpg"
    },
    "Artist": {
        "description": "Artists create visual, performing, or literary works. This broad field includes painters, sculptors, musicians, writers, and digital artists, who use their creativity to express ideas and evoke emotions.",
        "salary_range": "$2,000 - $10,000 per year (Annual) - Highly variable",
        "skills": [
            "Creativity",
            "Specific Art Medium (e.g., painting, digital art, music)",
            "Self-promotion",
            "Attention to Detail",
            "Adaptability"
        ],
        "schools": [
            "Royal University of Fine Arts (RUFA)",
            "Limkokwing University of Creative Technology",
            "Phare Ponleu Selpak (Artistic training NGO)"
        ],
        "companies": [
            "Art Galleries",
            "Design Studios",
            "Entertainment Industry",
            "Advertising Agencies",
            "Freelance/Self-employed"
        ],
        "image_path": "img/job1.png"
    },
    "Digital Marketer": {
        "description": "Digital marketers promote products or services online using various digital channels like social media, search engines, email, and websites. They focus on increasing brand awareness, driving traffic, and generating leads.",
        "salary_range": "$2,000 - $9,000 per year (Annual)",
        "skills": [
            "Social Media Marketing",
            "Content Creation",
            "SEO (Search Engine Optimization)",
            "Email Marketing",
            "Google Analytics",
            "Campaign Management"
        ],
        "schools": [
            "National University of Management (NUM) - Marketing",
            "Pannasastra University of Cambodia (PUC) - Marketing",
            "Royal University of Phnom Penh (RUPP) - Media & Communication"
        ],
        "companies": [
            "Digital Marketing Agencies",
            "E-commerce Businesses",
            "Tech Startups",
            "Large Corporations (in-house marketing teams)",
            "NGOs"
        ],
        "image_path": "img/Digital marketer.png"
    },
    "Human Resource (HR)": {
        "description": "manages recruitment, employee relations, training, and company policies to support staff and help the organization run smoothly.",
        "salary_range": "$2,000 - $9,000 per year (Annual)",
        "skills": [
            "Recruitment and interviewing",
            "Knowledge of Cambodian labor law",
            "Payroll and benefits administration",
            "Communication and interpersonal skills",
            "Problem-solving and conflict management"
        ],
        "schools": [
            "Human Resource University",
            "Pannasastra University of Cambodia",
            "Royal University of Phnom Penh",
            "The Knowledge Academy",
            "Cambodian Mekong University"
        ],
        "companies": [
            "private companies",
            "non-profit organizations",
            "government agencies",
            "Consulting Firms",
            "International Organizations"
        ],
        "image_path": "img/HR.jpg"
    }
}


# --- Dummy Data Generation for ML Model Training ---
def generate_dummy_data(num_samples=200):
    """
    Generates a synthetic dataset for training the ML model.
    Maps aggregated survey responses to career outcomes.
    """
    # Define the aggregated features (0-10 scale)
    feature_names = [
        'math_interest', 'science_interest', 'coding_interest', 'design_interest',
        'problem_solving_skill', 'communication_skill', 'creativity_skill', 'leadership_skill'
    ]
    
    # Define possible career outcomes - using all keys from JOB_DETAILS
    career_outcomes = list(JOB_DETAILS.keys())

    data = []
    labels = []
    
    rng = np.random.default_rng(seed=42)

    # Simplified generation logic for demonstration
    career_to_features = {
        'Doctor': {'science_interest': 5, 'problem_solving_skill': 5, 'communication_skill': 4},
        'Software Engineer': {'coding_interest': 5, 'math_interest': 4, 'problem_solving_skill': 5},
        'Architect': {'design_interest': 5, 'creativity_skill': 5, 'problem_solving_skill': 4},
        'Teacher': {'communication_skill': 5, 'leadership_skill': 4},
        'Graphic Designer': {'design_interest': 5, 'creativity_skill': 5},
        'Marketing Manager': {'communication_skill': 5, 'leadership_skill': 5},
        'Accountant': {'math_interest': 5, 'problem_solving_skill': 4},
        'Chef': {'creativity_skill': 4, 'leadership_skill': 4},
        'Lawyer': {'communication_skill': 5, 'problem_solving_skill': 5},
        'Human Resource (HR)': {'communication_skill': 5, 'leadership_skill': 4, 'problem_solving_skill': 3},
        'Project Manager': {'leadership_skill': 5, 'communication_skill': 5, 'problem_solving_skill': 4},
        'Researcher': {'science_interest': 5, 'problem_solving_skill': 5, 'creativity_skill': 4},
        'UX/UI Designer': {'design_interest': 5, 'creativity_skill': 5, 'communication_skill': 4},
        'Data Scientist': {'math_interest': 5, 'coding_interest': 5, 'problem_solving_skill': 5},
        'Fire Fighter': {'leadership_skill': 5, 'problem_solving_skill': 5, 'communication_skill': 4},
        'High School Teacher': {'communication_skill': 5, 'leadership_skill': 4},
        'Civil Site Engineer': {'math_interest': 5, 'problem_solving_skill': 5, 'design_interest': 4},
        'Architecture': {'design_interest': 5, 'creativity_skill': 5, 'problem_solving_skill': 4},
        'Artist': {'creativity_skill': 5, 'design_interest': 5, 'communication_skill': 3},
        'Digital Marketer': {'design_interest': 4, 'communication_skill': 5, 'creativity_skill': 4},
    }

    for _ in range(num_samples):
        career = rng.choice(career_outcomes)
        row = defaultdict(int)
        
        # Add a bias for the features associated with the chosen career
        for feature, bias in career_to_features.get(career, {}).items():
            row[feature] = int(min(10, max(0, bias + rng.integers(-3, 4))))

        # Fill in the rest with random values
        for feature in feature_names:
            if feature not in row:
                row[feature] = rng.integers(0, 11)

        data.append([row[f] for f in feature_names])
        labels.append(career)

    X = pd.DataFrame(data, columns=feature_names)
    y = pd.Series(labels)
    
    return X, y, feature_names, career_outcomes

# --- Machine Learning Model Training ---
def train_career_model():
    """
    Trains a Decision Tree Classifier model for career recommendation.
    """
    X, y, feature_names, career_outcomes = generate_dummy_data()

    # Create a pipeline with scaling and a classifier
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    model_pipeline.fit(X, y)
    print("Machine Learning model trained successfully.")
    return model_pipeline, feature_names, career_outcomes

# --- ML-based Recommendation Logic ---
def get_ml_career_recommendation(ml_model, feature_names, career_outcomes, raw_survey_responses, preferred_industry):
    """
    Uses the trained ML model to get career recommendations.

    Args:
        ml_model (Pipeline): The trained Scikit-learn pipeline.
        feature_names (list): List of feature names used during training.
        career_outcomes (list): List of possible career outcomes.
        raw_survey_responses (dict): A dictionary containing responses to the 20 questions (1-7 scale).
        preferred_industry (str): The user's selected preferred industry.

    Returns:
        tuple: (recommended_career, recommendation_score, top_careers_for_display)
    """
    def map_scale(value):
        return (value - 1) * (10 / 6) # Map 1-7 scale to 0-10

    # Aggregate raw survey responses into interest/skill categories (0-10 scale)
    category_scores = defaultdict(float)
    category_counts = defaultdict(int)

    # Updated mapping with 10 new questions
    q_to_category = {
        'q1': 'math_interest', 'q2': 'math_interest', 'q21': 'math_interest',
        'q3': 'science_interest', 'q4': 'science_interest', 'q22': 'science_interest',
        'q5': 'coding_interest', 'q6': 'coding_interest', 'q7': 'coding_interest', 'q23': 'coding_interest',
        'q8': 'design_interest', 'q9': 'design_interest', 'q24': 'design_interest',
        'q10': 'problem_solving_skill', 'q11': 'problem_solving_skill', 'q12': 'problem_solving_skill', 'q25': 'problem_solving_skill', 'q26': 'problem_solving_skill',
        'q13': 'communication_skill', 'q14': 'communication_skill', 'q15': 'communication_skill', 'q27': 'communication_skill', 'q28': 'communication_skill',
        'q16': 'creativity_skill', 'q17': 'creativity_skill', 'q29': 'creativity_skill',
        'q18': 'leadership_skill', 'q19': 'leadership_skill', 'q20': 'leadership_skill', 'q30': 'leadership_skill',
    }

    for q_key, response_value in raw_survey_responses.items():
        category = q_to_category.get(q_key)
        if category:
            category_scores[category] += map_scale(response_value)
            category_counts[category] += 1

    aggregated_data = {}
    for feature in feature_names:
        if category_counts[feature] > 0:
            aggregated_data[feature] = category_scores[feature] / category_counts[feature]
        else:
            aggregated_data[feature] = map_scale(4) # Default to neutral (4 on 1-7 scale, mapped to ~5 on 0-10) if no questions contributed

    # Prepare features for the model in the correct order
    input_features = pd.DataFrame([list(aggregated_data.values())], columns=feature_names)

    # Get probability predictions for each career
    probabilities = ml_model.predict_proba(input_features)[0]
    
    # Create a dictionary of career probabilities
    career_probs = {career: prob for career, prob in zip(ml_model.classes_, probabilities)}

    # Apply preferred industry boost (post-prediction)
    industry_boost_factor = 1.2
    career_industry_mapping = {
        "Software Engineer": ["IT", "Technology"],
        "Data Scientist": ["IT", "Research", "Finance"],
        "UX/UI Designer": ["Design", "IT"],
        "Project Manager": ["Management", "General"],
        "Researcher": ["Research", "Science"],
        "Doctor": ["Healthcare"],
        "Fire Fighter": ["Public Service", "General"],
        "Lawyer": ["Legal", "General"],
        "High School Teacher": ["Education", "General"],
        "Accountant": ["Finance", "General"],
        "Civil Site Engineer": ["Engineering", "Construction"],
        "Architecture": ["Design", "Construction"],
        "Artist": ["Arts", "Design"],
        "Digital Marketer": ["Marketing", "IT"],
        "Human Resource (HR)": ["Management", "General"]
    }

    for career, prob in career_probs.items():
        if preferred_industry in career_industry_mapping.get(career, []):
            career_probs[career] = min(1.0, prob * industry_boost_factor) # Cap at 1.0

    # Convert probabilities to scores (e.g., out of 100)
    career_scores = {career: prob * 100 for career, prob in career_probs.items()}

    # Get the top recommended career and its score
    if career_scores:
        recommended_career = max(career_scores, key=career_scores.get)
        recommendation_score = career_scores[recommended_career]
    else:
        recommended_career = "Uncertain"
        recommendation_score = 0.0

    # Get top 3 careers for display
    sorted_careers = sorted(career_scores.items(), key=lambda item: item[1], reverse=True)
    top_careers_for_display = [(career, score) for career, score in sorted_careers[:3]]

    return recommended_career, recommendation_score, top_careers_for_display


# --- Main Application Window ---
class CareerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ប្រព័ន្ធវិភាគសមត្ថភាព និងផ្តល់យោបល់ការងារ") # Competency Analysis and Career Counseling System
        # Adjusted window size to fix the QWindowsWindow::setGeometry warning
        self.setGeometry(100, 100, 1400, 1000)

        self.ml_model, self.feature_names, self.career_outcomes = train_career_model()

        self.init_ui()
        init_db()

    def init_ui(self):
        """Initializes the user interface."""
        palette = self.palette()
        # Changed page background color to white
        palette.setColor(QPalette.ColorRole.Window, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#333333"))
        # Changed text box background color to F5F5F5
        palette.setColor(QPalette.ColorRole.Base, QColor("#F5F5F5"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#333333"))
        # Changed primary color to 06D001
        palette.setColor(QPalette.ColorRole.Button, QColor("#06D001"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#ffffff"))
        self.setPalette(palette)

        font = QFont("Khmer OS Siemreap", 10)
        self.setFont(font)

        self.stacked_widget = QStackedWidget(self)
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.stacked_widget)


        # Now the job details page is the main hub, containing the intro and job list
        self.job_details_page = self.create_job_details_page() 
        self.survey_page = self.create_survey_page()
        self.results_page = self.create_results_page()
        self.history_page = self.create_history_page()

        self.stacked_widget.addWidget(self.job_details_page) # Index 0
        self.stacked_widget.addWidget(self.survey_page) # Index 1
        self.stacked_widget.addWidget(self.results_page) # Index 2
        self.stacked_widget.addWidget(self.history_page) # Index 3

        # Set the main page as the default and show the "Intro" content
        self.stacked_widget.setCurrentIndex(0)
        self.career_list_widget_job_details_page.setCurrentRow(0)
        self.display_intro_content()


    def create_survey_page(self):
        """Creates the survey input page widget with improved styling."""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        # Changed page background color to white
        widget.setStyleSheet("background-color: #FFFFFF;")

        header_label = QLabel("បំពេញសំណួរស្ទង់មតិ (៣០ សំណួរ)") # Updated question count
        header_label.setFont(QFont("Khmer OS Muol Light", 18))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Changed importance text/title color to 059212
        header_label.setStyleSheet("color: #059212; margin-bottom: 25px;")
        main_layout.addWidget(header_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        # Changed page background color to white
        scroll_area.setStyleSheet("QScrollArea { border: 1px solid #d0d0d0; border-radius: 10px; background-color: #FFFFFF; }")
        scroll_content = QWidget()
        self.survey_layout = QVBoxLayout(scroll_content)
        self.survey_layout.setContentsMargins(25, 25, 25, 25)
        self.survey_layout.setSpacing(20)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        student_info_prompt_label = QLabel("សូមបំពេញព័ត៌មានរបស់សិស្សជាមុនសិនៈ")
        student_info_prompt_label.setFont(QFont("Khmer OS Siemreap", 12))
        self.survey_layout.addWidget(student_info_prompt_label)

        student_name_label = QLabel("ឈ្មោះនិស្សិត*:")
        student_name_label.setFont(QFont("Khmer OS Siemreap", 12))
        self.survey_layout.addWidget(student_name_label)
        self.student_name_input = QLineEdit()

        self.student_name_input.setPlaceholderText("បញ្ចូលឈ្មោះរបស់អ្នក")
        self.student_name_input.setFont(QFont("Khmer OS Siemreap", 11))
        # Changed text box content background color to F5F5F5
        self.student_name_input.setStyleSheet(
            "QLineEdit { "
            "padding: 10px; border: 1px solid #a0a0a0; border-radius: 10px; "
            "background-color: #F5F5F5; "
            "}"
            # Changed importance text/title color to 059212
            "QLineEdit:focus { border: 2px solid #059212; background-color: #ffffff; }"
        )
        self.survey_layout.addWidget(self.student_name_input)
        self.survey_layout.addSpacing(25)

        industry_label = QLabel("ឧស្សាហកម្មដែលពេញចិត្ត:")
        industry_label.setFont(QFont("Khmer OS Siemreap", 12))
        self.survey_layout.addWidget(industry_label)
        self.survey_industry_combo = QComboBox()
        self.survey_industry_combo.addItems([
            "IT", "Design", "Management", "Research", "Finance", "Education",
            "Healthcare", "Public Service", "Legal", "Construction", "Engineering",
            "Arts", "Marketing", "General"
        ])
        self.survey_industry_combo.setFont(QFont("Khmer OS Siemreap", 11))
        self.survey_industry_combo.setStyleSheet(
            "QComboBox { "
            "padding: 10px; border: 1px solid #a0a0a0; border-radius: 10px; "
            # Changed text box content background color to F5F5F5
            "background-color: #F5F5F5; "
            # Changed primary color to 06D001
            "selection-background-color: #06D001; "
            "}"
            "QComboBox::drop-down { border: none; }"
        )
        self.survey_layout.addWidget(self.survey_industry_combo)
        self.survey_layout.addSpacing(30)


        # Updated questions list with 10 new questions
        self.questions = [
            "I enjoy solving complex mathematical problems.",
            "I find working with numbers and data engaging.",
            "I am curious about how the natural world and scientific principles work.",
            "I enjoy learning about new scientific discoveries and theories.",
            "I enjoy writing code and developing software applications.",
            "I like to build and troubleshoot computer systems.",
            "I am interested in artificial intelligence and machine learning.",
            "I enjoy creating visual designs and artistic layouts.",
            "I pay attention to user experience and interface aesthetics.",
            "I enjoy breaking down complex problems into smaller, manageable parts.",
            "I like to find innovative solutions to challenges.",
            "I am good at analytical thinking and logical reasoning.",
            "I am comfortable presenting ideas and information to groups.",
            "I enjoy collaborating with others to achieve a common goal.",
            "I can clearly explain complex topics to different audiences.",
            "I enjoy generating new and original ideas.",
            "I am comfortable thinking outside the box and experimenting.",
            "I like to take charge and guide a team towards a goal.",
            "I am good at organizing tasks and managing resources.",
            "I enjoy motivating others and resolving conflicts.",
            "I am good at financial planning and managing budgets.", # New Q21 - Math
            "I am fascinated by topics like biology, chemistry, or physics.", # New Q22 - Science
            "I enjoy working with logical systems and precise instructions.", # New Q23 - Coding
            "I like to use software tools to create visual content.", # New Q24 - Design
            "I can stay calm and focused when faced with a difficult challenge.", # New Q25 - Problem-solving
            "I enjoy researching different viewpoints to solve a puzzle.", # New Q26 - Problem-solving
            "I am skilled at listening to and understanding other people's perspectives.", # New Q27 - Communication
            "I am comfortable giving feedback and coaching others.", # New Q28 - Communication
            "I am interested in storytelling and creating compelling narratives.", # New Q29 - Creativity
            "I find it rewarding to mentor others and help them grow.", # New Q30 - Leadership
        ]

        self.question_button_groups = {}

        for i, q_text in enumerate(self.questions):
            q_num = i + 1
            question_label = QLabel(f"សំណួរ {q_num}: {q_text}")
            question_label.setFont(QFont("Khmer OS Siemreap", 11, QFont.Weight.Bold))
            question_label.setStyleSheet("color: #333333;")
            self.survey_layout.addWidget(question_label)

            h_layout = QHBoxLayout()
            h_layout.setSpacing(10)

            button_group = QButtonGroup(self)
            self.question_button_groups[f'q{q_num}'] = button_group

            disagree_label = QLabel("Disagree")
            disagree_label.setFont(QFont("Khmer OS Siemreap", 10, QFont.Weight.Bold))
            disagree_label.setStyleSheet("color: #F72C5B;")
            h_layout.addWidget(disagree_label)

            for val in range(1, 8):
                radio_button = QRadioButton()
                radio_button.setProperty("value", val)
                radio_button.setFixedSize(30, 30)
                # Changed primary color to 06D001 and hover to 059212
                radio_button.setStyleSheet(
                    "QRadioButton::indicator { "
                    "width: 25px; height: 25px; border-radius: 12px; "
                    "border: 2px solid #a0a0a0; "
                    "background-color: #f0f0f0; "
                    "}"
                    "QRadioButton::indicator:unchecked:hover { background-color: #e0e0e0; }"
                    "QRadioButton::indicator:checked { "
                    "background-color: #06D001; "
                    "border: 2px solid #06D001; "
                    "}"
                )
                if val == 4:
                    radio_button.setChecked(True)
                h_layout.addWidget(radio_button, alignment=Qt.AlignmentFlag.AlignCenter)
                button_group.addButton(radio_button, val)

            agree_label = QLabel("Agree")
            agree_label.setFont(QFont("Khmer OS Siemreap", 10, QFont.Weight.Bold))
            agree_label.setStyleSheet("color: #06D001;")
            h_layout.addWidget(agree_label)
            
            h_layout.addStretch(1)
            self.survey_layout.addLayout(h_layout)
            self.survey_layout.addSpacing(15)

        self.survey_layout.addSpacing(30)

        submit_button = QPushButton("បំពេញការស្ទង់មតិ")
        submit_button.setFont(QFont("Khmer OS Siemreap", 14, QFont.Weight.Bold))
        submit_button.setFixedSize(250, 55)
        # Changed primary color to 06D001 and hover to 059212
        submit_button.setStyleSheet(
            "QPushButton { "
            "background-color: #06D001; color: white; border-radius: 27px; "
            "border: none; padding: 10px 20px; "
            "}"
            "QPushButton:hover { "
            "background-color: #059212; "
            "}"
        )
        submit_button.clicked.connect(self.submit_survey)
        self.survey_layout.addWidget(submit_button, alignment=Qt.AlignmentFlag.AlignCenter)

        back_button = QPushButton("ត្រឡប់ទៅទំព័រដើម")
        back_button.setFont(QFont("Khmer OS Siemreap", 11))
        back_button.setFixedSize(180, 65) # Changed size
        back_button.setStyleSheet(
            "QPushButton { "
            "background-color: #F72C5B; color: white; border-radius: 25px; "
            "border: none; padding: 10px 20px; margin-top: 15px; "
            "}"
            "QPushButton:hover { "
            "background-color: #FF748B; "
            "}"
        )
        back_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0)) # Back to the main job details page (now the hub)
        self.survey_layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.survey_layout.addStretch(1)

        return widget

    def create_results_page(self):
        """
        Creates the results display page with the new layout:
        Title on top, screenshot on left, details on right, and buttons at the bottom.
        """
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Changed page background color to white
        widget.setStyleSheet("background-color: #FFFFFF;")

        # Page Title
        header_label = QLabel("លទ្ធផល និងការណែនាំអាជីព")
        header_label.setFont(QFont("Khmer OS Muol Light", 20))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Changed importance text/title color to 059212
        header_label.setStyleSheet("color: #059212; margin-top: 10px; margin-bottom: 10px;")
        main_layout.addWidget(header_label)

        # Content layout (image on left, details on right)
        content_h_layout = QHBoxLayout()
        content_h_layout.setSpacing(20)
        content_h_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Screenshot image on the left, now with a square frame
        self.results_banner_label = QLabel()
        self.results_banner_label.setFixedSize(200, 200) # Updated to be square
        self.results_banner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_banner_label.setScaledContents(True)
        # Changed text box content background color to F5F5F5
        self.results_banner_label.setStyleSheet("border: 2px solid #e0e0e0; border-radius: 15px; background-color: #F5F5F5;")
        content_h_layout.addWidget(self.results_banner_label)

        # Student and recommendation details on the right
        details_frame = QFrame()
        details_frame.setFrameShape(QFrame.Shape.StyledPanel)
        # Changed text box content background color to F5F5F5
        details_frame.setStyleSheet("background-color: #F5F5F5; border: 1px solid #e0e0e0; border-radius: 15px; padding: 25px;")
        details_layout = QVBoxLayout(details_frame)
        details_layout.setSpacing(10)

        self.results_student_name_label = QLabel("")
        self.results_student_name_label.setFont(QFont("Khmer OS Siemreap", 14, QFont.Weight.Bold))
        self.results_student_name_label.setStyleSheet("color: #333333;")
        details_layout.addWidget(self.results_student_name_label)

        self.recommended_career_label = QLabel("")
        self.recommended_career_label.setFont(QFont("Khmer OS Muol Light", 18, QFont.Weight.Bold))
        # Changed importance text/title color to 059212
        self.recommended_career_label.setStyleSheet("color: #059212;")
        details_layout.addWidget(self.recommended_career_label)
        
        self.recommended_industry_label = QLabel("")
        self.recommended_industry_label.setFont(QFont("Khmer OS Siemreap", 12))
        self.recommended_industry_label.setStyleSheet("color: #555555;")
        details_layout.addWidget(self.recommended_industry_label)

        self.recommended_score_label = QLabel("")
        self.recommended_score_label.setFont(QFont("Khmer OS Siemreap", 12, QFont.Weight.Bold))
        # Changed importance text/title color to 059212
        self.recommended_score_label.setStyleSheet("color: #059212;")
        details_layout.addWidget(self.recommended_score_label)

        content_h_layout.addWidget(details_frame)
        main_layout.addLayout(content_h_layout)

        # "See Job Details" button
        view_details_button = QPushButton("មើលព័ត៌មានលម្អិតការងារ")
        view_details_button.setFont(QFont("Khmer OS Siemreap", 12, QFont.Weight.Bold))
        view_details_button.setFixedSize(250, 55)
        # Changed primary color to 06D001 and hover to 059212
        view_details_button.setStyleSheet(
            "QPushButton { "
            "background-color: #06D001; color: white; border-radius: 27px; "
            "border: none; padding: 10px 15px; "
            "}"
            "QPushButton:hover { "
            "background-color: #059212; "
            "}"
        )
        view_details_button.clicked.connect(self.show_job_details_for_recommended_career)
        main_layout.addWidget(view_details_button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Removed the back to home button from the results page as requested
        
        main_layout.addStretch(1)

        return widget

    def _create_styled_label(self, text, bold=True):
        """Helper function to create a consistently styled label."""
        label = QLabel(text)
        label.setFont(QFont("Khmer OS Siemreap", 12, QFont.Weight.Bold if bold else QFont.Weight.Normal))
        label.setStyleSheet("color: #444444; margin-bottom: -5px;")
        return label

    def _create_styled_textedit(self, height=None):
        """Helper function to create a consistently styled QTextEdit."""
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Khmer OS Siemreap", 11))
        # Changed text box content background color to F5F5F5
        text_edit.setStyleSheet("background-color: #F5F5F5; border: 1px solid #ced4da; border-radius: 10px; padding: 10px;")
        if height:
            text_edit.setFixedHeight(height)
        return text_edit


    def create_job_details_page(self):
        """
        Creates the main view where job types are listed on the left and
        details are shown on the right. This will be the default page.
        """
        widget = QWidget()
        main_h_layout = QHBoxLayout(widget)
        main_h_layout.setContentsMargins(0, 0, 0, 0)
        # Changed page background color to white
        widget.setStyleSheet("background-color: #FFFFFF;")

        # Left Panel
        self.left_panel_for_stacked_widget = QFrame()
        self.left_panel_for_stacked_widget.setFrameShape(QFrame.Shape.StyledPanel)
        # Changed text box content background color to F5F5F5
        self.left_panel_for_stacked_widget.setStyleSheet("background-color: #F5F5F5; border-right: 1px solid #e0e0e0;")
        left_layout = QVBoxLayout(self.left_panel_for_stacked_widget)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(10)

        title_label_clone = QLabel("ប្រព័ន្ធវិភាគសមត្ថភាព និងផ្តល់យោបល់ការងារ")
        title_label_clone.setFont(QFont("Khmer OS Muol Light", 14, QFont.Weight.Bold))
        title_label_clone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label_clone.setWordWrap(True)
        left_layout.addWidget(title_label_clone)

        desc_label_clone = QLabel("សូមស្វាគមន៏!")
        desc_label_clone.setFont(QFont("Khmer OS Siemreap", 10))
        desc_label_clone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label_clone.setWordWrap(True)
        left_layout.addWidget(desc_label_clone)

        survey_button_clone = QPushButton("បំពេញការស្ទង់មតិ")
        survey_button_clone.setFont(QFont("Khmer OS Siemreap", 11))
        # Changed primary color to 06D001 and hover to 059212
        survey_button_clone.setStyleSheet(
            "QPushButton { background-color: #06D001; color: white; padding: 10px; border-radius: 27px; }"
            "QPushButton:hover { background-color: #059212; }"
        )
        survey_button_clone.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        left_layout.addWidget(survey_button_clone)

        history_button_clone = QPushButton("មើលប្រវត្តិការស្ទង់មតិ")
        history_button_clone.setFont(QFont("Khmer OS Siemreap", 11))
        history_button_clone.setStyleSheet(
            "QPushButton { background-color: #F72C5B; color: white; padding: 10px; border-radius: 27px; }"
            "QPushButton:hover { background-color: #FF748B; }"
        )
        history_button_clone.clicked.connect(self.show_history_page)
        left_layout.addWidget(history_button_clone)

        search_label_clone = QLabel("ស្វែងរកប្រភេទការងារ:")
        search_label_clone.setFont(QFont("Khmer OS Siemreap", 11, QFont.Weight.Bold))
        left_layout.addWidget(search_label_clone)

        self.search_input_job_details_page = QLineEdit()
        self.search_input_job_details_page.setPlaceholderText("ស្វែងរក...")
        self.search_input_job_details_page.setFont(QFont("Khmer OS Siemreap", 10))
        # Changed text box content background color to F5F5F5
        self.search_input_job_details_page.setStyleSheet("padding: 8px; border: 1px solid #ced4da; border-radius: 10px; background-color: #F5F5F5;")
        self.search_input_job_details_page.textChanged.connect(self.filter_career_list_job_details_page)
        left_layout.addWidget(self.search_input_job_details_page)

        self.career_list_widget_job_details_page = QListWidget()
        self.career_list_widget_job_details_page.setFont(QFont("Khmer OS Siemreap", 10))
        # Changed text box content background color to F5F5F5
        self.career_list_widget_job_details_page.setStyleSheet("border: 1px solid #ced4da; border-radius: 10px; background-color: #F5F5F5;")
        self.career_list_widget_job_details_page.itemClicked.connect(self.on_career_selected)
        left_layout.addWidget(self.career_list_widget_job_details_page)
        
        self.populate_career_list_job_details_page()

        left_layout.addStretch(1)
        self.left_panel_for_stacked_widget.setFixedWidth(300)
        self.career_list_widget_job_details_page.setMinimumHeight(250)
        main_h_layout.addWidget(self.left_panel_for_stacked_widget)


        # Right Panel (Detailed Job Display)
        self.right_content_widget = QWidget()
        self.detailed_job_display_layout = QVBoxLayout(self.right_content_widget)
        self.detailed_job_display_layout.setContentsMargins(20, 20, 20, 20)
        self.detailed_job_display_layout.setSpacing(15)

        self.scroll_area_right_panel = QScrollArea()
        self.scroll_area_right_panel.setWidgetResizable(True)
        self.scroll_area_right_panel.setWidget(self.right_content_widget)
        # Changed page background color to white
        self.scroll_area_right_panel.setStyleSheet("border: none; background-color: #FFFFFF;")
        
        main_h_layout.addWidget(self.scroll_area_right_panel, 1)

        return widget

    def populate_career_list_job_details_page(self):
        """Populates the list widget on the job details page."""
        self.career_list_widget_job_details_page.clear()
        self.career_list_widget_job_details_page.addItem("Intro")  # Add "Intro" at the top
        for career in sorted(JOB_DETAILS.keys()):
            self.career_list_widget_job_details_page.addItem(career)

    def filter_career_list_job_details_page(self, text):
        """Filters the career list on the job details page."""
        self.career_list_widget_job_details_page.clear()
        
        # Always keep "Intro" at the top regardless of search
        self.career_list_widget_job_details_page.addItem("Intro")
        
        for career in sorted(JOB_DETAILS.keys()):
            if text.lower() in career.lower():
                self.career_list_widget_job_details_page.addItem(career)

    def on_career_selected(self, item):
        """Handler for when a career is selected from the list."""
        career_name = item.text()
        if career_name == "Intro":
            self.display_intro_content()
        else:
            self.display_job_details(career_name)

    def display_intro_content(self):
        """Displays the intro content in the right panel."""
        # Clear the existing layout
        for i in reversed(range(self.detailed_job_display_layout.count())):
            widget = self.detailed_job_display_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        self.detailed_job_display_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # App title - New addition
        app_title_label = QLabel("ប្រព័ន្ធវិភាគសមត្ថភាព និងផ្តល់យោបល់ការងារ")
        app_title_label.setFont(QFont("Khmer OS Muol Light", 20, QFont.Weight.Bold))
        app_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        app_title_label.setWordWrap(True)
        # Changed importance text/title color to 059212
        app_title_label.setStyleSheet("color: #059212; margin-bottom: 10px;")
        self.detailed_job_display_layout.addWidget(app_title_label)
        

        # 2. App description
        app_description_label = QLabel(
            "កម្មវិធីនេះត្រូវបានរចនាឡើងដើម្បីជួយសិស្សនិស្សិតក្នុង"
            "ការស្វែងយល់ពីចំណង់ចំណូលចិត្ត សមត្ថភាព និងផ្តល់យោបល់អំពី"
            "ផ្លូវអាជីពដែលសមស្របទៅតាមចំណុចខ្លាំងរបស់ពួកគេ។"
        )
        app_description_label.setFont(QFont("Khmer OS Siemreap", 12))
        app_description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        app_description_label.setWordWrap(True)
        app_description_label.setStyleSheet("margin-top: 10px; margin-bottom: 20px; color: #555;")
        self.detailed_job_display_layout.addWidget(app_description_label)
        
        # 3. Allbots image
        allbots_image_label = QLabel()
        pixmap = QPixmap("img/allbots.png").scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation) # Increased size
        allbots_image_label.setPixmap(pixmap)
        allbots_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        allbots_image_label.setStyleSheet("margin-top: 20px; margin-bottom: 20px; border: none;") # Removed border
        self.detailed_job_display_layout.addWidget(allbots_image_label)
        
        # 4. Instruction for user - Added as requested
        instruction_label = QLabel("ចុចប៊ូតុង 'បំពេញការស្ទង់មតិ' ខាងឆ្វេង ដើម្បីចាប់ផ្តើម!")
        instruction_label.setFont(QFont("Khmer OS Siemreap", 12, QFont.Weight.Bold))
        instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instruction_label.setStyleSheet("margin-top: 20px;")
        self.detailed_job_display_layout.addWidget(instruction_label)
        
        self.detailed_job_display_layout.addStretch(1)


    def display_job_details(self, career_name):
        """
        Populates the detailed job display area with information for the given career,
        including its associated image.
        """
        for i in reversed(range(self.detailed_job_display_layout.count())):
            widget = self.detailed_job_display_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.detailed_job_display_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        if career_name in JOB_DETAILS:
            details = JOB_DETAILS[career_name]

            job_title_label = QLabel(f"ព័ត៌មានលម្អិតសម្រាប់៖ {career_name}")
            job_title_label.setFont(QFont("Khmer OS Muol Light", 16, QFont.Weight.Bold))
            # Changed importance text/title color to 059212
            job_title_label.setStyleSheet("color: #059212; margin-bottom: 10px;")
            job_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.detailed_job_display_layout.addWidget(job_title_label)

            if 'image_path' in details and details['image_path']:
                job_image_label = QLabel()
                pixmap = QPixmap(details['image_path']).scaled(400, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                if pixmap.isNull():
                    print(f"Error: Could not load image for {career_name} from {details['image_path']}.")
                job_image_label.setText("រូបភាពមិនមាន")
                job_image_label.setFont(QFont("Khmer OS Siemreap", 12, QFont.Weight.Normal, True))
                job_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                job_image_label.setStyleSheet("color: #888; margin-bottom: 15px;")
                
                job_image_label.setPixmap(pixmap)
                job_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                job_image_label.setStyleSheet("margin-bottom: 15px; border-radius: 10px;")
                self.detailed_job_display_layout.addWidget(job_image_label)
            else:
                no_image_label = QLabel("រូបភាពមិនមាន")
                no_image_label.setFont(QFont("Khmer OS Siemreap", 12, QFont.Weight.Normal, True))
                no_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                no_image_label.setStyleSheet("color: #888; margin-bottom: 15px;")
                self.detailed_job_display_layout.addWidget(no_image_label)


            desc_label = QLabel("អំពីការងារ:")
            desc_label.setFont(QFont("Khmer OS Siemreap", 12, QFont.Weight.Bold))
            self.detailed_job_display_layout.addWidget(desc_label)
            description_text = QTextEdit()
            description_text.setReadOnly(True)
            description_text.setFont(QFont("Khmer OS Siemreap", 11))
            description_text.setHtml(f"<p>{details['description']}</p>")
            description_text.setFixedHeight(120)
            # Changed text box content background color to F5F5F5
            description_text.setStyleSheet("background-color: #F5F5F5; border: 1px solid #ced4da; border-radius: 10px; padding: 10px;")
            self.detailed_job_display_layout.addWidget(description_text)

            skills_label = QLabel("តម្រូវការជំនាញនេះ:")
            skills_label.setFont(QFont("Khmer OS Siemreap", 12, QFont.Weight.Bold))
            self.detailed_job_display_layout.addWidget(skills_label)
            skills_text = QTextEdit()
            skills_text.setReadOnly(True)
            skills_text.setFont(QFont("Khmer OS Siemreap", 11))
            skills_list = "".join([f"<li>{skill}</li>" for skill in details.get('skills', [])])
            skills_text.setHtml(f"<ul>{skills_list}</ul>")
            skills_text.setFixedHeight(100)
            # Changed text box content background color to F5F5F5
            skills_text.setStyleSheet("background-color: #F5F5F5; border: 1px solid #ced4da; border-radius: 10px; padding: 10px;")
            self.detailed_job_display_layout.addWidget(skills_text)

            schools_label = QLabel("សាលាដែលមានបង្រៀនជំនាញនេះ:")
            schools_label.setFont(QFont("Khmer OS Siemreap", 12, QFont.Weight.Bold))
            self.detailed_job_display_layout.addWidget(schools_label)
            schools_text = QTextEdit()
            schools_text.setReadOnly(True)
            schools_text.setFont(QFont("Khmer OS Siemreap", 11))
            schools_list = "".join([f"<li>{school}</li>" for school in details.get('schools', [])])
            schools_text.setHtml(f"<ul>{schools_list}</ul>")
            schools_text.setFixedHeight(80)
            # Changed text box content background color to F5F5F5
            schools_text.setStyleSheet("background-color: #F5F5F5; border: 1px solid #ced4da; border-radius: 10px; padding: 10px;")
            self.detailed_job_display_layout.addWidget(schools_text)


            companies_label = QLabel("ក្រុមហ៊ុនដែលអាចរកការងារនេះបានមាន:")
            companies_label.setFont(QFont("Khmer OS Siemreap", 12, QFont.Weight.Bold))
            self.detailed_job_display_layout.addWidget(companies_label)
            companies_text = QTextEdit()
            companies_text.setReadOnly(True)
            companies_text.setFont(QFont("Khmer OS Siemreap", 11))
            companies_list = "".join([f"<li>{company}</li>" for company in details.get('companies', [])])
            companies_text.setHtml("<ul>" + companies_list + "</ul>")
            companies_text.setFixedHeight(100)
            # Changed text box content background color to F5F5F5
            companies_text.setStyleSheet("background-color: #F5F5F5; border: 1px solid #ced4da; border-radius: 10px; padding: 10px;")
            self.detailed_job_display_layout.addWidget(companies_text)

            salary_label = QLabel("ប្រាក់ខែដែលអាចទទួលបាន:")
            salary_label.setFont(QFont("Khmer OS Siemreap", 12, QFont.Weight.Bold))
            self.detailed_job_display_layout.addWidget(salary_label)
            salary_value_label = QLabel(details['salary_range'])
            salary_value_label.setFont(QFont("Khmer OS Siemreap", 11))
            # Changed text box content background color to F5F5F5
            salary_frame = QFrame()
            salary_frame.setStyleSheet("background-color: #F5F5F5; border: 1px solid #ced4da; border-radius: 10px; padding: 10px;")
            salary_frame_layout = QHBoxLayout(salary_frame)
            salary_frame_layout.addWidget(salary_value_label)
            self.detailed_job_display_layout.addWidget(salary_frame)
            
        else:
            no_details_label = QLabel("សូមជ្រើសរើសប្រភេទការងារពីបញ្ជីខាងឆ្វេង។")
            no_details_label.setFont(QFont("Khmer OS Siemreap", 12))
            no_details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.detailed_job_display_layout.addWidget(no_details_label)

        self.detailed_job_display_layout.addStretch(1)


    def create_history_page(self):
        """Creates the history display page with a pie chart and student data side-by-side."""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(15)
        # Changed page background color to white
        widget.setStyleSheet("background-color: #FFFFFF;")

        header_label = QLabel("ប្រវត្តិការស្ទង់មតិ និងស្ថិតិ")
        header_label.setFont(QFont("Khmer OS Muol Light", 18))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Changed importance text/title color to 059212
        header_label.setStyleSheet("color: #059212; margin-bottom: 25px;")
        main_layout.addWidget(header_label)
        
        # --- Horizontal layout for content ---
        content_h_layout = QHBoxLayout()
        
        # Left widget for student data
        student_data_frame = QFrame()
        student_data_frame.setFrameShape(QFrame.Shape.StyledPanel)
        # Changed text box content background color to F5F5F5
        student_data_frame.setStyleSheet("background-color: #F5F5F5; border: 1px solid #e0e0e0; border-radius: 10px; padding: 10px;")
        student_data_layout = QVBoxLayout(student_data_frame)
        
        student_data_header = QLabel("ទិន្នន័យស្ទង់មតិរបស់និស្សិត")
        student_data_header.setFont(QFont("Khmer OS Siemreap", 12, QFont.Weight.Bold))
        student_data_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        student_data_layout.addWidget(student_data_header)
        
        self.history_text_area = QTextEdit()
        self.history_text_area.setReadOnly(True)
        self.history_text_area.setFont(QFont("Khmer OS Siemreap", 10))
        self.history_text_area.setStyleSheet("background-color: #ffffff; border-radius: 8px; padding: 10px;")
        student_data_layout.addWidget(self.history_text_area)
        
        content_h_layout.addWidget(student_data_frame, 1)

        # Right widget for the pie chart
        chart_frame = QFrame()
        chart_frame.setFrameShape(QFrame.Shape.StyledPanel)
        chart_frame.setStyleSheet("background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 10px; padding: 10px;")
        chart_layout = QVBoxLayout(chart_frame)
        
        chart_header = QLabel("ស្ថិតិអាជីពដែលបានណែនាំ")
        chart_header.setFont(QFont("Khmer OS Siemreap", 12, QFont.Weight.Bold))
        chart_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chart_layout.addWidget(chart_header)
        
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        chart_layout.addWidget(self.canvas)
        
        content_h_layout.addWidget(chart_frame, 1)
        
        main_layout.addLayout(content_h_layout)

        back_button = QPushButton("ត្រឡប់ទៅទំព័រដើម")
        back_button.setFont(QFont("Khmer OS Siemreap", 11))
        back_button.setFixedSize(180, 65) # Changed size
        back_button.setStyleSheet(
            "QPushButton { "
            "background-color: #F72C5B; color: white; border-radius: 25px; "
            "border: none; padding: 10px 20px; margin-top: 20px; "
            "}"
            "QPushButton:hover { "
            "background-color: #FF748B; "
            "}"
        )
        back_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0)) # Back to the main job details page (now the hub)
        main_layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignCenter)

        return widget

    def show_history_page(self):
        """Populates and displays the history page, including the pie chart."""
        self.history_text_area.clear()
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT id, student_name, preferred_industry, recommended_career, recommendation_score, timestamp, raw_survey_responses FROM survey_responses ORDER BY timestamp DESC")
        results = cursor.fetchall()
        
        cursor.execute("SELECT recommended_career, COUNT(*) FROM survey_responses GROUP BY recommended_career ORDER BY COUNT(*) DESC")
        job_counts_results = cursor.fetchall()
        conn.close()

        # --- Display student data in the left widget ---
        if not results:
            self.history_text_area.setText("មិនទាន់មានទិន្នន័យប្រវត្តិស្ទង់មតិនៅឡើយទេ។")
            self.history_text_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            history_html = "<h3>ប្រវត្តិនៃលទ្ធផលការស្ទង់មតិ:</h3><br>"
            for i, row in enumerate(results):
                id, student_name, preferred_industry, recommended_career, recommendation_score, timestamp, raw_survey_responses_json = row
                
                history_html += (
                    f"<div style='margin-bottom: 20px; padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #f9f9f9;'>"
                    f"<strong>ID:</strong> {id}<br>"
                    f"<strong>ឈ្មោះនិស្សិត:</strong> {student_name}<br>"
                    f"<strong>ឧស្សាហកម្មពេញចិត្ត:</strong> {preferred_industry}<br>"
                    f"<strong>អាជីពណែនាំ:</strong> {recommended_career}<br>"
                    f"<strong>អត្រាសមត្ថភាព:</strong> {recommendation_score:.2f}%<br>"
                    f"<strong>កាលបរិច្ឆេទ:</strong> {timestamp}<br>"
                )

                if raw_survey_responses_json:
                    history_html += "<br><strong>ចម្លើយស្ទង់មតិ:</strong><br>"
                    raw_responses = json.loads(raw_survey_responses_json)
                    sorted_q_keys = sorted(raw_responses.keys(), key=lambda x: int(x[1:]))
                    for q_key in sorted_q_keys:
                        q_num = int(q_key[1:])
                        question_text = self.questions[q_num - 1] if q_num <= len(self.questions) else f"សំណួរ {q_num}"
                        
                        response_value = raw_responses[q_key]
                        response_map = {
                            1: "Strongly Disagree", 2: "Disagree", 3: "Slightly Disagree",
                            4: "Neutral", 5: "Slightly Agree", 6: "Agree", 7: "Strongly Agree"
                        }
                        display_response = response_map.get(response_value, f"Value ({response_value})")

                        history_html += f"&nbsp;&nbsp;&nbsp;&nbsp;<strong>សំណួរ {q_num}:</strong> {question_text}<br>"
                        history_html += f"&nbsp;&nbsp;&nbsp;&nbsp;<strong>ចម្លើយ:</strong> {display_response} ({response_value})<br>"
                history_html += "</div>"
            self.history_text_area.setHtml(history_html)
            self.history_text_area.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # --- Draw the pie chart in the right widget ---
        self.ax.clear()
        if not job_counts_results:
            self.ax.text(0.5, 0.5, "No data for chart.", horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes, fontsize=12)
            self.canvas.draw()
        else:
            labels = [row[0] for row in job_counts_results]
            sizes = [row[1] for row in job_counts_results]
            
            # Autopct formats the percentage, startangle rotates the chart
            self.ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
            self.ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            self.fig.tight_layout()
            self.canvas.draw()
        
        self.stacked_widget.setCurrentIndex(3)


    def show_results_page(self, student_name, preferred_industry, recommended_career, recommendation_score):
        """
        Switches to the results page and displays the recommendation.
        """
        # Remove the image from the results page
        self.results_banner_label.setVisible(False)

        # Update the labels with dynamic data
        self.results_student_name_label.setText(f"ឈ្មោះនិស្សិត: {student_name}")
        self.recommended_industry_label.setText(f"ឧស្សាហកម្មដែលពេញចិត្ត: {preferred_industry}")
        self.recommended_career_label.setText(f"អាជីពដែលបានណែនាំ: {recommended_career}")
        self.recommended_score_label.setText(f"អត្រាសមត្ថភាព: {recommendation_score:.2f}%")

        self.stacked_widget.setCurrentIndex(2)

    def show_job_details_for_recommended_career(self):
        """
        Switches to the job details page and automatically displays the details
        for the most recently recommended career.
        """
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT recommended_career FROM survey_responses ORDER BY timestamp DESC LIMIT 1")
        last_recommended_career = cursor.fetchone()
        conn.close()

        if last_recommended_career:
            # Find the item in the list and select it
            career_name = last_recommended_career[0]
            items = self.career_list_widget_job_details_page.findItems(career_name, Qt.MatchFlag.MatchExactly)
            if items:
                self.career_list_widget_job_details_page.setCurrentItem(items[0])
                self.display_job_details(career_name) # Manually trigger display
            
            self.stacked_widget.setCurrentIndex(0)
        else:
            QMessageBox.information(self, "No Recommendation Yet", "សូមបំពេញការស្ទង់មតិជាមុនសិន ដើម្បីទទួលបានការណែនាំអាជីព។")


    def submit_survey(self):
        """Collects survey responses, processes them, and displays results."""
        student_name = self.student_name_input.text().strip()
        if not student_name:
            QMessageBox.warning(self, "Missing Information", "សូមបញ្ចូលឈ្មោះនិស្សិត។")
            return

        raw_responses = {}
        all_questions_answered = True
        for q_key, button_group in self.question_button_groups.items():
            checked_button = button_group.checkedButton()
            if checked_button:
                raw_responses[q_key] = checked_button.property("value")
            else:
                all_questions_answered = False
                break
        
        if not all_questions_answered:
            QMessageBox.warning(self, "Missing Responses", "សូមឆ្លើយគ្រប់សំណួរទាំងអស់។")
            return

        preferred_industry = self.survey_industry_combo.currentText()

        # Get ML recommendation
        recommended_career, recommendation_score, top_careers_for_display = \
            get_ml_career_recommendation(
                self.ml_model, self.feature_names, self.career_outcomes,
                raw_responses, preferred_industry
            )

        # Save to database
        try:
            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO survey_responses (student_name, raw_survey_responses, preferred_industry, recommended_career, recommendation_score) VALUES (?, ?, ?, ?, ?)",
                (student_name, json.dumps(raw_responses), preferred_industry, recommended_career, recommendation_score)
            )
            conn.commit()
            conn.close()
            QMessageBox.information(self, "Submission Successful", "ការស្ទង់មតិត្រូវបានដាក់ស្នើដោយជោគជ័យ!")
            # Correctly pass arguments to show_results_page
            self.show_results_page(student_name, preferred_industry, recommended_career, recommendation_score)
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Database Error", f"មានបញ្ហាជាមួយមូលដ្ឋានទិន្នន័យ: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CareerApp()
    window.show()
    sys.exit(app.exec())