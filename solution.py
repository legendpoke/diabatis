"""
Comprehensive Health Assessment System
Diabetes Risk Prediction + Symptom Analysis + PDF Report Generation

Dependencies:
pip install scikit-learn pandas numpy langchain langchain-openai openai fpdf
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# PDF generation
from fpdf import FPDF

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ============================================================================
# 1. ML RISK PREDICTION ENGINE
# ============================================================================

class DiabetesRiskPredictor:
    """ML model for diabetes risk prediction with uncertainty quantification"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'hba1c', 'fasting_glucose', 'cholesterol_total', 'hdl', 'ldl',
            'triglycerides', 'family_history', 'smoking', 'physical_activity_hours'
        ]
        self.is_trained = False
    
    def generate_synthetic_training_data(self, n_samples=1000):
        """Generate realistic synthetic patient data for demonstration"""
        np.random.seed(42)
        
        # Generate features with realistic distributions
        age = np.random.normal(50, 15, n_samples).clip(18, 85)
        bmi = np.random.normal(28, 6, n_samples).clip(15, 50)
        bp_sys = np.random.normal(130, 20, n_samples).clip(90, 200)
        bp_dia = np.random.normal(85, 12, n_samples).clip(60, 120)
        hba1c = np.random.normal(5.7, 0.8, n_samples).clip(4.0, 10.0)
        glucose = np.random.normal(105, 20, n_samples).clip(70, 200)
        chol = np.random.normal(200, 40, n_samples).clip(120, 350)
        hdl = np.random.normal(50, 15, n_samples).clip(20, 100)
        ldl = np.random.normal(120, 35, n_samples).clip(50, 250)
        trig = np.random.normal(150, 50, n_samples).clip(50, 400)
        family_hist = np.random.binomial(1, 0.3, n_samples)
        smoking = np.random.binomial(1, 0.2, n_samples)
        activity = np.random.exponential(3, n_samples).clip(0, 20)
        
        # Generate target based on risk factors
        risk_score = (
            (age > 45) * 0.2 +
            (bmi > 30) * 0.3 +
            (bp_sys > 140) * 0.15 +
            (hba1c > 6.0) * 0.4 +
            (glucose > 125) * 0.3 +
            family_hist * 0.25 +
            smoking * 0.1 +
            (activity < 2) * 0.15 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        target = (risk_score > 0.5).astype(int)
        
        X = pd.DataFrame({
            'age': age,
            'bmi': bmi,
            'blood_pressure_systolic': bp_sys,
            'blood_pressure_diastolic': bp_dia,
            'hba1c': hba1c,
            'fasting_glucose': glucose,
            'cholesterol_total': chol,
            'hdl': hdl,
            'ldl': ldl,
            'triglycerides': trig,
            'family_history': family_hist,
            'smoking': smoking,
            'physical_activity_hours': activity
        })
        
        return X, target
    
    def train(self):
        """Train the ML model"""
        X, y = self.generate_synthetic_training_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Calculate accuracy
        X_test_scaled = self.scaler.transform(X_test)
        accuracy = self.model.score(X_test_scaled, y_test)
        print(f"Model trained. Test accuracy: {accuracy:.2%}")
        
        return accuracy
    
    def predict_risk(self, patient_data):
        """
        Predict diabetes risk with uncertainty quantification
        
        Returns: dict with risk_score, confidence, risk_level, and feature_importance
        """
        if not self.is_trained:
            self.train()
        
        # Prepare patient data
        patient_df = pd.DataFrame([patient_data])[self.feature_names]
        patient_scaled = self.scaler.transform(patient_df)
        
        # Get prediction probability
        risk_prob = self.model.predict_proba(patient_scaled)[0][1]
        
        # Calculate uncertainty using ensemble variance
        # (in production, use calibrated uncertainty)
        confidence = min(max(abs(risk_prob - 0.5) * 2, 0.6), 0.95)
        
        # Get feature importance for this prediction
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Determine risk level
        if risk_prob >= 0.7:
            risk_level = "High"
        elif risk_prob >= 0.4:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        return {
            'risk_score': float(risk_prob),
            'confidence': float(confidence),
            'risk_level': risk_level,
            'feature_importance': feature_importance,
            'prediction_date': datetime.now().isoformat()
        }
    
    def get_counterfactual_scenarios(self, patient_data, target_risk=0.3):
        """
        Generate counterfactual: what changes would reduce risk?
        """
        current_risk = self.predict_risk(patient_data)['risk_score']
        scenarios = []
        
        # Modifiable factors
        modifiable = {
            'bmi': {'min': 18.5, 'target': 25, 'current': patient_data['bmi']},
            'blood_pressure_systolic': {'min': 90, 'target': 120, 'current': patient_data['blood_pressure_systolic']},
            'hba1c': {'min': 4.0, 'target': 5.5, 'current': patient_data['hba1c']},
            'physical_activity_hours': {'min': 0, 'target': 5, 'current': patient_data['physical_activity_hours']},
            'smoking': {'min': 0, 'target': 0, 'current': patient_data['smoking']}
        }
        
        for factor, values in modifiable.items():
            modified_data = patient_data.copy()
            modified_data[factor] = values['target']
            new_risk = self.predict_risk(modified_data)['risk_score']
            risk_reduction = current_risk - new_risk
            
            if risk_reduction > 0.05:  # Only significant changes
                scenarios.append({
                    'factor': factor,
                    'current_value': values['current'],
                    'target_value': values['target'],
                    'risk_reduction': float(risk_reduction),
                    'new_risk_score': float(new_risk)
                })
        
        # Sort by impact
        scenarios.sort(key=lambda x: x['risk_reduction'], reverse=True)
        return scenarios


# ============================================================================
# 2. SYMPTOM ANALYSIS ENGINE
# ============================================================================

class SymptomAnalyzer:
    """Analyze symptoms to identify potential diseases"""
    
    def __init__(self, openai_api_key):
        if not openai_api_key or "test" in openai_api_key.lower():
            raise ValueError("Please set a valid OpenAI API key")
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,  # Low temperature for medical accuracy
            openai_api_key=openai_api_key
        )
    
    def analyze_symptoms(self, symptoms_text, age=None, gender=None, existing_conditions=None):
        """Analyze symptoms and provide differential diagnosis"""
        
        prompt = PromptTemplate(
            input_variables=["symptoms", "age", "gender", "existing_conditions"],
            template="""You are a medical AI assistant analyzing patient symptoms for potential diseases.

Patient Information:
- Age: {age if age else 'Not specified'}
- Gender: {gender if gender else 'Not specified'}
- Existing Conditions: {existing_conditions if existing_conditions else 'None reported'}

Reported Symptoms: {symptoms}

Analyze these symptoms and provide:

1. DIFFERENTIAL DIAGNOSIS:
   - List top 3-5 most likely conditions based on symptoms
   - For each condition, provide:
     * Probability level (High/Medium/Low)
     * Key matching symptoms
     * Additional symptoms to check for

2. URGENCY ASSESSMENT:
   - How urgent is medical attention needed? (Emergency/Urgent/Routine)
   - Red flags or warning signs present

3. SPECIFIC DIABETES ASSESSMENT:
   - How many diabetes symptoms are present?
   - Are there classic diabetes indicators?
   - Risk level for diabetes based on symptoms alone

4. RECOMMENDED NEXT STEPS:
   - Immediate actions if needed
   - Tests or specialists to consider
   - When to seek emergency care

Be thorough but concise. Use medical terminology appropriately.
Always emphasize that this is for informational purposes only, not medical diagnosis.
"""
        )
        
        chain = prompt | self.llm
        
        response = chain.invoke({
            "symptoms": symptoms_text,
            "age": str(age) if age else "Not specified",
            "gender": gender if gender else "Not specified",
            "existing_conditions": existing_conditions if existing_conditions else "None reported"
        })
        
        # Also get structured output for diseases detection
        output_parser = JsonOutputParser()
        
        structured_prompt = PromptTemplate(
            input_variables=["symptoms"],
            partial_variables={"format_instructions": """Return a valid JSON object with these keys:
- likely_conditions: array of objects with 'disease', 'probability', 'matching_symptoms'
- diabetes_symptoms_present: array of diabetes symptoms found
- diabetes_risk_level: string (High/Medium/Low)
- urgency_level: string (Emergency/Urgent/Routine)
- immediate_concerns: array of concerning findings"""},
            template="""
Based on these symptoms: {symptoms}

Analyze for disease patterns. Return ONLY valid JSON.

{format_instructions}
"""
        )
        
        structured_chain = structured_prompt | self.llm | output_parser
        
        try:
            structured_analysis = structured_chain.invoke({
                "symptoms": symptoms_text
            })
        except:
            structured_analysis = {"error": "Could not parse structured analysis"}
        
        return {
            "detailed_analysis": response.content,
            "structured_analysis": structured_analysis
        }


# ============================================================================
# 3. GenAI EXPLANATION & RECOMMENDATION ENGINE
# ============================================================================

class ClinicalInsightGenerator:
    """LangChain-based GenAI engine for clinical explanations"""
    
    def __init__(self, openai_api_key):
        if not openai_api_key or "test" in openai_api_key.lower():
            raise ValueError("Please set a valid OpenAI API key")
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,  # Lower temperature for medical accuracy
            openai_api_key=openai_api_key
        )
    
    def generate_clinician_report(self, patient_data, ml_results, counterfactuals, symptom_analysis=None):
        """Generate detailed clinical report for healthcare providers"""
        
        symptom_context = ""
        if symptom_analysis:
            symptom_context = f"\nSymptom Analysis:\n{symptom_analysis}"
        
        prompt = PromptTemplate(
            input_variables=["patient_data", "ml_results", "counterfactuals", "symptom_context"],
            template="""You are a clinical decision support AI assistant helping physicians.

Patient Data:
{patient_data}

ML Risk Assessment:
{ml_results}

Counterfactual Analysis (What-if scenarios):
{counterfactuals}
{symptom_context}

Generate a concise clinical report with:

1. RISK SUMMARY (2-3 sentences)
   - Current risk level and confidence
   - Key contributing factors

2. SYMPTOM-DISEASE CORRELATION
   - How symptoms align with diabetes risk
   - Other potential conditions to consider

3. MODIFIABLE RISK DRIVERS (prioritized list)
   - Top 3 factors the patient can change
   - Expected impact of each intervention

4. RECOMMENDED ACTIONS
   - Immediate tests/referrals needed
   - Lifestyle intervention priorities
   - Follow-up timeline

5. CLINICAL NOTES
   - Any red flags or urgent concerns
   - Considerations for shared decision-making

Be specific, actionable, and evidence-based. Use medical terminology appropriate for clinicians.
"""
        )
        chain = prompt | self.llm

        report = chain.invoke({
            "patient_data": json.dumps(patient_data, indent=2),
            "ml_results": json.dumps(ml_results, indent=2),
            "counterfactuals": json.dumps(counterfactuals, indent=2),
            "symptom_context": symptom_context
        })

        return report.content
    
    def generate_patient_explanation(self, patient_data, ml_results, counterfactuals, symptom_analysis=None):
        """Generate patient-friendly explanation"""
        
        symptom_context = ""
        if symptom_analysis:
            symptom_context = f"\nYour Reported Symptoms:\n{symptom_analysis}"
        
        prompt = PromptTemplate(
            input_variables=["patient_data", "ml_results", "counterfactuals", "symptom_context"],
            template="""You are a health educator explaining health risks to a patient.

Patient's Health Data:
{patient_data}

Risk Assessment:
{ml_results}

What Changes Could Help:
{counterfactuals}
{symptom_context}

Create a clear, empathetic explanation for the patient that includes:

1. UNDERSTANDING YOUR SYMPTOMS
   - What your symptoms might indicate
   - How they relate to diabetes risk

2. YOUR HEALTH STATUS (plain language)
   - What the numbers mean
   - Why we're checking these things

3. YOUR DIABETES RISK (avoid scary language)
   - What this risk level means for you
   - This is preventable and manageable

4. OTHER HEALTH CONSIDERATIONS
   - Other conditions your symptoms might suggest
   - When to seek immediate care

5. WHAT YOU CAN DO (concrete, encouraging)
   - Top 3 changes that would help most
   - How each change helps reduce your risk
   - Realistic goals you can start with

6. NEXT STEPS
   - What to expect from your doctor
   - Timeline for check-ins

Use simple language (8th grade reading level). Be encouraging and empowering, not alarming.
Avoid medical jargon. Focus on what the patient can control.
"""
        )
        chain = prompt | self.llm

        response = chain.invoke({
            "patient_data": json.dumps(patient_data, indent=2),
            "ml_results": json.dumps(ml_results, indent=2),
            "counterfactuals": json.dumps(counterfactuals, indent=2),
            "symptom_context": symptom_context
        })

        return response.content
        
    def generate_intervention_plan(self, patient_data, ml_results, counterfactuals, symptom_analysis=None):
        """Generate structured intervention recommendations"""
        
        symptom_context = ""
        if symptom_analysis:
            symptom_context = f"\nSymptom Analysis: {symptom_analysis}"
        
        format_instructions = """
           Return a valid JSON object with these keys:
         - immediate_actions
         - short_term_goals
         - lifestyle_changes
         - monitoring_plan
         - red_flags
         - specialist_referrals
         - urgent_care_needed (boolean)
        """
        prompt = PromptTemplate(
            input_variables=["patient_data", "ml_results", "counterfactuals", "symptom_context"],
            partial_variables={"format_instructions": format_instructions},
            template="""
            Based on this patient's diabetes risk assessment and symptom analysis, create a structured intervention plan.

Patient Data: {patient_data}
Risk Assessment: {ml_results}
Impact Analysis: {counterfactuals}
{symptom_context}

{format_instructions}

Return ONLY valid JSON. No markdown. No explanations.
"""
        )
        output_parser = JsonOutputParser()

        chain = prompt | self.llm | output_parser

        result = chain.invoke({
            "patient_data": json.dumps(patient_data, indent=2),
            "ml_results": json.dumps(ml_results, indent=2),
            "counterfactuals": json.dumps(counterfactuals, indent=2),
            "symptom_context": symptom_context
        })

        return result


# ============================================================================
# 4. PDF REPORT GENERATOR (FIXED VERSION)
# ============================================================================

class PDFReportGenerator:
    """Generate PDF reports from analysis results with Unicode support"""
    
    def __init__(self):
        self.pdf = None
    
    def _clean_text(self, text):
        """Clean text to remove or replace Unicode characters that cause issues"""
        if not text:
            return ""
        
        # Replace common Unicode characters with ASCII equivalents
        replacements = {
            '•': '-',      # Bullet point to dash
            '→': '->',     # Right arrow
            '–': '-',      # En dash
            '—': '-',      # Em dash
            '°': ' deg',   # Degree symbol
            '±': '+/-',    # Plus-minus
            '≥': '>=',     # Greater than or equal
            '≤': '<=',     # Less than or equal
            '×': 'x',      # Multiplication sign
            '÷': '/',      # Division sign
            'α': 'alpha',  # Greek letters
            'β': 'beta',
            'μ': 'mu',
            '…': '...',    # Ellipsis
        }
        
        # First, decode if needed
        if isinstance(text, str):
            for old, new in replacements.items():
                text = text.replace(old, new)
        
        return text
    
    def _add_section(self, pdf, title, content):
        """Add a section with proper formatting"""
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=self._clean_text(title), ln=1)
        pdf.set_font("Arial", '', 12)
        
        if content:
            # Clean and split content
            cleaned_content = self._clean_text(content)
            
            # Handle very long content by splitting into chunks
            paragraphs = cleaned_content.split('\n')
            for para in paragraphs:
                if para.strip():
                    # Split long paragraphs into multiple lines
                    words = para.split()
                    lines = []
                    current_line = ""
                    
                    for word in words:
                        if len(current_line) + len(word) + 1 <= 80:  # 80 chars per line
                            if current_line:
                                current_line += " " + word
                            else:
                                current_line = word
                        else:
                            lines.append(current_line)
                            current_line = word
                    
                    if current_line:
                        lines.append(current_line)
                    
                    # Add lines to PDF
                    for line in lines:
                        pdf.cell(0, 8, txt=line, ln=1)
                    pdf.ln(2)  # Small gap between paragraphs
        
        pdf.ln(5)
    
    def _add_list(self, pdf, title, items):
        """Add a list section"""
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=self._clean_text(title), ln=1)
        pdf.set_font("Arial", '', 12)
        
        if items:
            if isinstance(items, list):
                for item in items:
                    cleaned_item = self._clean_text(str(item))
                    pdf.cell(10)  # Indent
                    pdf.cell(0, 8, txt=f"- {cleaned_item}", ln=1)
            else:
                cleaned_item = self._clean_text(str(items))
                pdf.cell(0, 8, txt=f"- {cleaned_item}", ln=1)
        
        pdf.ln(5)
    
    def create_patient_report(self, patient_info, analysis_results, filename="health_report.pdf"):
        """Create a comprehensive PDF health report with Unicode handling"""
        
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt="COMPREHENSIVE HEALTH ASSESSMENT REPORT", ln=1, align='C')
        pdf.ln(10)
        
        # 1. PATIENT INFORMATION
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="1. PATIENT INFORMATION", ln=1)
        pdf.set_font("Arial", '', 12)
        
        patient_details = [
            f"Name: {patient_info.get('name', 'Not Provided')}",
            f"Age: {patient_info.get('age', 'Not Provided')}",
            f"Gender: {patient_info.get('gender', 'Not Provided')}",
            f"Height: {patient_info.get('height', 'Not Provided')}",
            f"Weight: {patient_info.get('weight', 'Not Provided')}",
            f"BMI: {patient_info.get('bmi', 'Not Provided')}",
            f"Existing Conditions: {patient_info.get('existing_conditions', 'None')}",
            f"Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        for detail in patient_details:
            pdf.cell(0, 8, txt=self._clean_text(detail), ln=1)
        
        pdf.ln(10)
        
        # 2. DIABETES RISK ASSESSMENT
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="2. DIABETES RISK ASSESSMENT", ln=1)
        pdf.set_font("Arial", '', 12)
        
        ml_results = analysis_results.get('ml_prediction', {})
        if ml_results:
            risk_score = ml_results.get('risk_score', 0) * 100
            risk_level = ml_results.get('risk_level', 'Unknown')
            confidence = ml_results.get('confidence', 0) * 100
            
            risk_details = [
                f"Risk Score: {risk_score:.1f}%",
                f"Risk Level: {risk_level}",
                f"Confidence: {confidence:.1f}%",
                f"Assessment Date: {ml_results.get('prediction_date', 'N/A')}"
            ]
            
            for detail in risk_details:
                pdf.cell(0, 8, txt=detail, ln=1)
        
        pdf.ln(10)
        
        # 3. SYMPTOM ANALYSIS (if available)
        if analysis_results.get('symptom_analysis'):
            self._add_section(pdf, "3. SYMPTOM ANALYSIS", 
                            analysis_results['symptom_analysis'])
        
        # 4. KEY FINDINGS FROM SYMPTOM ANALYSIS
        structured_analysis = analysis_results.get('structured_symptom_analysis', {})
        if structured_analysis and isinstance(structured_analysis, dict):
            if structured_analysis.get('likely_conditions'):
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, txt="4. DETECTED CONDITIONS", ln=1)
                pdf.set_font("Arial", '', 12)
                
                conditions = structured_analysis.get('likely_conditions', [])
                for i, condition in enumerate(conditions[:3], 1):
                    if isinstance(condition, dict):
                        disease = condition.get('disease', 'Unknown')
                        probability = condition.get('probability', 'Unknown')
                        symptoms = condition.get('matching_symptoms', [])
                        
                        pdf.cell(0, 8, txt=f"{i}. {disease} ({probability} probability)", ln=1)
                        if symptoms:
                            pdf.cell(10)
                            pdf.cell(0, 8, txt=f"Matching symptoms: {', '.join(symptoms[:3])}", ln=1)
                        pdf.ln(2)
                
                pdf.ln(5)
        
        # 5. RISK REDUCTION STRATEGIES
        counterfactuals = analysis_results.get('counterfactuals', [])
        if counterfactuals:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, txt="5. RISK REDUCTION STRATEGIES", ln=1)
            pdf.set_font("Arial", '', 12)
            
            for i, scenario in enumerate(counterfactuals[:3], 1):
                factor = scenario['factor'].replace('_', ' ').title()
                current_val = scenario['current_value']
                target_val = scenario['target_value']
                reduction = scenario['risk_reduction'] * 100
                
                pdf.cell(0, 8, txt=f"{i}. {factor}:", ln=1)
                pdf.cell(10)
                pdf.cell(0, 8, txt=f"Current: {current_val:.1f}", ln=1)
                pdf.cell(10)
                pdf.cell(0, 8, txt=f"Target: {target_val:.1f}", ln=1)
                pdf.cell(10)
                pdf.cell(0, 8, txt=f"Risk Reduction: {reduction:.1f}%", ln=1)
                pdf.ln(3)
            
            pdf.ln(5)
        
        # 6. INTERVENTION PLAN
        intervention_plan = analysis_results.get('intervention_plan', {})
        if isinstance(intervention_plan, dict) and intervention_plan:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, txt="6. PERSONALIZED INTERVENTION PLAN", ln=1)
            pdf.set_font("Arial", '', 12)
            
            for key, value in intervention_plan.items():
                if value and key != 'urgent_care_needed':  # Skip boolean flag
                    key_title = key.replace('_', ' ').title()
                    
                    if isinstance(value, list):
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 8, txt=f"{key_title}:", ln=1)
                        pdf.set_font("Arial", '', 12)
                        for item in value:
                            if item:  # Skip empty items
                                pdf.cell(10)
                                pdf.cell(0, 8, txt=self._clean_text(f"- {item}"), ln=1)
                        pdf.ln(2)
                    elif value and str(value).strip():
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 8, txt=f"{key_title}:", ln=1)
                        pdf.set_font("Arial", '', 12)
                        pdf.cell(10)
                        pdf.cell(0, 8, txt=self._clean_text(str(value)), ln=1)
                        pdf.ln(2)
            
            pdf.ln(5)
        
        # 7. IMPORTANT NOTES
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="7. IMPORTANT NOTES", ln=1)
        pdf.set_font("Arial", '', 12)
        
        notes = [
            "This report is generated by an AI-powered health assessment system.",
            "It provides informational support based on the data provided.",
            "The accuracy depends on the completeness and accuracy of input data.",
            "Regular health check-ups with healthcare providers are essential.",
            "Monitor any changes in symptoms and seek medical attention when needed."
        ]
        
        for note in notes:
            pdf.cell(0, 8, txt=note, ln=1)
        
        pdf.ln(10)
        
        # DISCLAIMER
        pdf.set_font("Arial", 'I', 10)
        disclaimer = """IMPORTANT DISCLAIMER: This report is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of information in this report."""
        
        # Split disclaimer into multiple lines
        words = disclaimer.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= 100:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        for line in lines:
            pdf.cell(0, 5, txt=line, ln=1)
        
        # Save PDF
        try:
            pdf.output(filename)
            return filename
        except Exception as e:
            # Fallback: try with a different filename
            try:
                safe_filename = "health_report_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pdf"
                pdf.output(safe_filename)
                return safe_filename
            except:
                raise e


# ============================================================================
# 5. INTEGRATED CLINICAL WORKFLOW
# ============================================================================

class ClinicalDecisionSupportSystem:
    """Complete CDS workflow integrating ML and GenAI"""
    
    def __init__(self, openai_api_key):
        self.predictor = DiabetesRiskPredictor()
        self.symptom_analyzer = SymptomAnalyzer(openai_api_key)
        self.insight_generator = ClinicalInsightGenerator(openai_api_key)
        self.pdf_generator = PDFReportGenerator()
        print("Initializing ML model...")
        self.predictor.train()
        print("CDS System ready.\n")
    
    def collect_patient_information(self):
        """Collect comprehensive patient information interactively"""
        print("\n" + "="*70)
        print("👤 PATIENT INFORMATION COLLECTION")
        print("="*70)
        
        patient_info = {}
        
        # Basic Information
        print("\n--- Basic Information ---")
        patient_info['name'] = input("Full Name: ").strip()
        
        while True:
            try:
                age = int(input("Age: ").strip())
                if 0 < age < 120:
                    patient_info['age'] = age
                    break
                else:
                    print("Please enter a valid age (1-119)")
            except ValueError:
                print("Please enter a valid number")
        
        patient_info['gender'] = input("Gender (M/F/Other): ").strip().upper()
        
        # Height and Weight
        print("\n--- Physical Measurements ---")
        while True:
            try:
                height_cm = float(input("Height (cm): ").strip())
                if 50 < height_cm < 250:
                    patient_info['height'] = f"{height_cm} cm"
                    break
                else:
                    print("Please enter a valid height (50-250 cm)")
            except ValueError:
                print("Please enter a valid number")
        
        while True:
            try:
                weight_kg = float(input("Weight (kg): ").strip())
                if 20 < weight_kg < 300:
                    patient_info['weight'] = f"{weight_kg} kg"
                    # Calculate BMI
                    bmi = weight_kg / ((height_cm/100) ** 2)
                    patient_info['bmi'] = round(bmi, 1)
                    print(f"Calculated BMI: {patient_info['bmi']}")
                    break
                else:
                    print("Please enter a valid weight (20-300 kg)")
            except ValueError:
                print("Please enter a valid number")
        
        # Medical History
        print("\n--- Medical History ---")
        patient_info['existing_conditions'] = input("Existing medical conditions (if any): ").strip() or "None"
        
        # Lifestyle Information
        print("\n--- Lifestyle Information ---")
        while True:
            try:
                activity = float(input("Physical activity hours per week: ").strip())
                if 0 <= activity <= 168:  # 24*7 = 168 hours in a week
                    patient_info['physical_activity_hours'] = activity
                    break
                else:
                    print("Please enter a valid number (0-168)")
            except ValueError:
                print("Please enter a valid number")
        
        smoking = input("Do you smoke? (yes/no): ").strip().lower()
        patient_info['smoking'] = 1 if smoking in ['yes', 'y'] else 0
        
        family_history = input("Family history of diabetes? (yes/no): ").strip().lower()
        patient_info['family_history'] = 1 if family_history in ['yes', 'y'] else 0
        
        # Symptoms
        print("\n--- Current Symptoms ---")
        print("Describe any symptoms you're experiencing (or press Enter to skip):")
        print("Example: 'feeling thirsty, frequent urination, fatigue'")
        symptoms = input("Symptoms: ").strip()
        patient_info['symptoms'] = symptoms if symptoms else None
        
        return patient_info
    
    def collect_medical_data(self, patient_info):
        """Collect medical lab data with smart defaults based on patient info"""
        print("\n" + "="*70)
        print("📊 MEDICAL DATA COLLECTION")
        print("="*70)
        print("\nPlease enter your medical data (press Enter for suggested values):")
        
        # Smart defaults based on age and BMI
        age = patient_info.get('age', 50)
        bmi = patient_info.get('bmi', 25)
        
        # Default values based on patient characteristics
        defaults = {
            'blood_pressure_systolic': 130 if age < 50 else 140,
            'blood_pressure_diastolic': 85,
            'hba1c': 5.5 if bmi < 25 else 6.0,
            'fasting_glucose': 95 if bmi < 25 else 105,
            'cholesterol_total': 200 if age < 50 else 220,
            'hdl': 55 if patient_info.get('gender') == 'F' else 45,
            'ldl': 120,
            'triglycerides': 150
        }
        
        medical_data = {
            'age': patient_info['age'],
            'bmi': patient_info['bmi'],
            'family_history': patient_info['family_history'],
            'smoking': patient_info['smoking'],
            'physical_activity_hours': patient_info['physical_activity_hours']
        }
        
        # Collect medical values
        medical_fields = [
            ('blood_pressure_systolic', 'Systolic Blood Pressure (mmHg)'),
            ('blood_pressure_diastolic', 'Diastolic Blood Pressure (mmHg)'),
            ('hba1c', 'HbA1c (%)'),
            ('fasting_glucose', 'Fasting Glucose (mg/dL)'),
            ('cholesterol_total', 'Total Cholesterol (mg/dL)'),
            ('hdl', 'HDL Cholesterol (mg/dL)'),
            ('ldl', 'LDL Cholesterol (mg/dL)'),
            ('triglycerides', 'Triglycerides (mg/dL)')
        ]
        
        for field, label in medical_fields:
            while True:
                default = defaults.get(field, 0)
                value = input(f"{label} (default: {default}): ").strip()
                
                if not value:
                    medical_data[field] = default
                    break
                
                try:
                    val = float(value)
                    # Basic validation
                    if field == 'hba1c' and (val < 3 or val > 15):
                        print("HbA1c should be between 3-15%")
                        continue
                    elif 'blood_pressure' in field:
                        if 'systolic' in field and (val < 70 or val > 250):
                            print("Systolic BP should be between 70-250 mmHg")
                            continue
                        elif 'diastolic' in field and (val < 40 or val > 150):
                            print("Diastolic BP should be between 40-150 mmHg")
                            continue
                    elif 'glucose' in field and (val < 50 or val > 500):
                        print("Glucose should be between 50-500 mg/dL")
                        continue
                    
                    medical_data[field] = val
                    break
                except ValueError:
                    print("Please enter a valid number")
        
        return medical_data
    
    def analyze_patient(self, patient_info, medical_data):
        """Complete risk analysis workflow"""
        
        patient_name = patient_info.get('name', 'Patient')
        
        print(f"\n{'='*70}")
        print(f"🧬 COMPREHENSIVE HEALTH ANALYSIS: {patient_name}")
        print(f"{'='*70}\n")
        
        symptom_analysis_result = None
        structured_symptom_analysis = None
        
        # Step 0: Symptom Analysis (if provided)
        if patient_info.get('symptoms'):
            print("🩺 [AI] Analyzing symptoms for disease patterns...")
            try:
                symptom_analysis = self.symptom_analyzer.analyze_symptoms(
                    patient_info['symptoms'],
                    patient_info.get('age'),
                    patient_info.get('gender'),
                    patient_info.get('existing_conditions')
                )
                symptom_analysis_result = symptom_analysis['detailed_analysis']
                structured_symptom_analysis = symptom_analysis['structured_analysis']
                print("   ✓ Symptom analysis complete\n")
            except Exception as e:
                print(f"   ⚠️ Symptom analysis failed: {e}\n")
                symptom_analysis_result = None
        
        # Step 1: ML Risk Prediction
        print("🤖 [ML] Running diabetes risk prediction model...")
        ml_results = self.predictor.predict_risk(medical_data)
        print(f"   Risk Score: {ml_results['risk_score']:.1%}")
        print(f"   Risk Level: {ml_results['risk_level']}")
        print(f"   Confidence: {ml_results['confidence']:.1%}\n")
        
        # Step 2: Counterfactual Analysis
        print("🔍 [ML] Analyzing intervention scenarios...")
        counterfactuals = self.predictor.get_counterfactual_scenarios(medical_data)
        if counterfactuals:
            print(f"   Found {len(counterfactuals)} modifiable risk factors\n")
        
        # Step 3: GenAI Clinical Report
        print("🧠 [GenAI] Generating comprehensive clinical insights...")
        clinician_report = self.insight_generator.generate_clinician_report(
            medical_data, ml_results, counterfactuals, symptom_analysis_result
        )
        
        # Step 4: GenAI Patient Explanation
        print("💬 [GenAI] Creating patient-friendly explanation...")
        patient_explanation = self.insight_generator.generate_patient_explanation(
            medical_data, ml_results, counterfactuals, symptom_analysis_result
        )
        
        # Step 5: GenAI Intervention Plan
        print("📋 [GenAI] Structuring personalized intervention plan...\n")
        intervention_plan = self.insight_generator.generate_intervention_plan(
            medical_data, ml_results, counterfactuals, symptom_analysis_result
        )
        
        return {
            'patient_info': patient_info,
            'medical_data': medical_data,
            'ml_prediction': ml_results,
            'counterfactuals': counterfactuals,
            'symptom_analysis': symptom_analysis_result,
            'structured_symptom_analysis': structured_symptom_analysis,
            'clinician_report': clinician_report,
            'patient_explanation': patient_explanation,
            'intervention_plan': intervention_plan,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def print_results(self, results):
        """Pretty print all results"""
        
        patient_info = results['patient_info']
        
        print(f"\n{'='*70}")
        print("👤 PATIENT SUMMARY")
        print(f"{'='*70}")
        print(f"Name: {patient_info.get('name', 'Not Provided')}")
        print(f"Age: {patient_info.get('age', 'Not Provided')}")
        print(f"Gender: {patient_info.get('gender', 'Not Provided')}")
        print(f"Height: {patient_info.get('height', 'Not Provided')}")
        print(f"Weight: {patient_info.get('weight', 'Not Provided')}")
        print(f"BMI: {patient_info.get('bmi', 'Not Provided')}")
        
        print(f"\n{'='*70}")
        print("📊 DIABETES RISK ASSESSMENT")
        print(f"{'='*70}")
        print(f"Risk Score: {results['ml_prediction']['risk_score']:.1%}")
        print(f"Risk Level: {results['ml_prediction']['risk_level']}")
        print(f"Confidence: {results['ml_prediction']['confidence']:.1%}\n")
        
        if results.get('symptom_analysis'):
            print(f"{'='*70}")
            print("🩺 SYMPTOM ANALYSIS")
            print(f"{'='*70}")
            print(results['symptom_analysis'])
        
        print(f"\n{'='*70}")
        print("🎯 RISK REDUCTION STRATEGIES")
        print(f"{'='*70}")
        for i, scenario in enumerate(results['counterfactuals'][:3], 1):
            print(f"\n{i}. {scenario['factor'].replace('_', ' ').title()}")
            print(f"   Current: {scenario['current_value']:.1f}")
            print(f"   Target: {scenario['target_value']:.1f}")
            print(f"   Risk Reduction: {scenario['risk_reduction']:.1%}")
        
        print(f"\n{'='*70}")
        print("📋 PERSONALIZED ACTION PLAN")
        print(f"{'='*70}")
        if isinstance(results['intervention_plan'], dict):
            for key, value in results['intervention_plan'].items():
                if value:
                    key_title = key.replace('_', ' ').title()
                    print(f"\n{key_title}:")
                    if isinstance(value, list):
                        for item in value:
                            print(f"  • {item}")
                    else:
                        print(f"  {value}")
    
    def generate_pdf_report(self, results, filename=None):
        """Generate PDF report from analysis results"""
        if not filename:
            patient_name = results['patient_info'].get('name', 'Patient').replace(' ', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"Health_Report_{patient_name}_{timestamp}.pdf"
        
        print(f"\n📄 Generating PDF report: {filename}")
        
        try:
            pdf_filename = self.pdf_generator.create_patient_report(
                results['patient_info'], 
                results,
                filename
            )
            print(f"✅ PDF report saved as: {pdf_filename}")
            return pdf_filename
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            return None


# ============================================================================
# 6. MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    print("="*70)
    print("🩺 COMPREHENSIVE HEALTH ASSESSMENT SYSTEM")
    print("="*70)
    
    # Get API key
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    if not OPENAI_API_KEY:
        print("\n⚠️  OpenAI API Key not found in environment variables.")
        OPENAI_API_KEY = input("Please enter your OpenAI API key: ").strip()
    
    if not OPENAI_API_KEY or len(OPENAI_API_KEY) < 20:
        print("\n❌ Invalid API key format.")
        print("Please get a valid key from: https://platform.openai.com/api-keys")
        print("Then set it as: export OPENAI_API_KEY=your-key-here")
        return
    
    try:
        # Initialize system
        print("\n🚀 Initializing Health Assessment System...")
        cds = ClinicalDecisionSupportSystem(OPENAI_API_KEY)
        
        # Collect patient information
        patient_info = cds.collect_patient_information()
        
        # Collect medical data
        medical_data = cds.collect_medical_data(patient_info)
        
        # Perform analysis
        results = cds.analyze_patient(patient_info, medical_data)
        
        # Display results
        cds.print_results(results)
        
        # Generate PDF report
        print("\n" + "="*70)
        print("📄 REPORT GENERATION")
        print("="*70)
        
        generate_pdf = input("\nGenerate PDF report? (yes/no): ").strip().lower()
        if generate_pdf in ['yes', 'y']:
            pdf_file = cds.generate_pdf_report(results)
            if pdf_file:
                print(f"\n📋 Report includes:")
                print("• Patient Information")
                print("• Diabetes Risk Assessment")
                print("• Symptom Analysis")
                print("• Risk Reduction Strategies")
                print("• Personalized Intervention Plan")
                print("• Important Disclaimer")
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        print("\n⚠️  IMPORTANT DISCLAIMER:")
        print("This system provides informational support only.")
        print("It is NOT a substitute for professional medical advice.")
        print("Always consult with qualified healthcare providers.")
        
        # Save JSON results
        save_json = input("\nSave detailed results as JSON? (yes/no): ").strip().lower()
        if save_json in ['yes', 'y']:
            filename = f"health_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                # Make sure all content is serializable
                serializable = results.copy()
                serializable['clinician_report'] = str(serializable['clinician_report'])
                serializable['patient_explanation'] = str(serializable['patient_explanation'])
                json.dump(serializable, f, indent=2, default=str)
            print(f"✅ Detailed results saved to {filename}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your OpenAI API key and credits")
        print("2. Ensure stable internet connection")
        print("3. Required packages: pip install scikit-learn pandas numpy langchain langchain-openai openai fpdf")


def run_simplified_version():
    """Simplified version without OpenAI"""
    print("\n" + "="*70)
    print("🧪 SIMPLIFIED HEALTH ASSESSMENT")
    print("="*70)
    
    print("\nThis version provides basic diabetes risk assessment without AI analysis.")
    print("For full symptom analysis and detailed reports, please use the full version.")
    
    # Collect basic info
    print("\n--- Basic Information ---")
    name = input("Name: ").strip()
    
    while True:
        try:
            age = int(input("Age: ").strip())
            if 0 < age < 120:
                break
            else:
                print("Please enter valid age (1-119)")
        except:
            print("Please enter a number")
    
    print("\n--- Physical Measurements ---")
    while True:
        try:
            height_cm = float(input("Height (cm): ").strip())
            if 50 < height_cm < 250:
                break
            else:
                print("Please enter valid height (50-250 cm)")
        except:
            print("Please enter a number")
    
    while True:
        try:
            weight_kg = float(input("Weight (kg): ").strip())
            if 20 < weight_kg < 300:
                break
            else:
                print("Please enter valid weight (20-300 kg)")
        except:
            print("Please enter a number")
    
    # Calculate BMI
    bmi = weight_kg / ((height_cm/100) ** 2)
    
    print(f"\n{'='*70}")
    print("📊 BASIC HEALTH SUMMARY")
    print("="*70)
    print(f"Name: {name}")
    print(f"Age: {age}")
    print(f"Height: {height_cm} cm")
    print(f"Weight: {weight_kg} kg")
    print(f"BMI: {bmi:.1f}")
    
    # Simple risk assessment
    print("\n📈 DIABETES RISK INDICATORS:")
    risk_factors = []
    
    if age > 45:
        risk_factors.append("Age over 45")
    
    if bmi >= 25:
        risk_factors.append(f"BMI {bmi:.1f} (Overweight/Obese)")
    elif bmi < 18.5:
        risk_factors.append(f"BMI {bmi:.1f} (Underweight)")
    
    if risk_factors:
        print("⚠️  Risk factors identified:")
        for factor in risk_factors:
            print(f"  • {factor}")
        print("\nConsider discussing these with your healthcare provider.")
    else:
        print("✅ No major risk factors identified based on basic information.")
    
    print("\n🔍 Common diabetes symptoms to watch for:")
    symptoms_list = [
        "Excessive thirst and hunger",
        "Frequent urination",
        "Fatigue and weakness",
        "Blurred vision",
        "Slow-healing sores",
        "Unexplained weight loss"
    ]
    
    for symptom in symptoms_list:
        print(f"• {symptom}")
    
    print("\n📋 For comprehensive analysis including:")
    print("• AI-powered symptom analysis")
    print("• Detailed disease risk assessment")
    print("• Personalized intervention plans")
    print("• PDF report generation")
    print("\nPlease use the full version with an OpenAI API key.")


if __name__ == "__main__":
    print("="*70)
    print("🩺 WELCOME TO HEALTH ASSESSMENT SYSTEM")
    print("="*70)
    
    # Check for API key
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    if OPENAI_API_KEY and len(OPENAI_API_KEY) > 20:
        print("\n✅ Full version available with OpenAI API key.")
        choice = input("Run full version? (yes/no): ").strip().lower()
        if choice in ['yes', 'y']:
            main()
        else:
            run_simplified_version()
    else:
        print("\n⚠️  No valid OpenAI API key found.")
        run_simplified_version()