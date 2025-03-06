import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import io
import PyPDF2
import docx
import base64
import streamlit as st

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file."""
    doc = docx.Document(docx_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def extract_text_from_uploaded_file(uploaded_file):
    """Extract text from uploaded file based on file type."""
    if uploaded_file is None:
        return ""
        
    # Get file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        # Process based on file type
        if file_extension == 'pdf':
            return extract_text_from_pdf(uploaded_file)
        elif file_extension in ['docx', 'doc']:
            return extract_text_from_docx(uploaded_file)
        elif file_extension in ['txt', 'rtf']:
            # For text files, read directly
            return uploaded_file.getvalue().decode('utf-8')
        else:
            return "Unsupported file format. Please upload PDF, DOCX, or TXT files."
    except Exception as e:
        return f"Error processing file: {str(e)}"

class ATSScorer:
    """
    An ATS scoring system using unsupervised machine learning to evaluate resumes.
    This system uses techniques like TF-IDF, LSA, clustering, and similarity scoring.
    """
    
    def __init__(self, job_description, resumes, n_clusters=5):
        """
        Initialize the ATS scoring system.
        
        Parameters:
        -----------
        job_description : str
            The text of the job description.
        resumes : list
            List of resume texts to be evaluated.
        n_clusters : int
            Number of clusters for KMeans.
        """
        self.job_description = job_description
        self.resumes = resumes
        self.n_clusters = min(n_clusters, len(resumes)) if len(resumes) > 0 else 1
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.resume_scores = {}
        
    def preprocess_text(self, text):
        """Clean and preprocess text for analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_features(self):
        """Convert text to numerical features using TF-IDF."""
        # Preprocess all documents
        preprocessed_jd = self.preprocess_text(self.job_description)
        preprocessed_resumes = [self.preprocess_text(resume) for resume in self.resumes]
        
        # Combine all texts for vectorization
        all_docs = [preprocessed_jd] + preprocessed_resumes
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(all_docs)
        
        # Apply dimensionality reduction
        n_components = min(100, self.tfidf_matrix.shape[1]-1, self.tfidf_matrix.shape[0]-1)
        self.svd = TruncatedSVD(n_components=max(2, n_components))
        self.lsa_matrix = self.svd.fit_transform(self.tfidf_matrix)
        
        # Extract JD vector and resume vectors
        self.jd_vector = self.lsa_matrix[0].reshape(1, -1)
        self.resume_vectors = self.lsa_matrix[1:]
        
    def cluster_resumes(self):
        """Cluster resumes to identify patterns."""
        if len(self.resumes) < 2:
            # Not enough resumes to cluster
            self.cluster_labels = np.array([0])
            self.best_cluster = 0
            return
            
        # Apply KMeans clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.cluster_labels = self.kmeans.fit_predict(self.resume_vectors)
        
        # Calculate cluster centroids and find closest cluster to JD
        distances = []
        for centroid in self.kmeans.cluster_centers_:
            dist = cosine_similarity(self.jd_vector, centroid.reshape(1, -1))[0][0]
            distances.append(dist)
        
        self.best_cluster = np.argmax(distances)
        
    def calculate_similarity_scores(self):
        """Calculate similarity scores between JD and resumes."""
        # Direct cosine similarity between JD and each resume
        similarities = cosine_similarity(self.jd_vector, self.resume_vectors)
        
        # Calculate term overlap scores
        jd_terms = set(self.vectorizer.get_feature_names_out())
        resume_term_scores = []
        
        for i, resume in enumerate(self.resumes):
            preprocessed = self.preprocess_text(resume)
            resume_terms = set(preprocessed.split())
            overlap = len(resume_terms.intersection(jd_terms)) / len(jd_terms) if len(jd_terms) > 0 else 0
            resume_term_scores.append(overlap)
        
        # Combine scores
        for i in range(len(self.resumes)):
            # Base similarity score
            base_score = similarities[0][i] * 70
            
            # Term overlap bonus (up to 20 points)
            term_bonus = resume_term_scores[i] * 20
            
            # Cluster alignment bonus (up to 10 points)
            if len(self.resumes) > 1:
                cluster_bonus = 10 if self.cluster_labels[i] == self.best_cluster else 0
            else:
                cluster_bonus = 5  # Only one resume, give half the bonus
            
            # Combined score
            total_score = min(100, base_score + term_bonus + cluster_bonus)
            self.resume_scores[i] = round(total_score, 2)
    
    def evaluate(self):
        """Run the entire evaluation pipeline."""
        if not self.resumes or all(not resume for resume in self.resumes):
            return {}
            
        print("Extracting features...")
        self.extract_features()
        
        print("Clustering resumes...")
        self.cluster_resumes()
        
        print("Calculating scores...")
        self.calculate_similarity_scores()
        
        return self.resume_scores
    
    def get_keyword_importance(self):
        """Get top keywords and their importance."""
        if not hasattr(self, 'vectorizer') or not hasattr(self, 'svd'):
            return {}
            
        feature_names = self.vectorizer.get_feature_names_out()
        components = self.svd.components_
        keywords = {}
        
        # Get top keywords from LSA components
        for i in range(min(5, len(components))):
            top_indices = np.argsort(np.abs(components[i]))[-10:]
            for idx in top_indices:
                if feature_names[idx] not in keywords:
                    keywords[feature_names[idx]] = abs(components[i][idx])
                else:
                    keywords[feature_names[idx]] += abs(components[i][idx])
        
        # Normalize and return top 20
        total = sum(keywords.values())
        if total > 0:
            for key in keywords:
                keywords[key] = keywords[key] / total
            
        return dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:20])
    
    def get_detailed_analysis(self, resume_idx):
        """Get detailed analysis for a specific resume."""
        if resume_idx >= len(self.resumes):
            return "Invalid resume index."
            
        # Get preprocessed text
        preprocessed = self.preprocess_text(self.resumes[resume_idx])
        
        # Calculate keyword presence
        important_keywords = self.get_keyword_importance()
        found_keywords = {}
        
        for keyword in important_keywords:
            if keyword in preprocessed:
                found_keywords[keyword] = important_keywords[keyword]
        
        # Missing important keywords
        missing_keywords = {k: v for k, v in important_keywords.items() if k not in found_keywords}
        
        # Cluster information
        if len(self.resumes) > 1:
            cluster_id = self.cluster_labels[resume_idx]
            cluster_match = "Yes" if cluster_id == self.best_cluster else "No"
            
            # Resume similarity to other resumes in same cluster
            cluster_indices = [i for i, label in enumerate(self.cluster_labels) if label == cluster_id]
            if len(cluster_indices) > 1:
                cluster_vectors = self.resume_vectors[cluster_indices]
                this_vector = self.resume_vectors[resume_idx].reshape(1, -1)
                similarities = cosine_similarity(this_vector, cluster_vectors)[0]
                avg_similarity = np.mean([sim for i, sim in enumerate(similarities) if cluster_indices[i] != resume_idx])
            else:
                avg_similarity = 1.0  # Only resume in cluster
        else:
            cluster_id = 0
            cluster_match = "N/A (Single Resume)"
            avg_similarity = 1.0
            
        return {
            "score": self.resume_scores[resume_idx],
            "found_keywords": found_keywords,
            "missing_keywords": list(missing_keywords.keys())[:5],
            "cluster_id": cluster_id,
            "best_cluster_match": cluster_match,
            "similarity_to_jd": cosine_similarity(self.jd_vector, self.resume_vectors[resume_idx].reshape(1, -1))[0][0],
            "similarity_to_cluster": avg_similarity
        }

# Streamlit UI for the ATS Scoring System
def create_ats_app():
    st.title("ATS Resume Scoring System")
    st.write("Upload your resume and job description to get an ATS score based on unsupervised machine learning.")
    
    # Set up tabs for different functionalities
    tab1, tab2 = st.tabs(["Individual Resume Analysis", "Compare Multiple Resumes"])
    
    with tab1:
        st.header("Upload Your Resume")
        uploaded_resume = st.file_uploader("Choose a resume file", type=["pdf", "docx", "txt"], key="single_resume")
        
        st.header("Job Description")
        job_description = st.text_area("Paste the job description here", height=200)
        
        if st.button("Analyze Resume", key="analyze_single"):
            if uploaded_resume is not None and job_description:
                with st.spinner("Analyzing your resume..."):
                    # Extract text from uploaded resume
                    resume_text = extract_text_from_uploaded_file(uploaded_resume)
                    
                    if not isinstance(resume_text, str) or resume_text.startswith("Error") or resume_text.startswith("Unsupported"):
                        st.error(resume_text)
                    else:
                        # Create ATS scorer with a single resume
                        ats = ATSScorer(job_description, [resume_text])
                        scores = ats.evaluate()
                        
                        if scores:
                            # Display score with gauge
                            score = scores[0]
                            st.subheader(f"Your ATS Score: {score}/100")
                            
                            # Create a progress bar for the score
                            st.progress(score/100)
                            
                            # Score interpretation
                            if score >= 80:
                                st.success("Excellent match! Your resume is well-aligned with this job description.")
                            elif score >= 60:
                                st.info("Good match. Consider enhancing your resume with some missing keywords.")
                            else:
                                st.warning("Your resume may need significant improvements to pass ATS filters.")
                            
                            # Get detailed analysis
                            analysis = ats.get_detailed_analysis(0)
                            
                            # Display keyword analysis
                            st.subheader("Keyword Analysis")
                            
                            # Found keywords
                            st.write("**Found Keywords:**")
                            found_keywords = analysis["found_keywords"]
                            if found_keywords:
                                # Create a dataframe for better visualization
                                keywords_df = pd.DataFrame({
                                    'Keyword': list(found_keywords.keys()),
                                    'Importance': list(found_keywords.values())
                                })
                                keywords_df = keywords_df.sort_values('Importance', ascending=False)
                                
                                # Display as a horizontal bar chart
                                st.bar_chart(keywords_df.set_index('Keyword'))
                            else:
                                st.write("No significant keywords found.")
                                
                            # Missing keywords
                            st.write("**Missing Important Keywords:**")
                            missing_keywords = analysis["missing_keywords"]
                            if missing_keywords:
                                for keyword in missing_keywords:
                                    st.write(f"- {keyword}")
                            else:
                                st.write("No significant keywords missing.")
                            
                            # Recommendations
                            st.subheader("Recommendations")
                            recommendations = []
                            
                            if score < 80:
                                recommendations.append("Consider adding the missing keywords to your resume.")
                            if score < 70:
                                recommendations.append("Restructure your resume to better match the job description's terminology.")
                            if score < 60:
                                recommendations.append("Review the job description carefully and tailor your experience to highlight relevant skills.")
                            
                            if recommendations:
                                for i, rec in enumerate(recommendations, 1):
                                    st.write(f"{i}. {rec}")
                            else:
                                st.write("Your resume is well-optimized for this job description.")
                        else:
                            st.error("Failed to analyze the resume. Please check your inputs and try again.")
            else:
                st.error("Please upload a resume and provide a job description.")
    
    with tab2:
        st.header("Upload Multiple Resumes")
        uploaded_resumes = st.file_uploader("Choose resume files", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="multi_resume")
        
        st.header("Job Description")
        job_description_multi = st.text_area("Paste the job description here", height=200, key="jd_multi")
        
        if st.button("Compare Resumes", key="analyze_multi"):
            if uploaded_resumes and job_description_multi:
                with st.spinner("Analyzing resumes..."):
                    # Extract text from each uploaded resume
                    resume_texts = []
                    resume_names = []
                    
                    for uploaded_file in uploaded_resumes:
                        resume_text = extract_text_from_uploaded_file(uploaded_file)
                        if not isinstance(resume_text, str) or resume_text.startswith("Error") or resume_text.startswith("Unsupported"):
                            st.error(f"{uploaded_file.name}: {resume_text}")
                        else:
                            resume_texts.append(resume_text)
                            resume_names.append(uploaded_file.name)
                    
                    if resume_texts:
                        # Create ATS scorer with multiple resumes
                        ats = ATSScorer(job_description_multi, resume_texts)
                        scores = ats.evaluate()
                        
                        if scores:
                            # Create a dataframe for comparison
                            results = []
                            for idx, score in scores.items():
                                analysis = ats.get_detailed_analysis(idx)
                                results.append({
                                    'Resume': resume_names[idx],
                                    'Score': score,
                                    'Keywords Found': len(analysis["found_keywords"]),
                                    'Keywords Missing': len(analysis["missing_keywords"]),
                                    'Best Cluster Match': analysis["best_cluster_match"],
                                    'Similarity to JD': round(analysis["similarity_to_jd"] * 100, 2)
                                })
                            
                            results_df = pd.DataFrame(results)
                            results_df = results_df.sort_values('Score', ascending=False)
                            
                            # Display comparison table
                            st.subheader("Resume Comparison")
                            st.dataframe(results_df)
                            
                            # Visualize scores
                            st.subheader("Score Comparison")
                            st.bar_chart(results_df.set_index('Resume')['Score'])
                            
                            # Detailed analysis of top resume
                            if len(results_df) > 0:
                                top_resume_idx = results_df.index[0]
                                st.subheader(f"Detailed Analysis: {resume_names[top_resume_idx]}")
                                
                                analysis = ats.get_detailed_analysis(top_resume_idx)
                                
                                # Found keywords
                                st.write("**Top Keywords Found:**")
                                found_keywords = analysis["found_keywords"]
                                if found_keywords:
                                    for keyword, importance in list(found_keywords.items())[:10]:
                                        st.write(f"- {keyword} (Importance: {importance:.3f})")
                                
                                # Missing keywords
                                st.write("**Top Missing Keywords:**")
                                missing_keywords = analysis["missing_keywords"]
                                if missing_keywords:
                                    for keyword in missing_keywords:
                                        st.write(f"- {keyword}")
                        else:
                            st.error("Failed to analyze the resumes. Please check your inputs and try again.")
                    else:
                        st.error("No valid resumes were uploaded. Please check the file formats.")
            else:
                st.error("Please upload at least one resume and provide a job description.")

if __name__ == "__main__":
    create_ats_app()