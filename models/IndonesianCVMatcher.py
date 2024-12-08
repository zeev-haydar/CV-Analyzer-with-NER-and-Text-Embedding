# IndonesianCVMatcher.py

import torch
import re
from transformers.models.auto import AutoTokenizer, AutoModel
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List
from utils.timer import timer_decorator
import numpy as np
import os

def clean_punctuation_except_period_and_colon(text):
  cleaned_text = re.sub(r'[^a-zA-Z0-9\s\.\,\:]', '', text)
  return cleaned_text

class IndonesianCVMatcher():
    def __init__(self,  job_req_dict=None, model_name='indolem/indobert-base-uncased', ner_model_name='cahya/bert-base-indonesian-NER', stop_words = None):
        super(IndonesianCVMatcher, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.job_req_dict = job_req_dict
        self.stop_words = stop_words
        
        model_weights_dir = os.path.join(os.getcwd(), 'model_weights')

        # Ensure the directory exists
        os.makedirs(model_weights_dir, exist_ok=True)

        # Define paths for the main model and NER model
        main_model_dir = os.path.join(model_weights_dir, 'indobert')
        ner_model_dir = os.path.join(model_weights_dir, 'indonesian-ner')

        # Load or download and save the main model
        if not os.path.exists(main_model_dir):
            print(f"Downloading and saving {model_name} to {main_model_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer.save_pretrained(main_model_dir)
            self.model.save_pretrained(main_model_dir)
        else:
            print(f"Loading {model_name} from {main_model_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(main_model_dir)
            self.model = AutoModel.from_pretrained(main_model_dir)

        # Load or download and save the NER model
        if not os.path.exists(ner_model_dir):
            print(f"Downloading and saving {ner_model_name} to {ner_model_dir}...")
            self.ner_pipeline = pipeline('ner', model=ner_model_name, tokenizer=ner_model_name, grouped_entities=True)
            AutoTokenizer.from_pretrained(ner_model_name).save_pretrained(ner_model_dir)
            AutoModel.from_pretrained(ner_model_name).save_pretrained(ner_model_dir)
        else:
            print(f"Loading {ner_model_name} from {ner_model_dir}...")
            ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_dir)
            ner_model = AutoModel.from_pretrained(ner_model_dir)
            self.ner_pipeline = pipeline('ner', model=ner_model_name, tokenizer=ner_model_name, grouped_entities=True)

        # Move model to the appropriate device
        self.model.to(self.device)

    def preprocess_text(self, text) -> str:
      # Convert text to lowercase
      text = text.lower()
      # Remove punctuation but not number
      text = clean_punctuation_except_period_and_colon(text)

      text = re.sub(r'\s+', ' ', text)


      # Remove extra whitespace
      text = ' '.join(text.split())

      # lemmatization


      # Remove stopwords
      if self.stop_words is not None:
        words = text.split()
        words = [word for word in words if word not in self.stop_words]

        return ' '.join(words)

      return text

    def get_bert_embeddings(self, text: str) -> np.ndarray:
      # Tokenize and prepare input
      inputs = self.tokenizer(
          text,
          max_length=512,
          padding=True,
          truncation=True,
          return_tensors='pt'
      ).to(self.device)

      # Get embeddings
      with torch.no_grad():
          outputs = self.model(**inputs)
          embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

      return embeddings[0]

    def extract_skills(self, text: str) -> List[str]:
      # Common Indonesian technical terms and patterns
      # text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
      text = clean_punctuation_except_period_and_colon(text)
      text = re.sub(r'\s+', ' ', text)
      text = ' '.join(text.split())
      text = text.lower()

      skill_patterns = [
        r'(?i)(?:ahli|mampu|menguasai|berpengalaman dalam)\s+([\w\s]+)',
        r'(?i)(?:keahlian|kemampuan|keterampilan)[\s:]+([^.]+)',
        r'(?i)(?:sertifikasi|sertifikat)\s+([\w\s]+)',
        r'(?i)pengalaman\s+(?:dengan|dalam)\s+([\w\s]+)',
      ]

      skills = []
      for pattern in skill_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
          skill = match.group(1).strip().lower()
          if len(skill) > 0:
            skills.append(skill)
      # print(f"Skills: {skills}")
      return list(set(skills))

    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        text = clean_punctuation_except_period_and_colon(text)
        max_length = 512  # Max length for BERT-based models
        tokens = self.ner_pipeline.tokenizer(text, max_length=510, truncation=True, padding=True)
        decoded = self.ner_pipeline.tokenizer.decode(tokens['input_ids'])

        # Updated line to pass the decoded text using the 'text' key:
        entities = self.ner_pipeline(decoded)
        entity_dict = {}
        for entity in entities:
            label = entity['entity_group']
            entity_text = entity['word'].strip()
            if label not in entity_dict:
                entity_dict[label] = []
            entity_dict[label].append(entity_text)
        # Ensure no duplicate entities
        for key in entity_dict:
            entity_dict[key] = list(set(entity_dict[key]))
        # print("Entities:", entity_dict)
        return entity_dict

    def calculate_entity_match_score(self, cv_entities, job_entities):
        entity_match_score = 0
        matching_details = {}

        # Find matching labels (keys in both dictionaries)
        matching_labels = set(cv_entities.keys()).intersection(job_entities.keys())

        for label in matching_labels:
            # Find common and total entities for the current label
            common_entities = set(cv_entities[label]).intersection(job_entities[label])
            total_entities = set(cv_entities[label]).union(job_entities[label])

            if total_entities:  # Avoid division by zero
                match_score = len(common_entities) / len(total_entities)
                entity_match_score += match_score
                matching_details[label] = list(common_entities)  # Store matched entities for each label

        # Average the score across all matching labels
        if matching_labels:
            entity_match_score /= len(matching_labels)

        return entity_match_score, matching_details


    def extract_ipk(self, text: str) -> float:
        text = re.sub(r'[^a-zA-Z0-9\s\.]', ' ', text)
        angka = r'([0-4](?:\.\d{1,2})?)(?:/\s*[0-4](?:\.\d{1,3})?)?'
        # matches = re.findall(angka, text.lower())
        # for match in matches:
        #     try:
        #         # print(match[0])
        #         ipk = float(match[0])
        #     except ValueError:
        #         pass
        # pattern = r'(?:ipk(?:\s*[:\s]*minimal|\s*:)?\s*|)'+angka
        pattern = r'(?i)\bipk\b.*?([0-4](?:\.\d{1,3})?)'
        matches = re.findall(pattern, text)
        if matches:
            # print(matches)
            return max(float(match) for match in matches)
        return 0.0

    @timer_decorator
    def calculate_similarity_scores(self, cv_text: str, job_req: str, mode:str = 'embedding') -> Dict[str, float]:
      cv_processed = self.preprocess_text(cv_text)
      job_processed = self.preprocess_text(job_req)

      # cv_text = clean_punctuation_except_period_and_colon(cv_text)
      # job_req = clean_punctuation_except_period_and_colon(job_req)

      cv_embedding = self.get_bert_embeddings(cv_text)
      job_embedding = self.get_bert_embeddings(job_req)

      similarity_score = cosine_similarity([cv_embedding], [job_embedding])[0][0]

      cv_skills = self.extract_skills(cv_text)
      job_skills = self.extract_skills(job_req)

       # Extract named entities
      cv_entities = self.extract_named_entities(cv_text)
      job_entities = self.extract_named_entities(job_req)

      # Calculate entity match score (e.g., job titles, organizations)
      entity_match_score, _ = self.calculate_entity_match_score(cv_entities, job_entities)

      if cv_skills and job_skills:
          # Combine skills into a single string for embedding calculation
          if mode == 'embedding':
            cv_skills_text = ' '.join(cv_skills)
            job_skills_text = ' '.join(job_skills)

            cv_skills_embedding = self.get_bert_embeddings(cv_skills_text)
            job_skills_embedding = self.get_bert_embeddings(job_skills_text)
            skills_similarity = cosine_similarity([cv_skills_embedding], [job_skills_embedding])[0][0]
            skills_similarity_factor = 1000
            keyword_overlap_factor = 100
          elif mode == 'tf-idf':
            vectorizer = TfidfVectorizer()
            skill_corpus = [' '.join(cv_skills), ' '.join(job_skills)]
            tfidf_matrix = vectorizer.fit_transform(skill_corpus)
            skills_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0]
            skills_similarity_factor = 10
            keyword_overlap_factor = 1
          else:
            raise ValueError("Invalid mode. Choose 'embedding' or 'tf-idf'.")
      else:
          skills_similarity = 0

      # Calculate keyword overlap
      cv_words = set(cv_processed.split())
      job_words = set(job_processed.split())
      keyword_overlap = len(cv_words.intersection(job_words)) / len(job_words)

      # Calculate the overall score
      # overall_score = (similarity_score + keyword_overlap + skills_similarity**3 + entity_match_score**(0.1)+3) / (4+3)
      compensation_value = 0
      overall_score = (similarity_score*0.04 + (skills_similarity)*0.75 + keyword_overlap*0.2 + entity_match_score*0.05 + compensation_value)/(1+compensation_value)

      return {
            'semantic_similarity': float(similarity_score),
            'skills_similarity': float(skills_similarity),
            'keyword_overlap': float(keyword_overlap),
            'entity_match_score': float(entity_match_score),
            'overall_score': float(overall_score)

      }

    def extract_years_of_experience(self, text: str) -> int:
        pattern = r'(\d+)\s+(tahun|years)'
        matches = re.findall(pattern, text.lower())
        if matches:
            return max(int(match[0]) for match in matches)
        return 0

    def print_match_requirements(self, cv_text: str, job_req: str, file_handle=None):
        cv_processed = self.preprocess_text(cv_text)
        job_processed = self.preprocess_text(job_req)

        # Extract years of experience
        cv_experience = self.extract_years_of_experience(cv_text)
        job_experience = self.extract_years_of_experience(job_req)

        # Extract skills
        cv_skills = self.extract_skills(cv_text)
        job_skills = self.extract_skills(job_req)

        # Extract named entities
        cv_entities = self.extract_named_entities(cv_text)
        job_entities = self.extract_named_entities(job_req)

        # Calculate entity match score and get matching details
        entity_match_score, matching_details = self.calculate_entity_match_score(cv_entities, job_entities)

        cv_ipk = self.extract_ipk(cv_text)
        job_ipk = self.extract_ipk(job_req)

        # Find overlapping skills
        matched_skills = set(cv_skills).intersection(set(job_skills))

        output = []

        output.append("\nMatched Requirements:")
        if cv_experience >= job_experience:
            output.append(f"- Required experience met: {cv_experience} years (required: {job_experience} years)")
        else:
            output.append(f"- Experience mismatch: {cv_experience} years (required: {job_experience} years)")

        if matched_skills:
            output.append(f"- Overlapping skills: {', '.join(matched_skills)}")
        else:
            output.append("- No overlapping skills found.")

        if job_ipk > 0:
            if cv_ipk >= job_ipk:
                output.append(f"- Required IPK met: {cv_ipk} (required: {job_ipk})")
            else:
                output.append(f"- IPK mismatch: {cv_ipk} (required: {job_ipk})")
        else:
            output.append("- No specific IPK requirement found in the job description.")

        output.append("\nEntity Matching Details:")
        if matching_details:
            for label, entities in matching_details.items():
                if entities:
                    output.append(f"- {label}: {', '.join(entities)}")
        else:
            output.append("- No matching entities found.")

        output.append("\nJob Requirement Skills:")
        output.append(f"- {', '.join(job_skills)}")

        output.append("\nCV Skills:")
        output.append(f"- {', '.join(cv_skills)}")
        
        key_result = {
            "entity_match": matching_details,
            "matched_skills":matched_skills,
            "cv_ipk": cv_ipk,
            "job_ipk": job_ipk,
            "cv_entities": cv_entities,
            "job_entities": job_entities,
            "cv_skills": cv_skills,
            "job_skills": job_skills,
            "cv_experience": cv_experience,
            "job_experience": job_experience,
            'analysis_score':self.calculate_similarity_scores(cv_text, job_req)
        }

        result = "\n".join(output)

        if file_handle:
            file_handle.write(result + "\n")
        else:
            # print(result)
            return key_result, result

    def rank_candidates(self, cvs: List[Dict], job_req: str) -> List[Dict]:
        rankings = []

        for cv in cvs:
            scores = self.calculate_similarity_scores(cv['text'], job_req)
            rankings.append({
                'candidate_id': cv.get('id', ''),
                'candidate_name': cv.get('name', ''),
                'scores': scores,
                
            })

        # Sort by overall score
        rankings.sort(key=lambda x: x['scores']['overall_score'], reverse=True)
        return rankings

