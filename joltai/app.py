from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import PyPDF2
import spacy
import os

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Function to extract skills dynamically from the resume text using NLP
def extract_skills(text):
    doc = nlp(text)
    extracted_skills = set()
    for token in doc:
        if token.pos_ in ["PROPN", "NOUN"]:
            extracted_skills.add(token.text)
    return list(extracted_skills)

# Function to fetch job listings from LinkedIn (v1 and v2)
def fetch_linkedin_job_listings(skill, version="v1"):
    url = "https://linkedin-data-api.p.rapidapi.com/search-jobs"
    if version == "v2":
        url = "https://linkedin-data-api.p.rapidapi.com/search-jobs-v2"

    querystring = {
        "keywords": skill,
        "locationId": "105214831",  # Location ID for India
        "datePosted": "anyTime",
        "sort": "mostRelevant"
    }

    headers = {
        "x-rapidapi-key": "337669e716msh2c219abd9c6130dp1cab35jsne06e86b103ac",  # Replace with your RapidAPI key
        "x-rapidapi-host": "linkedin-data-api.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code != 200:
        return []
    return response.json()
@app.route('/extract_skills', methods=['POST'])
def extract_skills_from_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded."}), 400

    resume = request.files["resume"]
    temp_path = f"temp_{resume.filename}"
    resume.save(temp_path)
    
    try:
        resume_text = extract_text_from_pdf(temp_path)
        skills = extract_skills(resume_text)
    finally:
        os.remove(temp_path)  # Ensure file is deleted even if an error occurs

    return jsonify({
        "extracted_skills": skills,
        "message": "Skills extracted successfully."
    })

# Function to fetch job listings from CareerJet
def fetch_careerjet_job_listings(skill, location):
    base_url = "https://www.careerjet.co.in/jobs"
    params = {'s': skill.replace(' ', '+'), 'l': location.replace(' ', '+')}
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    job_cards = soup.find_all('article', class_='job clicky')

    job_listings = []
    for job in job_cards:
        title_tag = job.find('h2').find('a')
        company_tag = job.find('p', class_='company')
        location_tag = job.find('ul', class_='location')
        description_tag = job.find('div', class_='desc')
        date_posted_tag = job.find('span', class_='badge badge-r badge-s badge-icon')

        if title_tag:
            job_listings.append({
                'title': title_tag.text.strip(),
                'company': company_tag.text.strip() if company_tag else 'N/A',
                'location': location_tag.text.strip() if location_tag else 'N/A',
                'description': description_tag.text.strip() if description_tag else 'N/A',
                'date_posted': date_posted_tag.text.strip() if date_posted_tag else 'N/A',
                'url': f"https://www.careerjet.co.in{title_tag['href']}"
            })
    return job_listings

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search_jobs():
    if request.method == 'POST':
        search_type = request.form.get("type")

        if search_type == "keyword":
            skill = request.form.get('skill')
            location = request.form.get('location')
            linkedin_jobs_v1 = fetch_linkedin_job_listings(skill, version="v1")
            linkedin_jobs_v2 = fetch_linkedin_job_listings(skill, version="v2")
            careerjet_jobs = fetch_careerjet_job_listings(skill, location)
            return jsonify({
                "results": linkedin_jobs_v1.get('data', []) + linkedin_jobs_v2.get('data', []) + careerjet_jobs
            })

        elif search_type == "resume":
            if "resume" not in request.files:
                return jsonify({"error": "No resume file uploaded."}), 400
            resume = request.files["resume"]
            temp_path = f"temp_{resume.filename}"
            resume.save(temp_path)
            resume_text = extract_text_from_pdf(temp_path)
            os.remove(temp_path)

            skills = extract_skills(resume_text)
            return jsonify({
                "extracted_skills": skills,
                "message": "Skills extracted successfully."
            })

        return jsonify({"error": "Invalid search type."}), 400

    return render_template('search.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
