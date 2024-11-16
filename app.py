from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Function to fetch LinkedIn job listings
def fetch_linkedin_job_listings(skill, location, version="v1"):
    if version == "v1":
        url = "https://linkedin-data-api.p.rapidapi.com/search-jobs"
    else:
        url = "https://linkedin-data-api.p.rapidapi.com/search-jobs-v2"

    querystring = {
        "keywords": skill,
        "locationId": "92000000",
        "datePosted": "anyTime",
        "sort": "mostRelevant"
    }

    headers = {
        "x-rapidapi-key": "81cbc86b63mshdf6186b29d0004ep150de1jsn454a4e227c03",  # Replace with your RapidAPI key
        "x-rapidapi-host": "linkedin-data-api.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code != 200:
        return {"error": f"Failed to fetch LinkedIn job listings (v{version}) for {skill}. Status Code: {response.status_code}"}

    return response.json()

# Function to fetch CareerJet job listings
def fetch_careerjet_job_listings(skill, location):
    base_url = "https://www.careerjet.co.in/jobs"
    params = {'s': skill.replace(' ', '+'), 'l': location.replace(' ', '+')}
    response = requests.get(base_url, params=params)

    if response.status_code != 200:
        return {"error": "Failed to retrieve CareerJet listings."}

    soup = BeautifulSoup(response.text, 'html.parser')
    job_cards = soup.find_all('article', class_='job clicky')

    job_listings = []
    for job in job_cards:
        title_tag = job.find('h2').find('a')
        company_tag = job.find('p', class_='company')
        location_tag = job.find('ul', class_='location')
        description_tag = job.find('div', class_='desc')
        date_posted_tag = job.find('span', class_='badge badge-r badge-s badge-icon')

        company = company_tag.find('a') if company_tag else None
        location = location_tag.find('li') if location_tag else None
        description = description_tag if description_tag else None
        date_posted = date_posted_tag if date_posted_tag else None

        if title_tag:
            job_listings.append({
                'title': title_tag.text.strip(),
                'company': company.text.strip() if company else 'N/A',
                'location': location.text.strip() if location else 'N/A',
                'description': description.text.strip() if description else 'N/A',
                'date_posted': date_posted.text.strip() if date_posted else 'N/A',
                'url': f"https://www.careerjet.co.in{title_tag['href']}" if title_tag else ''
            })
    return job_listings

@app.route('/')
def index():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search_jobs():
    skill = request.form.get('skill')
    location = request.form.get('location')

    linkedin_jobs_v1 = fetch_linkedin_job_listings(skill, location, version="v1")
    linkedin_jobs_v2 = fetch_linkedin_job_listings(skill, location, version="v2")
    careerjet_jobs = fetch_careerjet_job_listings(skill, location)

    return jsonify({
        "linkedin_v1": linkedin_jobs_v1,
        "linkedin_v2": linkedin_jobs_v2,
        "careerjet": careerjet_jobs
    })

if __name__ == '__main__':
    app.run(debug=True)
