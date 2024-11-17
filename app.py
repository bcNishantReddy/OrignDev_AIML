import requests
from bs4 import BeautifulSoup
import PyPDF2
import spacy
import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import os
import google.generativeai as genai
from werkzeug.utils import secure_filename
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import torch
import mediapipe as mp
import cv2
import numpy as np
import traceback
import tempfile
import shutil
import gc
from moviepy.editor import VideoFileClip

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your secret key
nlp = spacy.load("en_core_web_sm")

# Configure Gemini API
genai.configure(api_key="AIzaSyC93_yezXaB6q0dKVGVQZRP9newUTAiuqo")  # Replace with your Gemini API key
model = genai.GenerativeModel(model_name='gemini-1.5-flash')

# Video upload directory
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Whisper Model
whisper_model = WhisperForConditionalGeneration.from_pretrained("fine_tuned_whisper")
whisper_processor = WhisperProcessor.from_pretrained("fine_tuned_whisper")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class AnalysisMetrics:
    def __init__(self):
        self.face_metrics = []
        self.pose_metrics = []
        self.hand_metrics = []
        self.eye_contact_frames = 0
        self.total_frames = 0
        self.posture_issues = 0
        self.gesture_counts = 0
        self.transcript = []
        self.video_duration = 0

class VideoProcessor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.holistic.close()

def extract_audio(video_path, audio_path):
    """Extract audio from video using MoviePy."""
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path, fps=16000, nbytes=2, codec='pcm_s16le')
        audio_clip.close()
        video_clip.close()
    except Exception as e:
        print(f"Error in extract_audio: {e}")
        raise

def get_video_duration(video_path):
    """Get video duration using MoviePy."""
    try:
        video_clip = VideoFileClip(video_path)
        duration = video_clip.duration
        video_clip.close()
        return duration
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return 0

def process_video_frames(video_path, analysis, processor):
    """Process video frames using OpenCV and MediaPipe."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = processor.holistic.process(frame_rgb)
            analysis.total_frames += 1

            # Analyze face landmarks for eye contact
            if results.face_landmarks:
                landmarks = results.face_landmarks.landmark
                if len(landmarks) > 263:
                    nose = landmarks[1]
                    left_eye = landmarks[33]
                    right_eye = landmarks[263]
                    eye_avg_x = (left_eye.x + right_eye.x) / 2
                    if abs(nose.x - eye_avg_x) < 0.05:
                        analysis.eye_contact_frames += 1

            # Analyze pose landmarks for posture
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                if len(landmarks) > 12:
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
                    if shoulder_diff > 0.05:
                        analysis.posture_issues += 1

            # Count gestures
            if results.left_hand_landmarks or results.right_hand_landmarks:
                analysis.gesture_counts += 1

    except Exception as e:
        print(f"Error in process_video_frames: {e}")
        raise
    finally:
        if 'cap' in locals():
            cap.release()

def analyze_video_and_audio(video_path):
    """Analyze the video and extract audio for transcription."""
    print("Starting video analysis...")
    analysis = AnalysisMetrics()
    temp_dir = None

    try:
        # Use the uploaded video directly
        mp4_path = video_path
        temp_dir = tempfile.mkdtemp()
        wav_path = os.path.join(temp_dir, "audio.wav")

        # Extract audio from the video using MoviePy
        print("Extracting audio...")
        extract_audio(mp4_path, wav_path)

        # Process video frames for analysis
        print("Processing video frames...")
        with VideoProcessor() as processor:
            process_video_frames(mp4_path, analysis, processor)

        # Get video duration
        analysis.video_duration = get_video_duration(mp4_path)

        # Transcribe audio using Whisper
        print("Transcribing audio...")
        transcript = transcribe_audio(wav_path)

        return transcript, analysis

    except Exception as e:
        print(f"Error in analyze_video_and_audio: {e}")
        traceback.print_exc()
        raise
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            try:
                gc.collect()
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Error cleaning up temporary files: {e}")

def transcribe_audio(audio_path):
    """Transcribe the audio using Whisper."""
    try:
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        inputs = whisper_processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
        with torch.no_grad():
            logits = whisper_model.generate(inputs["input_features"])
        transcript = whisper_processor.batch_decode(logits, skip_special_tokens=True)[0]
        print("*")
        print(transcript)
        print("")
        return transcript
    except Exception as e:
        print(f"Error in transcribe_audio: {e}")
        raise

@app.route('/record')
def record():
    """Render the record page where the user inputs the job role."""
    return render_template('record.html')

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    """Generate job-specific questions using the Gemini API."""
    job_role = request.form.get('job_role')
    if not job_role:
        flash('Job role is required')
        return redirect(url_for('record'))

    try:
        # Generate questions using Gemini API
        prompt = f"List 5 questions for the job role of {job_role}. Make sure the questions are relevant to the job role and important. Start with a basic introduction. do not use special charcaters like * and also do not use any placeholders. Think you are the interviewer"
        response = model.generate_content(prompt)
        print("\n")
        print(response.text)
        # Ensure the API response has the expected structure
        if not response.text.strip():
            flash('Failed to generate questions from Gemini API')
            return redirect(url_for('record'))

        questions = response.text.strip().split('\n')
        return render_template('upload.html', questions=questions)

    except Exception as e:
        print(f"Error generating questions: {e}")
        flash('Internal server error while generating questions')
        return redirect(url_for('record'))

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """Analyze the uploaded video and generate a report."""
    temp_video_path = None

    try:
        video_file = request.files.get('video')
        questions = request.form.getlist('questions')

        if not video_file or not questions:
            flash('Video and questions are required')
            return redirect(url_for('record'))

        # Save uploaded video
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, secure_filename(video_file.filename))
        video_file.save(temp_video_path)

        # Analyze video
        transcript, analysis = analyze_video_and_audio(temp_video_path)

        # Generate feedback
        questions_formatted = '\n'.join(f"{i+1}. {q}" for i, q in enumerate(questions))
        answers_prompt = f"Based on these questions:\n{questions_formatted}\nAnd these answers:\n{transcript}\nProvide short feedback and suggestions. do not use extra characters like *, just simple text and no placeholders"
        gemini_response = model.generate_content(answers_prompt)

        # Calculate metrics
        if analysis.total_frames > 0:
            eye_contact_percentage = (analysis.eye_contact_frames / analysis.total_frames) * 100
            posture_issues_percentage = (analysis.posture_issues / analysis.total_frames) * 100
        else:
            eye_contact_percentage = 0
            posture_issues_percentage = 0

        # Prepare report
        report = {
            'transcript': transcript,
            'eye_contact': f"{eye_contact_percentage:.2f}%",
            'posture_issues': f"{posture_issues_percentage:.2f}%",
            'gesture_count': analysis.gesture_counts,
            'video_duration': f"{analysis.video_duration:.1f} seconds",
            'gemini_suggestions': gemini_response.text
        }

        return render_template('report.html', report=report)
    
    except Exception as e:
        print(f"Error in analyze_video: {e}")
        traceback.print_exc()
        flash('Internal server error during analysis.')
        return redirect(url_for('record'))
        
    finally:
        # Clean up temporary files
        if temp_video_path and os.path.exists(os.path.dirname(temp_video_path)):
            try:
                gc.collect()
                shutil.rmtree(os.path.dirname(temp_video_path), ignore_errors=True)
            except Exception as e:
                print(f"Error cleaning up temporary files: {e}")

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
        "x-rapidapi-key": "48c3ed5d56msh4dbbf3910cba93fp1f1256jsn9eddbcbf6641",
        "x-rapidapi-host": "linkedin-data-api.p.rapidapi.com"
    }


    try:
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            return response.json()  # Ensure this is a dict with 'data'
        else:
            print(f"Error fetching LinkedIn jobs: {response.status_code}")
            return {"data": []}  # Return an empty list on failure
    except Exception as e:
        print(f"Exception while fetching LinkedIn jobs: {e}")
        return {"data": []}  # Return an empty list on exception
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
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
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
        else:
            print(f"Error fetching CareerJet jobs: {response.status_code}")
            return []
    except Exception as e:
        print(f"Exception while fetching CareerJet jobs: {e}")
        return []
    
@app.route('/profile')
def profile():
    return render_template('profile.html')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search_jobs():
    if request.method == 'POST':
        try:
            data = request.get_json()
            search_type = data.get("type")

            if search_type not in {"keyword", "resume", "linkedin"}:
                return jsonify({"error": "Invalid search type."}), 400

            # Shared variables
            skills = data.get("skills", [])
            location = data.get("location", "")
            
            # Keyword Search
            if search_type == "keyword":
                if not skills:
                    return jsonify({"error": "No skills provided for keyword search."}), 400
                
                results = fetch_combined_job_listings(skills, location)
                return jsonify({"results": results})

            # Resume Search
            elif search_type == "resume":
                if not skills:
                    return jsonify({"error": "No skills provided for resume search."}), 400
                
                results = fetch_combined_job_listings(skills, location)
                return jsonify({"results": results})

            # LinkedIn Search
            elif search_type == "linkedin":
                linkedin_url = data.get("linkedinUrl")
                if not linkedin_url:
                    return jsonify({"error": "LinkedIn URL not provided"}), 400
                print("linkedin url: ", linkedin_url)
                profile_data = fetch_linkedin_profile_data(linkedin_url)
                if not profile_data:
                    return jsonify({"error": "Failed to fetch LinkedIn profile data"}), 500
                
                skills = extract_skills_from_linkedin(profile_data)
                if not skills:
                    return jsonify({"error": "No skills found in LinkedIn profile"}), 400

                results = fetch_combined_job_listings(skills, location)
                return jsonify({
                    "profile_data": {
                        "name": profile_data.get("fullName", "N/A"),
                        "headline": profile_data.get("headline", "N/A"),
                        "location": profile_data.get("location", "N/A"),
                        "skills": skills
                    },
                    "results": results
                })
        except Exception as e:
            print(f"Error processing search request: {e}")
            return jsonify({"error": "Internal server error"}), 500

    return render_template('search.html')

@app.route('/analyze_profile', methods=['POST'])
def analyze_profile():
    linkedin_url = request.form.get("linkedin")
    
    if not linkedin_url:
        flash("No LinkedIn URL provided.")
        return redirect(url_for('profile'))  # Redirect to a form page or error page
    
    try:
        # Fetch LinkedIn profile data
        profile_data = fetch_linkedin_profile_data(linkedin_url)
        
        if not profile_data:
            flash("Failed to fetch LinkedIn profile data. Please verify the URL.")
            return redirect(url_for('profile'))

        # Extract relevant data from LinkedIn profile
        fname = profile_data.get("firstName", "N/A")
        lname = profile_data.get("lastName", "N/A")
        name = f"{fname} {lname}"
        headline = profile_data.get("headline", "N/A")
        skills = [skill.get("name") for skill in profile_data.get("skills", []) if skill.get("name")]
        experience = profile_data.get("position", [])

        # Prepare data for Gemini analysis
        skills_list = ", ".join(skills) if skills else "No skills found"
        experience_summary = "\n".join(
            f"{exp.get('title', 'N/A')} at {exp.get('companyName', 'N/A')}" for exp in experience
        )

        # Construct prompt for Gemini
        prompt = (
            f"Analyze the following LinkedIn profile information:\n\n"
            f"Name: {name}\n"
            f"Headline: {headline}\n"
            f"Skills: {skills_list}\n"
            f"Experience:\n{experience_summary}\n\n"
            f"Identify the strengths and weaknesses in this profile, and suggest 5 "
            f"questions an interviewer might ask based on this profile. "
            f"Provide constructive feedback in few lines in a professional tone. do not use special characters like * i need simple formatted text."
        )

        # Generate response using Gemini
        gemini_response = model.generate_content(prompt)

        # Render the analysis results on a new HTML page
        return render_template(
            "profile_analysis.html",
            name=name,
            headline=headline,
            skills=skills_list,
            experience=experience_summary,
            analysis=gemini_response.text
        )
    
    except Exception as e:
        print(f"Error in analyze_profile: {e}")
        flash("An error occurred while analyzing the profile. Please try again later.")
        return redirect(url_for('profile'))

@app.route('/compatibility_score/<job_title>', methods=['GET', 'POST'])
def compatibility_score(job_title):
    if request.method == 'POST':
        linkedin_url = request.form.get('linkedin')

        if not linkedin_url:
            flash("Please provide your LinkedIn URL.")
            return redirect(request.url)

        try:
            # Fetch LinkedIn profile data
            profile_data = fetch_linkedin_profile_data(linkedin_url)
            if not profile_data:
                flash("Failed to fetch LinkedIn profile data. Please verify the URL.")
                return redirect(request.url)

            # Extract skills from LinkedIn profile
            user_skills = extract_skills_from_linkedin(profile_data)

            # Generate required skills using Gemini
            prompt = (
                f"List the top 5 skills required for the job title '{job_title}'. "
                "Format the response as <skill1>,<skill2>,<skill3>, etc."
            )
            gemini_response = model.generate_content(prompt)
            required_skills = [skill.strip() for skill in gemini_response.text.strip().split(',')]

            # Calculate compatibility
            matched_skills = list(set(user_skills).intersection(set(required_skills)))
            compatibility_score = (len(matched_skills) / len(required_skills)) * 100 if required_skills else 0
            missing_skills = list(set(required_skills) - set(matched_skills))

            return render_template(
                'compatibility_score.html',
                job_title=job_title,
                required_skills=required_skills,
                user_skills=user_skills,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                compatibility_score=round(compatibility_score, 2)
            )

        except Exception as e:
            print(f"Error in compatibility_score: {e}")
            flash("An error occurred while calculating the compatibility score.")
            return redirect(request.url)

    return render_template('compatibility_form.html', job_title=job_title)

@app.route('/custom_message/<job_title>', methods=['GET'])
def generate_custom_message(job_title):
    linkedin_url = request.args.get('linkedin')
    if not linkedin_url:
        flash("LinkedIn URL is required.")
        return redirect(request.referrer or '/')

    try:
        profile_data = fetch_linkedin_profile_data(linkedin_url)
        if not profile_data:
            flash("Failed to fetch LinkedIn profile data.")
            return redirect(request.referrer or '/')

        # Extract user's data
        name = profile_data.get("fullName", "N/A")
        headline = profile_data.get("headline", "N/A")
        skills = ", ".join(extract_skills_from_linkedin(profile_data))
        experience = ", ".join(
            f"{exp.get('title', 'N/A')} at {exp.get('companyName', 'N/A')}"
            for exp in profile_data.get("experience", [])
        )

        # Generate custom message using Gemini
        prompt = (
            f"Create a customized cover letter for the job title '{job_title}' "
            f"using the following information about the applicant:\n\n"
            f"Name: {name}\n"
            f"Headline: {headline}\n"
            f"Skills: {skills}\n"
            f"Experience: {experience}\n\n"
            "Provide the response as a professional and structured cover letter."
        )
        gemini_response = model.generate_content(prompt)

        return render_template(
            'custom_message.html',
            job_title=job_title,
            custom_message=gemini_response.text.strip()
        )

    except Exception as e:
        print(f"Error in generate_custom_message: {e}")
        flash("An error occurred while generating the custom message.")
        return redirect(request.referrer or '/')



def fetch_linkedin_profile_data(linkedin_url):
    if not linkedin_url.startswith("https://www.linkedin.com/"):
        print("Invalid LinkedIn URL format.")
        return None

    url = "https://linkedin-data-api.p.rapidapi.com/get-profile-data-by-url"
    querystring = {"url": linkedin_url}

    headers = {
        "x-rapidapi-key": "48c3ed5d56msh4dbbf3910cba93fp1f1256jsn9eddbcbf6641",  # Replace with your RapidAPI key
        "x-rapidapi-host": "linkedin-data-api.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch LinkedIn profile: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Exception occurred while fetching LinkedIn profile: {e}")
        return None


def extract_skills_from_linkedin(profile_data):
    if not profile_data:
        return []

    skills = []
    if 'skills' in profile_data:
        skills = [skill['name'] for skill in profile_data['skills']]

    return skills

def fetch_combined_job_listings(skills, location):
    """Helper function to fetch job listings from all sources."""
    skill_query = ", ".join(skills)
    try:
        linkedin_jobs_v1 = fetch_linkedin_job_listings(skill_query, version="v1") or {"data": []}
        linkedin_jobs_v2 = fetch_linkedin_job_listings(skill_query, version="v2") or {"data": []}
        careerjet_jobs = fetch_careerjet_job_listings(skill_query, location) or []

        return (
            linkedin_jobs_v1.get('data', []) +
            linkedin_jobs_v2.get('data', []) +
            careerjet_jobs
        )
    
    except Exception as e:
        print(f"Error fetching job listings: {e}")
        return []


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
