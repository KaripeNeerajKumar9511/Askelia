from django.shortcuts import render
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import pymysql
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import numpy as np
import requests
import json
import cloudinary.uploader
import pdfplumber
import warnings
from .models import ReportAnalysis

# Deep Translator for multi-language support
from deep_translator import GoogleTranslator

warnings.filterwarnings("ignore", category=SyntaxWarning)

# ------------------ TF-IDF Setup ------------------ #
filename = []
word_vector = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", r" ( ", string)
    string = re.sub(r"\)", r" ) ", string)
    string = re.sub(r"\?", r" ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return cleanPost(string.strip().lower())

# Load dataset
with open('dataset/question.json', "r") as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split("#")
        cleanedLine = clean_str(arr[0])
        word_vector.append(cleanedLine.strip().lower())
        filename.append(arr[1])
    f.close()

stopwords_list = nltk.corpus.stopwords.words("english")
tfidf_vectorizer = TfidfVectorizer(
    stop_words=stopwords_list, use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
tfidf = tfidf_vectorizer.fit_transform(word_vector).toarray()
df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names_out())
df = df.values
X = np.asarray(df[:, 0:df.shape[1]])
filename = np.asarray(filename)
word_vector = np.asarray(word_vector)

# ------------------ Basic Views ------------------ #
def MyChatBot(request):
    return render(request, 'index.html')

def UserScreen(request):
    return render(request, 'UserScreen.html')

def User(request):
    return render(request, 'User.html')

def Logout(request):
    return render(request, 'index.html')

def test(request):
    return render(request, 'test.html')

def Register(request):
    return render(request, 'Register.html')

# ------------------ ChatBot Response ------------------ #
def ChatData(request):
    if request.method == 'GET':
        question = request.GET.get('mytext', '')
        lang = request.GET.get('lang', 'en')  # Get selected language
        
        if not question:
            return HttpResponse('No question received', content_type="text/plain")
        
        # Translate input to English for dataset search
        question_en = question
        if lang != 'en':
            try:
                question_en = GoogleTranslator(source='auto', target='en').translate(question)
            except Exception as e:
                print(f"Translation error: {e}")
                question_en = question
        
        # Process the English question for dataset matching
        cleanedLine = clean_str(question_en).strip().lower()
        testArray = [cleanedLine]
        testStory = tfidf_vectorizer.transform(testArray).toarray()[0]

        similarity = 0
        user_story = 'Sorry! I am not trained for given question'
        for i in range(len(X)):
            classify_user = dot(X[i], testStory)/(norm(X[i])*norm(testStory))
            if classify_user > similarity and classify_user > 0.50:
                similarity = classify_user
                user_story = filename[i]

        # Translate output back to selected language
        final_response = user_story
        if lang != 'en':
            try:
                final_response = GoogleTranslator(source='en', target=lang).translate(user_story)
            except Exception as e:
                print(f"Translation error: {e}")
                final_response = user_story

        return HttpResponse(final_response, content_type="text/plain")

# ------------------ User Login / Signup ------------------ #
def UserLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        index = 0
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root',
                              password='', database='chatbot', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("SELECT * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    index = 1
                    break
        if index == 1:
            context = {'data': f'Welcome {username}'}
            return render(request, 'UserScreen.html', context)
        else:
            context = {'data': 'Login failed'}
            return render(request, 'User.html', context)

def Signup(request):
    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        contact = request.POST.get('contact', '')
        email = request.POST.get('email', '')
        address = request.POST.get('address', '')

        db_connection = pymysql.connect(
            host='127.0.0.1', port=3306, user='root', password='', database='chatbot', charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES(%s,%s,%s,%s,%s)"
        db_cursor.execute(student_sql_query, (username, password, contact, email, address))
        db_connection.commit()

        if db_cursor.rowcount == 1:
            context = {'data': 'Signup Process Completed'}
        else:
            context = {'data': 'Error in signup process'}
        return render(request, 'Register.html', context)

# ------------------ Doctor Analysis ------------------ #

# def DoctorAnalysis(request):
#     print("DEBUG: DoctorAnalysis called with method", request.method)
#     result = {}
#     report_urls = []

#     if request.method == "POST":
#         patient_name = request.POST.get("name")
#         age = request.POST.get("age")
#         gender = request.POST.get("gender")
#         bp = request.POST.get("bp")
#         diabetic = request.POST.get("diabetic")
#         hyperthyroidism = request.POST.get("hyperthyroidism")
#         issues = request.POST.get("issues")

#         uploaded_files = request.FILES.getlist("reports")
#         report_urls = []
#         reports_for_gemini = []

#         for file in uploaded_files:
#             try:
#                 upload_result = cloudinary.uploader.upload(file, resource_type="auto")
#                 url = upload_result["secure_url"]
#                 report_urls.append(url)

#                 text_content = ""
#                 if file.name.lower().endswith(".pdf"):
#                     try:
#                         with pdfplumber.open(file) as pdf:
#                             for page in pdf.pages:
#                                 text_content += page.extract_text() + "\n"
#                     except Exception as e:
#                         text_content = f"Unable to extract text from PDF {file.name}: {str(e)}"
#                 elif file.name.lower().endswith((".txt", ".docx")):
#                     text_content = f"Text extraction not implemented for {file.name}"
#                 else:
#                     text_content = f"This is an image file: {file.name}. URL: {url}"

#                 reports_for_gemini.append({
#                     "name": file.name,
#                     "text": text_content,
#                     "url": url
#                 })
#             except Exception as e:
#                 print(f"Error uploading {file.name}: {str(e)}")

#         patient_payload = {
#             "name": patient_name,
#             "age": int(age),
#             "gender": gender,
#             "bp": bp,
#             "diabetic": diabetic,
#             "hyperthyroidism": hyperthyroidism,
#             "healthIssues": issues
#         }

#         GEMINI_API_KEY = "AIzaSyBNjoIYvWa4F_MBrQn651Ekyc_EW6D0f5g"

#         try:
#             prompt_text = (
#     "You are a medical AI assistant. Analyze the following patient data and reports carefully. "
#     "Provide detailed findings, possible conditions, suggested tests, medications, and guidance on contacting a doctor. "
#     "If the file is an image, suggest what a doctor might look for. If the file is a PDF or text, analyze the lab results, values, and symptoms. "
#     "Return only valid JSON with this format, do not include text outside JSON:\n\n"
#     "{\n"
#     "  'patientInfo': {...},\n"
#     "  'reportAnalyses':[{\n"
#     "      'reportUrl':'string',\n"
#     "      'detailedFindings':'string',\n"
#     "      'labResults':{},\n"
#     "      'anomalies':[],\n"
#     "      'possibleConditions':[],\n"
#     "      'doctorNotes':'string',\n"
#     "      'suggestedTests':['string'],\n"
#     "      'suggestedMedications':['string'],\n"
#     "      'contactDoctor':'string'\n"
#     "  }],\n"
#     "  'overallSummary':'string',\n"
#     "  'doctorRecommendations':['string']\n"
#     "}\n\n"
#     f"Patient info: {json.dumps(patient_payload)}\n"
#     f"Reports content: {json.dumps(reports_for_gemini)}\n\n"
#     "Analyze all uploaded reports and provide detailed guidance, recommendations, and next steps for the patient."
# )


#             response = requests.post(
#                 "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
#                 headers={
#                     "Content-Type": "application/json",
#                     "X-goog-api-key": GEMINI_API_KEY,
#                 },
#                 json={"contents": [{"parts": [{"text": prompt_text}]}]}
#             )

#             resp_json = response.json()
#             print("DEBUG GEMINI RESPONSE JSON:", json.dumps(resp_json, indent=2))

#             if "candidates" in resp_json and len(resp_json["candidates"]) > 0:
#                 rawText = resp_json["candidates"][0]["content"]["parts"][0]["text"]
#                 cleaned = rawText.strip()
#                 if cleaned.startswith("```json"):
#                     cleaned = cleaned[len("```json"):].strip()
#                 if cleaned.endswith("```"):
#                     cleaned = cleaned[:-3].strip()
#                 try:
#                     result = json.loads(cleaned)
#                 except Exception:
#                     result = {"error": "Invalid JSON from Gemini", "raw": cleaned}
#             else:
#                 result = {"error": "No analysis returned from Gemini", "details": resp_json}

#             # Save to DB
#             ReportAnalysis.objects.create(
#                 patient_name=patient_name,
#                 age=age,
#                 gender=gender,
#                 bp=bp,
#                 diabetic=diabetic,
#                 hyperthyroidism=hyperthyroidism,
#                 health_issues=issues,
#                 report_urls=json.dumps(report_urls),
#                 ai_result=json.dumps(result)
#             )

#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             result = {"error": str(e)}

#     return render(request, "doctor_analysis.html", {"result": result, "report_urls": report_urls})


def DoctorAnalysis(request):
    print("DEBUG: DoctorAnalysis called with method", request.method)
    result = {}
    report_urls = []

    if request.method == "POST":
        import json, base64, pdfplumber, traceback, requests, cloudinary.uploader
        from .models import ReportAnalysis

        patient_name = request.POST.get("name")
        age = request.POST.get("age")
        gender = request.POST.get("gender")
        bp = request.POST.get("bp")
        diabetic = request.POST.get("diabetic")
        hyperthyroidism = request.POST.get("hyperthyroidism")
        issues = request.POST.get("issues")

        uploaded_files = request.FILES.getlist("reports")
        report_urls = []
        reports_for_gemini = []

        for file in uploaded_files:
            try:
                # Read file bytes first (for Gemini image processing)
                file_bytes = file.read()
                file.seek(0)  # Reset pointer for Cloudinary upload

                # Upload to Cloudinary
                upload_result = cloudinary.uploader.upload(file, resource_type="auto")
                url = upload_result["secure_url"]
                report_urls.append(url)

                text_content = ""
                inline_image_data = None

                # Handle PDF
                if file.name.lower().endswith(".pdf"):
                    try:
                        with pdfplumber.open(file) as pdf:
                            for page in pdf.pages:
                                text_content += page.extract_text() + "\n"
                    except Exception as e:
                        text_content = f"Unable to extract text from PDF {file.name}: {str(e)}"

                # Handle images (JPEG, PNG, WEBP, BMP)
                elif file.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                    try:
                        inline_image_data = base64.b64encode(file_bytes).decode("utf-8")
                        text_content = f"Image file ({file.name}) prepared for visual analysis."
                    except Exception as e:
                        text_content = f"Error encoding image {file.name}: {str(e)}"

                # Handle text files
                elif file.name.lower().endswith(".txt"):
                    text_content = file_bytes.decode("utf-8")

                # Handle DOCX
                elif file.name.lower().endswith(".docx"):
                    text_content = "Text extraction not implemented for .docx yet."

                # Unsupported formats
                else:
                    text_content = f"Unsupported file type: {file.name}"

                reports_for_gemini.append({
                    "name": file.name,
                    "text": text_content,
                    "url": url,
                    "inlineData": inline_image_data
                })

            except Exception as e:
                print(f"Error uploading {file.name}: {str(e)}")

        # -------------------- Patient info --------------------
        patient_payload = {
            "name": patient_name,
            "age": int(age) if age else None,
            "gender": gender,
            "bp": bp,
            "diabetic": diabetic,
            "hyperthyroidism": hyperthyroidism,
            "healthIssues": issues
        }

        GEMINI_API_KEY = "AIzaSyBNjoIYvWa4F_MBrQn651Ekyc_EW6D0f5g"

        try:
            # ---- Construct detailed medical prompt ----
            prompt_text = (
    "You are a highly skilled medical AI assistant. Analyze the patient reports and generate a detailed medical analysis. "
    "Return only **valid JSON**, strictly following this structure:\n\n"
    "{\n"
    "  'patientInfo': {...},\n"
    "  'reportAnalyses':[{\n"
    "      'reportUrl':'string',\n"
    "      'detailedFindings':'string',\n"
    "      'labResults':{\n"
    "          'ParameterName': {\n"
    "              'value': float,\n"
    "              'unit': 'string',\n"
    "              'referenceRange': 'string',\n"
    "              'interpretation': 'string'\n"
    "          }\n"
    "      },\n"
    "      'anomalies':[],\n"
    "      'possibleConditions':[],\n"
    "      'doctorNotes':'string',\n"
    "      'suggestedTests':['string'],\n"
    "      'suggestedMedications':['string'],\n"
    "      'contactDoctor':'string'\n"
    "  }],\n"
    "  'overallSummary':'string',\n"
    "  'doctorRecommendations':['string']\n"
    "}\n\n"
    "Rules:\n"
    "1. Every lab result must be a JSON object, **never a string or Python-style dict**.\n"
    "2. Include numeric 'value', 'unit', 'referenceRange', and patient-friendly 'interpretation'.\n"
    "3. If multiple lab parameters exist, include them all in 'labResults' as separate keys.\n"
    "4. Provide full explanations, suggested tests, medications, and follow-up advice.\n"
    "5. Do not include any text outside the JSON structure.\n\n"
    f"Patient info: {json.dumps(patient_payload)}\n"
    f"Reports: {json.dumps(reports_for_gemini)}\n"
)


            # ---- Prepare Gemini parts ----
            gemini_parts = [{"text": prompt_text}]

            for report in reports_for_gemini:
                if report["inlineData"]:
                    # Detect MIME type dynamically
                    ext = report["name"].split(".")[-1].lower()
                    mime_type = (
                        "image/jpeg" if ext in ["jpg", "jpeg"]
                        else "image/png" if ext == "png"
                        else "image/webp" if ext == "webp"
                        else "image/bmp" if ext == "bmp"
                        else "application/octet-stream"
                    )
                    gemini_parts.append({
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": report["inlineData"]
                        }
                    })
                else:
                    gemini_parts.append({
                        "text": f"Report: {report['name']} - {report['text']}"
                    })

            # ---- Make Gemini API call ----
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                headers={
                    "Content-Type": "application/json",
                    "X-goog-api-key": GEMINI_API_KEY,
                },
                json={"contents": [{"parts": gemini_parts}]}
            )

            resp_json = response.json()
            print("DEBUG GEMINI RESPONSE JSON:", json.dumps(resp_json, indent=2))

            # ---- Parse Gemini response ----
            if "candidates" in resp_json and len(resp_json["candidates"]) > 0:
                rawText = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                cleaned = rawText.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[len("```json"):].strip()
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()
                try:
                    result = json.loads(cleaned)
                except Exception:
                    result = {"error": "Invalid JSON from Gemini", "raw": cleaned}
            else:
                result = {"error": "No analysis returned from Gemini", "details": resp_json}

            # ---- Save analysis to DB ----
            ReportAnalysis.objects.create(
                patient_name=patient_name,
                age=age,
                gender=gender,
                bp=bp,
                diabetic=diabetic,
                hyperthyroidism=hyperthyroidism,
                health_issues=issues,
                report_urls=json.dumps(report_urls),
                ai_result=json.dumps(result)
            )

        except Exception as e:
            traceback.print_exc()
            result = {"error": str(e)}

    return render(request, "doctor_analysis.html", {"result": result, "report_urls": report_urls})
