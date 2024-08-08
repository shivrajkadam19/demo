from flask import Flask, request, jsonify
import base64
import io
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def get_result(jd_txt, resume_txt):
    content = [jd_txt, resume_txt]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    matrix = tfidf_vectorizer.fit_transform(content)
    similarity_matrix = cosine_similarity(matrix)
    match = similarity_matrix[0][1] * 100
    return match

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    job_description_base64 = data.get('job_description')
    resume_base64 = data.get('resume')

    job_description = base64.b64decode(job_description_base64)
    resume = base64.b64decode(resume_base64)

    job_description_text = pdfplumber.open(io.BytesIO(job_description)).pages[0].extract_text()
    resume_text = pdfplumber.open(io.BytesIO(resume)).pages[0].extract_text()

    match = get_result(job_description_text, resume_text)
    return jsonify({'match': round(match, 2)})

if __name__ == '__main__':
    app.run()
