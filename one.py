from flask import Flask, request, jsonify
import requests
from googleapiclient.discovery import build
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

from googleapiclient.errors import HttpError
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__)

api_key = 'AIzaSyAiK9L0hvKFxyZJzAY8u3p7jPHhRavHccs'
model_path = './ft_model'

tokenizer_path = './ft_tokenizer'
model = BertForSequenceClassification.from_pretrained(model_path)

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)


@app.route('/recommend/<vid>', methods=['GET'])
def recommend(vid):
    title, desc = fetch_video_info(vid)
    
    if title:
        words = fetch_biased_words(title)

        keywords = extract_keywords(words)
        
        if not keywords:
            keywords = get_low_confidence(words)
        
        video_df = fetch_top_videos(keywords)



        entry = pd.Series({"Keyword": "curr", "Video ID": vid, "Title": title, "Description": desc})

        video_df = video_df._append(entry, ignore_index=True)

        recommendations = generate_recommendations(video_df, vid)

        return jsonify({
            "video_id": vid,
            "title": title,

            "recommendations": recommendations
        })
    else:
        return jsonify({"error": "Video not found"}), 404
    




def fetch_video_info(vid):
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)

        response = youtube.videos().list(part='snippet', id=vid).execute()
        
        if 'items' in response and len(response['items']) > 0:
            video_info = response['items'][0]['snippet']

            video_title = video_info.get('title')

            video_desc = video_info.get('description')

            return video_title, video_desc
        else:
            return None, None

    except HttpError as e:
        print(f'Error fetching video info: {e}')
        return None, None

def fetch_biased_words(title):
    API_URL = "https://api-inference.huggingface.co/models/newsmediabias/UnBIAS-Named-Entity-Recognition"

    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while querying the API: {e}")
            return None

    payload = {"inputs": title}
    output = query(payload)
    
    words = []
    if output:
        for i in range(len(output)):
            words.append(output[i]['word'])
    return words

def extract_keywords(words):

    inputs = tokenizer(words, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        outs = model(**inputs)
    probs = torch.nn.functional.softmax(outs.logits, dim=-1)

    labels = ["Not Related", "Related to Education Study Coding"]

    predictions = [(word, labels[prob.argmax()], prob.max().item()) for word, prob in zip(words, probs)]

    keywords = []
    for word, label, confidence in predictions:
        if label == "Related to Education Study Coding":

            keywords.append(word)
    return keywords





def get_low_confidence(words):


    inputs = tokenizer(words, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outs = model(**inputs)
    probs = torch.nn.functional.softmax(outs.logits, dim=-1)


    labels = ["Not Related", "Related to Education Study Coding"]
    predictions = [(word, labels[prob.argmax()], prob.max().item()) for word, prob in zip(words, probs)]

    min_conf = float('inf')
    min_conf_word = None



    for word, label, confidence in predictions:
        if label == "Not Related" and confidence < min_conf:
            min_conf = confidence
            min_conf_word = word
    if min_conf_word:
        return [min_conf_word]
    return []





def fetch_top_videos(keywords, max_results=10):


    youtube = build('youtube', 'v3', developerKey=api_key)
    results = []

    for keyword in keywords:
        request = youtube.search().list(q=keyword, part='snippet', type='video', maxResults=max_results)
        response = request.execute()
        for result in response['items']:
            vid = result['id']['videoId']


            desc = result['snippet']['description']

            desc = re.sub(r'http\S+|www\S+|https\S+', '', desc, flags=re.MULTILINE)
            results.append({"Keyword": keyword, "Video ID": vid, "Description": desc})

    return pd.DataFrame(results)





def generate_recommendations(video_df, vid):
    tfv = TfidfVectorizer(min_df=3, max_features=None, 
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), stop_words='english')
    

    tfv_matrix = tfv.fit_transform(video_df['Description'])


    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    indices = pd.Series(video_df.index, index=video_df['Video ID']).drop_duplicates()
    video_idx = indices[vid]


    sim_scores = list(enumerate(sig[video_idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    recommendations = []

    for i, score in sim_scores:
        
        similar_vid = video_df.iloc[i]['Video ID']

        recommendations.append(similar_vid)

    return recommendations

if __name__ == '__main__':
    app.run(debug=True)
