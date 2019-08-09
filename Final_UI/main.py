
from flask import Flask, request, render_template,send_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)


@app.route('/download')
def download():
    path = "/Users/Keerthy/Desktop/summer/new/Final_UI/Data/filee.pdf"
    return send_file(path, as_attachment=True)

@app.route("/recommender")
def recommender():
    return render_template('recommender.html')


@app.route("/results",methods=["GET","POST"])
def results():
	d=request.form.get("company")
	df2=getdata()
	da=recommend(d,df2)
	data=da.to_frame()
	return render_template('results.html',d=data)


@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/home")
def home():
    return render_template('home.html')

def getdata():
	df = pd.read_csv("/Users/Keerthy/Desktop/summer/new/Final_UI/Data/crunchbase-investments.csv", encoding='unicode_escape',low_memory=False)
	df=df.drop(['company_permalink','investor_permalink','investor_category_code','funded_at', 'funded_month', 'funded_quarter','funded_year','investor_state_code','investor_country_code','investor_city','raised_amount_usd'],axis=1)
	df=df.dropna(axis=0, subset=['company_category_code','company_state_code','company_country_code','company_region','company_city','funding_round_type','company_name','investor_name','investor_region'])
	df = df.drop_duplicates(subset='company_name', keep='first')

	features = ['company_category_code', 'company_country_code','company_state_code', 'company_region', 'company_city', 'investor_name','investor_region', 'funding_round_type']
	df1=df

	for feature in features:
		df1[feature] = df1[feature].apply(clean_data)

	df1['metric']=df1.apply(" ".join, axis=1)
	df2=df1.drop(['company_category_code', 'company_country_code','company_state_code', 'company_region', 'company_city', 'investor_name','investor_region', 'funding_round_type'],axis=1) 
	return df

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def recommend(a,df2):
	tf = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
	tfidf_matrix = tf.fit_transform(df2['metric'])
	cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
	df2= df2.reset_index()
	indices = pd.Series(df2.index, index=df2['company_name'])
	result=get_recommendations(a,cosine_sim,indices,df2)
	return result


           
def get_recommendations(company_name, cosine_sim,indices,df):
    idx = indices[company_name]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    company_indices = [i[0] for i in sim_scores]

    return df['company_name'].iloc[company_indices]

if __name__ == "__main__":
    app.run(debug=True)
