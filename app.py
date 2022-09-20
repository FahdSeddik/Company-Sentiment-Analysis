"""
Streamlit app
"""

import streamlit as st 
import pandas as pd
import snscrape.modules.twitter as sntwitter
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode
from transformers import pipeline

def setup_model():
    """
    Setup Model
    """
    # this will download 2 GB
    nlp = pipeline("sentiment-analysis", model='XLM-R-L-ARABIC-SENT')
    return nlp

def get_tweets(query,limit):
    """
    Get Tweets
    """
    #query = '"IBM" min_replies:10 min_faves:500 min_retweets:10 lang:ar'
    tweets = []
    pbar = st.progress(0)
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        pbar.progress(int(len(tweets)/limit*100))
        if len(tweets)==limit:
            break
        else:
            tweets.append([tweet.content])
    return tweets


# Fxn
def convert_to_df(sentiment):
    """
    Convert to df
    """
    label2 = sentiment.LABEL_2 if 'LABEL_2' in sentiment.index else 0
    label1 = sentiment.LABEL_1 if 'LABEL_1' in sentiment.index else 0
    label0 = sentiment.LABEL_0 if 'LABEL_0' in sentiment.index else 0
    sentiment_dict = {'Positive':label2,'Neutral':label0,'Negative':label1}
    sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['Sentiment','Count'])
    return sentiment_df


def get_sentiments(tweets,model):
    """
    Get sentiments
    """
    sentiments = []
    pbar = st.progress(0)
    for i,tweet in enumerate(tweets):
        pbar.progress(int((i+1)/len(tweets)*100))
        sentiments.append(model(tweet)[0]['label'])
    return sentiments

def main():
    """
    Main
    """
    st.title("Sentiment Analysis NLP App")
    st.subheader("Company Based Sentiment")

    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Home")
        temp = st.slider("Choose sample size",min_value=100,max_value=1000)
        #st.write("Number of Samples: ",temp)
        opt = st.selectbox('Language',pd.Series(['Arabic','English']))
        likes = st.slider("Minimum number of likes on tweet",min_value=0,max_value=500)
        retweets = st.slider("Minimum number of retweets on tweet",min_value=0,max_value=50)
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter company name here")
            submit_button = st.form_submit_button(label='Analyze')       
        # layout
        if submit_button:
            st.success(f'Selected Language: {opt}',)
            if opt=='Arabic':
                query = raw_text + ' lang:ar'
            else:
                query = raw_text + ' lang:en'
            query += f' min_faves:{likes} min_retweets:{retweets}'
            st.write('Retreiving Tweets...')
            tweets = get_tweets(query,temp)
            st.write(f'Found {len(tweets)}/{temp}')
            st.write('Loading Model...')
            nlp = setup_model()
            st.write('Analyzing Sentiments...')
            sentiments = get_sentiments(tweets,nlp)
            st.success('DONE')
            st.subheader("Results of "+raw_text)
            counts = pd.Series(sentiments).value_counts()
            #col1,col2 = st.columns(2)
            # Dataframe
            result_df = convert_to_df(counts)
            st.dataframe(result_df)

            fig1, ax1 = plt.subplots(figsize=(5,5))
            ax1.pie(result_df['Count'],labels=result_df['Sentiment'].values, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig1)
            tweets = pd.Series(tweets,name='Tweet')
            sentiments = pd.Series(sentiments,name='Sentiment (pred)').replace(
                {'LABEL_2':'Positive','LABEL_1':'Negative','LABEL_0':'Neutral'})
            all_df = pd.merge(left=tweets,right=sentiments,left_index=True,right_index=True)
            st.subheader("Tweets")
            gb = GridOptionsBuilder.from_dataframe(all_df)
            gb.configure_side_bar()
            grid_options = gb.build()
            AgGrid(
                all_df,
                gridOptions=grid_options,
                enable_enterprise_modules=True,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                fit_columns_on_grid_load=False,
            )
    else:
        st.subheader("About")


if __name__ == '__main__':
	main()