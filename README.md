<p align="center">

  <img src="https://www.freecodecamp.org/news/content/images/2020/09/wall-5.jpeg" alt="sentiments" width="500"/>

</p>

<hr>

# Company-Sentiment-Analysis
This is a Streamlit app for sentiment analysis that uses [XLM RoBERTa](https://huggingface.co/akhooli/xlm-r-large-arabic-sent) model on Hugging Face. The program scrapes twitter for tweets that are about a certain company.
The tweets are then fed into a model for sentiment analysis.  
* You can find our CUSTOM MODEL in this repo [here](https://github.com/FahdSeddik/EgyptionCompaniesReviews_Sentiment_analysis)

# Interface
Below is a video demo of the app.   

https://user-images.githubusercontent.com/62207434/192290351-86ad659d-ffcf-4e9a-bcec-2c01dc6119a0.mp4

# Installation
To be able to use this app, please follow the instructions below. First, you need to install requirements using the following command.
```bash
pip install -r requirements.txt
```
After that, you need to download this model from [Hugging Face](https://huggingface.co/akhooli/xlm-r-large-arabic-sent). 
```python
from transformers import pipeline
import tokenizers
# this will download 2 GB
nlp = pipeline("sentiment-analysis", model='akhooli/xlm-r-large-arabic-sent')

# Save it in the same app folder
# .save_pretrained(path)
# 'XLM-R-L-ARABIC-SENT' is the folder name of the model
nlp.save_pretrained('XLM-R-L-ARABIC-SENT')
```
This will produce a folder that has the model. ***Please include the folder in the same directory as 'app.py'.***  
In case you want to replace this model with another, you want to download your model and edit the `setup_model()` function. Implementation is shown below.
```python
def setup_model():
    """
    Setup Model
    """
    #*************************************************
    #  -==EDIT THE LINE BELOW WITH YOUR OWN MODEL==-
    #*************************************************
    nlp = pipeline("sentiment-analysis", model='XLM-R-L-ARABIC-SENT')
    return nlp
```
Now to run the app, just simply run the command below in a terminal.
```bash
streamlit run app.py
```
# How to use?
Below you can find all the settings you can tweak that are related to your query.  

<img src="https://user-images.githubusercontent.com/62207434/192295281-e7dd7b7e-685b-482b-90dd-6868ce5e8d92.png" alt="sentiments" width="500"/>  

If the program was unable to find enough tweets for a given query then the message below will show.  

![image](https://user-images.githubusercontent.com/62207434/192296501-cc9cfafa-f680-4504-8ae7-8d4dc8eb6faa.png)

**WordCloud is not available for Arabic due to the library not being compatable with Arabic.**


# About 
This Streamlit app was developed as part of my Data Scientist intern position at Banque Misr.
