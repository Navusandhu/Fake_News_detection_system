# Fake News classification

## Before executing any files,Please download all files and extract accordingly.

The identification of fraudulent posts on social media has recently attracted attention. Since this underlying prevention method of comparing websites to a list of labelled fake news sources is rigid, a machine learning technique is preferable. Our research aims to leverage Natural Language Processing to detect false news directly from news articles' text content.

## Problem Identification

Implement a machine learning algorithm to recognize whenever the news organization is propagating fake news stories. We want to deploy a database of labelled legitimate and false new articles to create a processor that can make information recommendations based on the content of the corpus. Based on many articles emanating from a source, the machine will focus on identifying credible news sources. Once a source is designated as a fake news provider, we may confidently anticipate that any future publications from that source will likewise be completely false. Because we will have many data points from each source, emphasizing on sources enlarge our article misinterpretation sensitivity.











## Initial Setup fro the web application:

For python setup you can visit here: [here](https://realpython.com/installing-python/)

Step 1: To create a virtual environment
```
$ cd web-app
$ python -m venv venv
$ source venv/Script/activate
```

Step 2: Install dependencies
```
pip install -r requirements.txt
```

Step 3: Run
```
streamlit run app.py
```
or
```
streamlit run app.py
```
This will run the app.py and serve your streamlit web application on your systems localhost, 
which could be viewed by visiting the link shown in your output console.

Console output:
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.0.107:8501
```
