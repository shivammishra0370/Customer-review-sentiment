# Customer-review-sentiment
An interactive Web Application built with Streamlit that utilizes Natural Language Processing (NLP) to classify and visualize the emotional tone of social media text.

**[🌐 Click Here to View Live Streamlit Application](https://customer-review-sentiment-0370.streamlit.app/)** 

## 📊 Dataset Source
**Sentiment140 Dataset**
**Official Repository Link:**  [Kaggle Sentiment140 Dataset Hub](https://www.kaggle.com/datasets/kazanova/sentiment140).


## Problem Statement
Social media platforms like Twitter generate millions of unstructured text updates every second. For companies, researchers, and brands, tracking public opinion or customer feedback manually is impossible. 

The goal of this project is to build an automated, end-to-end Machine Learning pipeline that takes unstructured text inputs, cleans social media noise (such as hashtags, hyperlinks, and username mentions), and instantly categorizes the underlying sentiment as either **Positive** or **Negative**. To ensure flexibility and transparency, the system provides real-time confidence scores and comparative evaluations using multiple classification algorithms.

---

## Tools Used
* **VS Code:** The primary Integrated Development Environment (IDE) used to write, debug, and organize the app script.
* **Jupyter Notebook:** Used for initial data exploration, text cleaning experimentation, model training, and performance metrics plotting.
* **GitHub:** Utilized for cloud version control and project hosting repository management.
* **Streamlit Community Cloud:** Used to deploy the local project files into a publicly accessible live web server.

---

## Libraries Used
* **Streamlit (`streamlit`):** Powering the entire interactive frontend web dashboard layout and sidebar navigation widgets.
* **Scikit-Learn (`scikit-learn`):** Used to build the `CountVectorizer` feature extraction model and train the underlying `MultinomialNB` and `LogisticRegression` classification brains.
* **Pandas (`pandas`):** Used for loading datasets, slicing balanced training data rows, and structuring metric tables.
* **NumPy (`numpy`):** Handles internal vectorized mathematical calculations and array operations.
* **Plotly (`plotly`):** Generates real-time, interactive visual assets like the distribution pie chart and the top keywords bar chart.
* **Seaborn & Matplotlib (`seaborn` / `matplotlib`):** Utilized inside the training notebook to draw confusion matrix heatmaps and evaluate models side-by-side.
* **Regex (`re`) & String (`string`):** Applied to scrub noise, strip punctuation, and clean standard Twitter artifacts.

---

## Key Project Outcomes
1. **Dynamic Model Swapping:** Developed a comparative UI allowing users to switch classification engines (Naive Bayes vs. Logistic Regression) on the fly to see how prediction confidence changes.
2. **Robust Baseline Performance:** Successfully trained both models on a balanced slice of the Kaggle Sentiment140 dataset, hitting a strong overall baseline accuracy of **91%**.
3. **Optimized Inference:** Serialized the training weights into lightweight `.pkl` files using Pickle, reducing dashboard text processing time to under a second without needing local training dataset storage.
4. **Interactive Dashboard View:** Designed an analytics UI featuring high-impact charts, data table break-downs, and explicit color-coded indicator blocks for an accessible presentation experience.
