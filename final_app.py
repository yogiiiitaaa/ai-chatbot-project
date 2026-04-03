import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import time
import random
import matplotlib.pyplot as plt# import matplotlib.pyplot as plt
# import seaborn as sns
import seaborn as sns

from textblob import TextBlob
from openai import OpenAI
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats

# =========================
# GPT SETUP (SAFE WAY)
# =========================
import os
client = OpenAI(api_key=os.getenv("1bc92249-965f-4891-80b7-10133068b0e1"))

# =========================
# DATABASE
# =========================
conn = sqlite3.connect("project.db", check_same_thread=False)
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS users(username TEXT, password TEXT)")
c.execute('''CREATE TABLE IF NOT EXISTS chats(
msg TEXT, reply TEXT, length INT, time REAL, sentiment REAL)''')
conn.commit()

# =========================
# HASH PASSWORD
# =========================
def hash_pass(p):
    return hashlib.sha256(p.encode()).hexdigest()

# =========================
# UI CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("🔥 AI Chatbot + Analytics System")

menu = ["Login","Signup"]
choice = st.sidebar.selectbox("Menu", menu)

# =========================
# SIGNUP
# =========================
if choice == "Signup":
    st.subheader("Create Account")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Signup"):
        c.execute("INSERT INTO users VALUES (?,?)",(user, hash_pass(pwd)))
        conn.commit()
        st.success("Account Created!")

# =========================
# LOGIN
# =========================
elif choice == "Login":
    st.subheader("Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?",
                  (user, hash_pass(pwd)))
        data = c.fetchall()

        if data:
            st.success("Login Success ✅")

            st.subheader("💬 Chatbot")

            # GPT BOT
            def bot(msg):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content":msg}]
                    )
                    return response.choices[0].message.content
                except:
                    return random.choice(["Hello 😊","Try again","Nice 👍"])

            msg = st.text_input("Enter Message")

            if st.button("Send"):
                start = time.time()
                reply = bot(msg)
                end = time.time()

                sentiment = TextBlob(msg).sentiment.polarity

                c.execute("INSERT INTO chats VALUES (?,?,?,?,?)",
                          (msg, reply, len(msg), end-start, sentiment))
                conn.commit()

                st.success(reply)

            # LOAD DATA
            df = pd.read_sql("SELECT * FROM chats", conn)

            if len(df) > 5:

                st.subheader("📊 Data")
                st.write(df)

                df.dropna(inplace=True)

                X = df[['length']]
                y = df['time']

                # Linear Regression
                lin = LinearRegression().fit(X,y)
                st.write("Slope:", lin.coef_[0])

                # Logistic Regression
                df['label'] = df['sentiment'].apply(lambda x: 1 if x>0 else 0)

                if len(df['label'].unique()) > 1:
                    log = LogisticRegression().fit(X, df['label'])
                    pred = log.predict(X)
                    st.write("Accuracy:", accuracy_score(df['label'], pred))

                # Decision Tree
                tree = DecisionTreeClassifier().fit(X, df['label'])

                # Pipeline
                pipe = Pipeline([
                    ('scale', StandardScaler()),
                    ('model', LogisticRegression())
                ])

                if len(df['label'].unique()) > 1:
                    pipe.fit(X, df['label'])
                    st.write("Pipeline Working ✅")

                # Statistics
                st.subheader("📈 Statistics")
                st.write("Mean:", df['length'].mean())
                st.write("Std Dev:", df['length'].std())

                # T-test
                pos = df[df['sentiment']>0]['length']
                neg = df[df['sentiment']<=0]['length']

                if len(pos)>1 and len(neg)>1:
                    t,p = stats.ttest_ind(pos, neg)
                    st.write("T-test:", t)
                    st.write("P-value:", p)

                # Graphs
                st.subheader("📊 Visualizations")

                st.line_chart(df['time'])
                st.bar_chart(df['length'])

                fig, ax = plt.subplots()
                ax.hist(df['length'])
                st.pyplot(fig)

                fig2, ax2 = plt.subplots()
                ax2.scatter(df['length'], df['time'])
                st.pyplot(fig2)

                # Heatmap
                fig4, ax4 = plt.subplots()
                sns.heatmap(df.corr(), annot=True, ax=ax4)
                st.pyplot(fig4)

        else:
            st.error("Invalid Login ❌")
