'''
Streamlit app - Your Resident Sommelier. This app aims to provide users with wine recommendations based on their usual wine choices. 

'''

import glob
import os
import re

import streamlit as st 
import pandas as pd
import numpy as np
import hmac

from pathlib import Path
from PIL import Image
from openai import OpenAI
import time
import streamlit.components.v1 as components


api_key = st.secrets["api-keys"]["api_openai"]
client = OpenAI(api_key= api_key)

assistant_id = 'asst_xnVAoWK2M7sEgowyLZXsrpHN'

hh_favicon_path = Path(__file__).parents[1] / "streamlit/h365.png"
hh_favicon = Image.open(hh_favicon_path)

hh_path = Path(__file__).parents[1] / "streamlit/banner.jpg"
hh_image = Image.open(hh_path)

pc_path = Path(__file__).parents[1] / "streamlit/prediction_collaborative.csv"
colab_rec = pd.read_csv(pc_path)


content_path = Path(__file__).parents[1] / "streamlit/content_results2.csv"
content_rec = pd.read_csv(content_path)

wine_master = Path(__file__).parents[1] / "streamlit/wine_dictionary.csv"
wine_data_index = pd.read_csv(wine_master)

colab_rec = colab_rec.rename(columns = {"Unnamed: 0": "wineid"})
colab_rec = colab_rec.set_index("wineid")
content_rec = content_rec.set_index("wineid")

#### Functions implementation ## 

def recommend_me_wines(user):
    try:
        if user["new"] == 1:
            df1 = pd.DataFrame(content_rec.loc[:, str(user["wine_id_chosen"])].sort_values(ascending = False))
            df1 = df1[df1.index != user["wine_id_chosen"]].reset_index()
            df1 = df1.rename(columns = {"wineid": "WineID"})
            df2 = pd.merge(df1,wine_data_index, on = "WineID")
            df2 = df2[df2["Type"] == user["interest"]].head(20)
            filtered_list = wine_data_index[wine_data_index["WineID"].isin(df2["WineID"].values)]
            random_recommendations = (filtered_list.sample(n = 5))["full_text"]
            return random_recommendations
        else:
            df1 = pd.DataFrame(colab_rec.loc[:,str(user["userid"])].sort_values(ascending = False))
            df1 = df1[df1.index != user["userid"]].reset_index()
            df1 = df1.rename(columns = {"wineid": "WineID"})
            df2 = pd.merge(df1,wine_data_index, on = "WineID")
            df2 = df2[df2["Type"] == user["interest"]].head(20)
            filtered_list = wine_data_index[wine_data_index["WineID"].isin(df2["WineID"].values)]
            random_recommendations = (filtered_list.sample(n = 5))["full_text"]
            return random_recommendations
    except:
        print("please provide valid user id")


###### Streamlit app page layout and data logic -- start here ###### 
st.set_page_config(page_title='NTUC Fairprice Wine Sommelier', 
                   page_icon=hh_favicon, 
                   layout='wide', 
                   initial_sidebar_state='expanded',
                   menu_items= {
                       'Get Help':'http://localhost:8501',
                       'Report a bug':'http://localhost:8501',
                       'About':'http://localhost:8501'                                              

                   })


st.image(hh_image, use_column_width=True)

# Sidebar
st.sidebar.header(("About"))
st.sidebar.markdown((
    "This wine recommender was created as part of my graduation capstone project for General Assembly DSI-SG-42. It's important to note that this application is not affiliated with NTUC Fairprice."
))

st.sidebar.header(("Instructions"))
st.sidebar.markdown((
    """
To test out the flow for an existing user (collaborative filtering):
- Username: jia_sheng
- Password: zhenggangisawesome

To test out the flow for a new user (content based cold-start):
- Username: jia_en
- Password: zhenggangisawesome

"""
))

st.sidebar.header(("Details"))
st.sidebar.markdown((
    """
There are three key components for this application:
- Building a Recommender System using the SVD algorithm within Surprise library.
- Generating Content and automated ChatBot through the OpenAI API. ChatBot obtains his wine knowledge from key wikipedia pages about wines. 
- Implementing a Tableau Dashboard to enhance user experience with valuable features.

"""
))

st.sidebar.header(("Contact"))
st.sidebar.markdown((
    "[Linkedin](https://www.linkedin.com/in/ziggy-lim/)"
))

def check_password():
    """Returns `True` if the user had a correct password."""
    
    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username", placeholder = "jia_sheng")
            st.text_input("Password", type="password", key="password", placeholder = "zhenggangisawesome")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            # del st.session_state["password"]  # Don't store the username or password.
            # del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect. Please contact customer support.")
    return False

if not check_password():
    # st.markdown(st.session_state["username"])
    st.stop()

#### Variable configuration ### 

tab1, tab2 = st.tabs(["Your Wine Sommelier","Chat with me!"])

user_id = st.session_state["username"]

with tab1:
    
    # st.header("⏳Your Resident Sommelier⏳")
    st.subheader("Find out new wine recommendations in our store")

    # Check if an option is selected
    if user_id == "jia_en":
        st.session_state["username"] = "jia_en"
        st.write(f'I see that you are here for the first time. Lets make your first purchase!')
        st.write(f'Fill up the below onboarding form for us to get to know you better and kickstart you on your wine journey')
        with st.form("Wine Preference Quiz"):
            st.write("Answer the following questions to determine your wine preference.")

            # Define questions and options
            questions = [
                "How do you usually like your beverages?",
                "What type of food do you enjoy pairing with wine?",
                "How do you prefer your wine to taste?",
                "Which aroma do you find most appealing in wine?"
            ]

            options = [
                ["Sweet and fruity", "Dry and crisp", "Bold and full-bodied"],
                ["Light salads and seafood", "Grilled chicken or pasta dishes", "Rich meats or aged cheeses"],
                ["Light and refreshing", "Balanced and smooth", "Intense and complex"],
                ["Floral and citrus", "Fruity and herbal", "Spicy and oaky"]
            ]

            # Initialize result counters
            result_counts = {"0": 0, "1": 0, "2": 0}

            # Display questions and collect answers
            for i, question in enumerate(questions):
                st.subheader(f"Question {i+1}: {question}")
                option_selected = st.radio(f"Select an option:", options[i], key=f"question_{i}")
                index = options[i].index(option_selected)
                result_counts[str(index)] += 1
            
            user_cuisine = st.selectbox("What kind of dish are you asking about pairing with?", ["Singapore", "Italian","Chinese", "Mexican", "Indian", "Japanese", "Thai", "French", "Mediterranean", "American", "Spanish", "Korean", "Greek", "Lebanese", "Vietnamese", "Brazilian"])

            st.divider()
            submit1 = st.form_submit_button('Submit') 

            if submit1:
                # Determine wine preference based on majority of answers
                preference = max(result_counts, key=result_counts.get)

                # Display result
                st.write("Based on your answers, your wine preference leans towards:")
                if preference == "0":
                    st.write("Light and refreshing wines like Sauvignon Blanc or Riesling")
                    user = {
                        "new": 1,
                        "interest": "White",
                        "userid": "",
                        "wine_id_chosen": 193485
                        }
                    
                    recommend = recommend_me_wines(user)
                    recommended_wines_renamed = pd.DataFrame(recommend).rename(columns={"full_text": "Curated List specially for you"})
                    st.markdown(recommended_wines_renamed.style.hide(axis="index").to_html(), unsafe_allow_html=True)
                    assistant = client.beta.assistants.retrieve(assistant_id)
                    st.text("")
                    st.text("")
                    st.subheader(f"Learn more about our recommendations! :grapes:")
                    with st.spinner("Wait uh....."):
                        assistant.description = "You are a helpful robot wine sommelier. You will only reply to wine related issues, else reply that you do not know."
                        client.beta.assistants.update(assistant.id, description=assistant.description)
                        thread = client.beta.threads.create()
                        message = client.beta.threads.messages.create(thread_id=thread.id,role="user",
                                                                    content= f"please give me a brief description of each wine mentioned here {recommend} as well as the food pairing that goes with it in a short and concise manner. Food pairings to be related to {user_cuisine} cuisine context.")
                        run = client.beta.threads.runs.create_and_poll(thread_id=thread.id,assistant_id=assistant.id,instructions="Replies should be in point form. Two points under each wine recommendation. These two points are 'description' and 'food pairing'. fFood pairing should just be itemised response and concise. ")
                        if run.status == 'completed': 
                            messages = client.beta.threads.messages.list(
                                thread_id=thread.id
                            )
                            st.markdown(messages.data[0].content[0].text.value)
                        else:
                            print(run.status)
                    st.success("Hope you enjoy our recommendation!")
                    st.text("")
 
                elif preference == "1":
                    st.write("Versatile and balanced wines like Pinot Noir or Chardonnay")
                    
                    user = {
                        "new": 1,
                        "interest": "Red",
                        "userid": "",
                        "wine_id_chosen": 171438
                        }
                    
                    recommend = recommend_me_wines(user)
                    recommended_wines_renamed = pd.DataFrame(recommend).rename(columns={"full_text": "Curated List specially for you"})
                    st.markdown(recommended_wines_renamed.style.hide(axis="index").to_html(), unsafe_allow_html=True)
                    assistant = client.beta.assistants.retrieve(assistant_id)
                    st.text("")
                    st.text("")
                    st.subheader(f"Learn more about our recommendations! :grapes:")
                    with st.spinner("Wait uh....."):
                        assistant.description = "You are a helpful robot wine sommelier. You will only reply to wine related issues, else reply that you do not know."
                        client.beta.assistants.update(assistant.id, description=assistant.description)
                        thread = client.beta.threads.create()
                        message = client.beta.threads.messages.create(thread_id=thread.id,role="user",
                                                                    content= f"please give me a brief description of each wine mentioned here {recommend} as well as the food pairing that goes with it in a short and concise manner. Food pairings to be related to {user_cuisine} cuisine context.")
                        run = client.beta.threads.runs.create_and_poll(thread_id=thread.id,assistant_id=assistant.id,instructions="Replies should be in point form. Two points under each wine recommendation. These two points are 'description' and 'food pairing'. Food pairing should just be itemised response and concise. ")
                        if run.status == 'completed': 
                            messages = client.beta.threads.messages.list(
                                thread_id=thread.id
                            )
                            st.markdown(messages.data[0].content[0].text.value)
                        else:
                            print(run.status)
                    st.success("Hope you enjoy our recommendation!")
                    st.text("")

                else:
                    st.write("Bold and complex wines like Cabernet Sauvignon or Syrah")
                    user = {
                        "new": 1,
                        "interest": "Red",
                        "userid": "",
                        "wine_id_chosen": "111495"
                        }
                    recommend = recommend_me_wines(user)
                    recommended_wines_renamed = pd.DataFrame(recommend).rename(columns={"full_text": "Curated List specially for you"})
                    st.markdown(recommended_wines_renamed.style.hide(axis="index").to_html(), unsafe_allow_html=True)
                    assistant = client.beta.assistants.retrieve(assistant_id)
                    st.text("")
                    st.text("")
                    st.subheader(f"Learn more about our recommendations! :grapes:")
                    with st.spinner("Wait uh....."):
                        assistant.description = "You are a helpful robot wine sommelier. You will only reply to wine related issues, else reply that you do not know."
                        client.beta.assistants.update(assistant.id, description=assistant.description)
                        thread = client.beta.threads.create()
                        message = client.beta.threads.messages.create(thread_id=thread.id,role="user",
                                                                    content= f"please give me a brief description of each wine mentioned here {recommend} as well as the food pairing that goes with it in a short and concise manner. Food pairings to be related to {user_cuisine} cuisine context.")
                        run = client.beta.threads.runs.create_and_poll(thread_id=thread.id,assistant_id=assistant.id,instructions="Replies should be in point form. Two points under each wine recommendation. These two points are 'description' and 'food pairing'. Food pairing should just be itemised response and concise. ")
                        if run.status == 'completed': 
                            messages = client.beta.threads.messages.list(
                                thread_id=thread.id
                            )
                            st.markdown(messages.data[0].content[0].text.value)
                        else:
                            print(run.status)
                    st.success("Hope you enjoy our recommendation!")
                    st.text("")                   

    elif user_id == "jia_sheng":
        st.session_state["username"] = "jia_sheng"
        user_new_id = 1356810

        if str(user_new_id) in colab_rec.columns:
            st.write(f'Welcome back Jia Sheng! For me to provide you with better recommendations, can you let me know if you are feeling a red or white today?')
            user_interest = st.selectbox("Red or White?", ["","Red", "White"])
            user_cuisine = st.selectbox("What kind of dish are you asking about pairing with?", ["Singapore", "Italian","Chinese", "Mexican", "Indian", "Japanese", "Thai", "French", "Mediterranean", "American", "Spanish", "Korean", "Greek", "Lebanese", "Vietnamese", "Brazilian"])
            if user_interest == "Red":
                st.write(f"Good choice for today's weather. We think that you will like the below offerings available in your nearest fairprice outlet.")
                user = {
                    "new": 0,
                    "interest": "Red",
                    "userid": 1356810,
                    "wine_id_chosen": ""
                    }
                
                recommend = recommend_me_wines(user)
                recommended_wines_renamed = pd.DataFrame(recommend).rename(columns={"full_text": "Curated List specially for you"})
                st.markdown(recommended_wines_renamed.style.hide(axis="index").to_html(), unsafe_allow_html=True)
                assistant = client.beta.assistants.retrieve(assistant_id)
                st.text("")
                st.text("")
                st.subheader(f"Learn more about our recommendations! :grapes:")
                with st.spinner("Wait uh....."):
                    assistant.description = "You are a helpful robot wine sommelier. You will only reply to wine related issues, else reply that you do not know."
                    client.beta.assistants.update(assistant.id, description=assistant.description)
                    thread = client.beta.threads.create()
                    message = client.beta.threads.messages.create(thread_id=thread.id,role="user",
                                                                content= f"please give me a brief description of each wine mentioned here {recommend} as well as the food pairing that goes with it in a short and concise manner. Food pairings to be related to {user_cuisine} cuisine context.")
                    run = client.beta.threads.runs.create_and_poll(thread_id=thread.id,assistant_id=assistant.id,instructions="Replies should be in point form. Two points under each wine recommendation. These two points are 'description' and 'food pairing'. Food pairing should just be itemised response and concise. ")
                    if run.status == 'completed': 
                        messages = client.beta.threads.messages.list(
                            thread_id=thread.id
                        )
                        st.markdown(messages.data[0].content[0].text.value)
                    else:
                        print(run.status)
                st.success("Hope you enjoy our recommendation!")
                st.text("")

            elif user_interest == "White":
                st.write(f"Good choice for today's weather. We think that you will like the below offerings available in your nearest fairprice outlet.")
                user = {
                    "new": 0,
                    "interest": "White",
                    "userid": 1356810,
                    "wine_id_chosen": ""
                    }
                recommend = recommend_me_wines(user)
                recommended_wines_renamed = pd.DataFrame(recommend).rename(columns={"full_text": "Curated List specially for you"})
                st.markdown(recommended_wines_renamed.style.hide(axis="index").to_html(), unsafe_allow_html=True)
                assistant = client.beta.assistants.retrieve(assistant_id)
                st.text("")
                st.text("")
                st.subheader(f"Learn more about our recommendations! :grapes:")
                with st.spinner("Wait uh....."):
                    assistant.description = "You are a helpful robot wine sommelier. You will only reply to wine related issues, else reply that you do not know."
                    client.beta.assistants.update(assistant.id, description=assistant.description)
                    thread = client.beta.threads.create()
                    message = client.beta.threads.messages.create(thread_id=thread.id,role="user",
                                                                content= f"please give me a brief description of each wine mentioned here {recommend} as well as the food pairing that goes with it in a short and concise manner. Food pairings to be related to {user_cuisine} cuisine context.")
                    run = client.beta.threads.runs.create_and_poll(thread_id=thread.id,assistant_id=assistant.id,instructions="Replies should be in point form. Two points under each wine recommendation. These two points are 'description' and 'food pairing'. Food pairing should just be itemised response and concise.  ")
                    if run.status == 'completed': 
                        messages = client.beta.threads.messages.list(
                            thread_id=thread.id
                        )
                        st.markdown(messages.data[0].content[0].text.value)
                    else:
                        print(run.status)
                st.success("Hope you enjoy our recommendation!")
                st.text("")

        elif user_id is not None:
                st.write(f"Please provide a valid user ID. If you are a new user, please return to the main page.")


with tab2:
    st.title("💬 Ask me anything about wine!")
    st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

    with st.status("Chat with our RoboSommelier", expanded=True) as status_box:

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            thread = client.beta.threads.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"Answer this question '{prompt}' as you would to a customer in a supermarket as a wine sommelier, using only the information about wine in the documents provided. Keep your response to 50 words. If the question is not about wine, say that you do not know. Do not mention about the existence of the files uploaded.",
                                }
                            ]
                            )
            
            run = client.beta.threads.runs.create(thread_id=thread.id,assistant_id=assistant_id,)
            status_box.update(label="Thank you for being patient with me", state="running")
        
            while run.status != "completed":
                time.sleep(3)
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

            messages = client.beta.threads.messages.list(thread_id=thread.id)
            status_box.update(label="Hope this is helpful!", state="complete", expanded=True)

            pattern = r'(【\d+†source】|【\d+:\d+†source】)'
            cleaned_text = re.sub(pattern, '', messages.data[0].content[0].text.value)

            msg = cleaned_text
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
