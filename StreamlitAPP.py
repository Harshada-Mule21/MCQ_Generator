import os
import json
import pandas as pd
import traceback 
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import read_file,get_table_data
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain

from langchain.callbacks import get_openai_callback

# loading json file
with open('D:\GenAIProjects\MCQ_Generator\Response.json','r') as file:
    RESPONSE_JSON = json.load(file)

# creating title for web application
st.title("MCQ Creater Application with Langchain 🦜")

#create a form
with st.form("user input"):
    # File upload
    uploaded_file = st.file_uploader("Upload pdf or text file")

    # Input fields

    mcq_count = st.number_input("No.of MCQs",min_value=3,max_value=50)

    subject = st.text_input("Insert subject",max_chars=20)

    tone = st.text_input("Complexity level of quetions",max_chars=20,placeholder='simple')

    button = st.form_submit_button("Create MCQs")

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner('loading.....'):
            try:
                text = read_file(uploaded_file)
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                        "text":text,
                        "number":mcq_count,
                        "subject":subject,
                        "tone":tone,
                        "response_json":json.dumps(RESPONSE_JSON)
                        }
                    )
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")
            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                print(response)
                if isinstance(response,dict):
                    # Extract the quiz data from the response
                    quiz = response.get("generated_quiz", None)
                    print(quiz)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index+1
                            st.table(df)
                            # Display the review in text area as well
                            st.text_area(label="Review",value=response["review"])
                        else:
                            st.error("Error in the table data")

                else:
                    st.write(response)