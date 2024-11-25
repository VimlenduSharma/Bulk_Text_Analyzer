#Streamlit_frontend
#app.py

import streamlit as st
from utils import (
    preprocess_text,
    split_text_into_chunks,
    create_vector_store_from_text_chunks,
    get_conversation_chain
)
import xmltodict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px

#Set_Streamlit_page_configuration
st.set_page_config(
    page_title="Bulk Text Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Title_and_Description
st.title("📊 Bulk Text Analyzer")
st.markdown("""
Welcome to the **Bulk Text Analyzer**! Upload your XML files to extract key terms and visualize word frequencies.
""")

#Sidebar_for_API_Key_and_File_Upload
st.sidebar.header("Settings")

#Input_for_OpenAI_API_Key
openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password",
    help="You can obtain an API key from [OpenAI](https://platform.openai.com/account/api-keys)."
)

#File_Uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload XML File",
    type=["xml"],
    accept_multiple_files=False,
    help="Upload an XML file containing the text data you want to analyze."
)

#_Function_to_parse_XML
def parse_xml(file):
    try:
        data_dict = xmltodict.parse(file.read())
        text = data_dict.get('root', {}).get('document', {}).get('content', '')
        return text
    except Exception as e:
        st.error(f"Error parsing XML: {e}")
        return ""

# Analyze Button
if uploaded_file and openai_api_key:
    if st.sidebar.button("Analyze"):
        with st.spinner("Processing..."):
            # Parse XML
            raw_text = parse_xml(uploaded_file)
            if not raw_text:
                st.error("No text found in the uploaded XML.")
                st.stop()
            
            # Preprocess Text
            preprocessed_text = preprocess_text(raw_text)
            
            # Split into Chunks
            text_chunks = split_text_into_chunks(preprocessed_text)
            
            # Create Vector Store
            vector_store = create_vector_store_from_text_chunks(text_chunks, openai_api_key)
            
            # Initialize Conversation Chain
            conversation_chain = get_conversation_chain(vector_store, openai_api_key)
            
            # Extract Key Terms using OpenAI
            query = "Extract key related words from the text."
            response = conversation_chain.run({"question": query, "chat_history": []})
            key_terms = response.get('answer', '')
            
            # Tokenize and Count Frequencies
            tokens = key_terms.split()
            word_counts = Counter(tokens)
            common_words = word_counts.most_common(20)
            
            # Prepare DataFrame for Visualization
            df = pd.DataFrame(common_words, columns=["Word", "Frequency"])
            
            # Display Results
            st.success("Analysis Complete!")
            st.subheader("Key Terms")
            st.write(key_terms)
            
            st.subheader("Word Frequency")
            fig = px.bar(df, x='Frequency', y='Word', orientation='h',
                         title='Top 20 Word Frequencies',
                         color='Frequency',
                         color_continuous_scale='Viridis')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            # Bar Chart using Seaborn
            plt.figure(figsize=(10,6))
            sns.barplot(x="Frequency", y="Word", data=df, palette="viridis")
            plt.title("Top 20 Word Frequencies")
            plt.xlabel("Frequency")
            plt.ylabel("Word")
            st.pyplot(plt)
else:
    st.info("Please upload an XML file and enter your OpenAI API Key in the sidebar.")
