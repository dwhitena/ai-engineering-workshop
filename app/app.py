import time
import re
import shutil
import os
import urllib

import html2text
import predictionguard as pg
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import PredictionGuard
import streamlit as st
from sentence_transformers import SentenceTransformer
import lancedb
from lancedb.embeddings import with_embeddings
import pandas as pd


#--------------------------#
# Prompt templates         #
#--------------------------#

demo_formatter_template = """\nUser: {user}
Assistant: {assistant}\n"""
demo_prompt = PromptTemplate(
    input_variables=["user", "assistant"],
    template=demo_formatter_template,
)

category_template = """### Instruction:
Read the below input and determine if it is a request to generate computer code? Respond "yes" or "no".

### Input:
{query}

### Response:
"""

category_prompt = PromptTemplate(
    input_variables=["query"],
    template=category_template
)

qa_template = """### Instruction:
Read the context below and respond with an answer to the question. If the question cannot be answered based on the context alone or the context does not explicitly say the answer to the question, write "Sorry I had trouble answering this question, based on the information I found."

### Input:
Context: {context}

Question: {query}

### Response:
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=qa_template
)

chat_template = """### Instruction:
You are a friendly and clever AI assistant. Respond to the latest human message in the input conversation below.

### Input:
{context}
Human: {query}
AI:

### Response:
"""

chat_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=chat_template
)

code_template = """### Instruction:
You are a code generation assistant. Respond with a code snippet and any explanation requested in the below input.

### Input:
{query}

### Response:
"""

code_prompt = PromptTemplate(
    input_variables=["query"],
    template=code_template
)


#-------------------------#
#    Vector search        #
#-------------------------#

# Embeddings setup
name="all-MiniLM-L12-v2"
model = SentenceTransformer(name)

def embed_batch(batch):
    return [model.encode(sentence) for sentence in batch]

def embed(sentence):
    return model.encode(sentence)

# LanceDB setup
if os.path.exists(".lancedb"):
    shutil.rmtree(".lancedb")
os.mkdir(".lancedb")
uri = ".lancedb"
db = lancedb.connect(uri)

def vector_search_urls(urls, query, sessionid):

    for url in urls:

        # Let's get the html off of a website.
        fp = urllib.request.urlopen(url)
        mybytes = fp.read()
        html = mybytes.decode("utf8")
        fp.close()

        # And convert it to text.
        h = html2text.HTML2Text()
        h.ignore_links = True
        text = h.handle(html)

        # Chunk the text into smaller pieces for injection into LLM prompts.
        text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        docs = text_splitter.split_text(text)
        docs = [x.replace('#', '-') for x in docs]

        # Create a dataframe with the chunk ids and chunks
        metadata = []
        for i in range(len(docs)):
            metadata.append([
                i,
                docs[i],
                url
            ])
        doc_df = pd.DataFrame(metadata, columns=["chunk", "text", "url"])
        
        # Embed the documents
        data = with_embeddings(embed_batch, doc_df)

        # Create the table if there isn't one.
        if sessionid not in db.table_names():
            db.create_table(sessionid, data=data)
        else:
            table = db.open_table(sessionid)
            table.add(data=data)

    # Perform the query
    table = db.open_table(sessionid)
    results = table.search(embed(query)).limit(1).to_df()
    results = results[results['_distance'] < 1.0]
    if len(results) == 0:
        doc_use = ""
    else:
        doc_use = results['text'].values[0]

    # Clean up
    db.drop_table(sessionid)

    return doc_use

#-------------------------#
#     Info Agent          #
#-------------------------#

tools = load_tools(["serpapi"], llm=PredictionGuard(model="Nous-Hermes-Llama2-13B"))
agent = initialize_agent(
    tools, 
    PredictionGuard(model="Nous-Hermes-Llama2-13B"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    max_execution_time=30)

#-------------------------#
#   Helper functions      #
#-------------------------#

def find_urls(text):
    return re.findall(r'(https?://[^\s]+)', text)

# QuestionID provides some help in determining if a sentence is a question.
class QuestionID:
    """
        QuestionID has the actual logic used to determine if sentence is a question
    """
    def padCharacter(self, character: str, sentence: str):
        if character in sentence:
            position = sentence.index(character)
            if position > 0 and position < len(sentence):

                # Check for existing white space before the special character.
                if (sentence[position - 1]) != " ":
                    sentence = sentence.replace(character, (" " + character))

        return sentence

    def predict(self, sentence: str):
        questionStarters = [
            "which", "wont", "cant", "isnt", "arent", "is", "do", "does",
            "will", "can"
        ]
        questionElements = [
            "who", "what", "when", "where", "why", "how", "sup", "?"
        ]

        sentence = sentence.lower()
        sentence = sentence.replace("\'", "")
        sentence = self.padCharacter('?', sentence)
        splitWords = sentence.split()

        if any(word == splitWords[0] for word in questionStarters) or any(
                word in splitWords for word in questionElements):
            return True
        else:
            return False


#---------------------#
# Streamlit config    #
#---------------------#

#st.set_page_config(layout="wide")

# Hide the hamburger menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


#--------------------------#
# Streamlit sidebar        #
#--------------------------#

st.sidebar.title("Super Chat 🚀")
st.sidebar.markdown(
    "This app provides a chat interface driven by various generative AI models and "
    "augmented (via information retrieval and agentic processing)."
)
url_text = st.sidebar.text_area(
    "Enter one or more urls for reference information (separated by a comma):", 
    "", height=100)
if len(url_text) > 0:
    urls = url_text.split(",")
else:
    urls = []


#--------------------------#
# Streamlit app            #
#--------------------------#

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # process the context
        examples = []
        turn = "user"
        example = {}
        for m in st.session_state.messages:
            latest_message = m["content"]
            example[turn] = m["content"]
            if turn == "user":
                turn = "assistant"
            else:
                turn = "user"
                examples.append(example)
                example = {}
        if len(example) > 4:
            examples = examples[-4:]

        # Determine what kind of message this is.
        with st.spinner("Trying to figure out what you are wanting..."):
            result = pg.Completion.create(
                model="WizardCoder",
                prompt=category_prompt.format(query=latest_message),
                output={
                    "type": "categorical",
                    "categories": ["yes", "no"]
                }
            )

        # configure out chain
        code = result['choices'][0]['output']
        qIDModel = QuestionID()
        question = qIDModel.predict(latest_message)

        if code == "no" and question:

            # if there are urls, let's embed them as a primary data source.
            if len(urls) > 0:
                with st.spinner("Performing vector search..."):
                    info_context = vector_search_urls(urls, latest_message, "assistant")
            else:
                info_context = ""

            # Handle the informational request.
            if info_context != "":
                with st.spinner("Generating a RAG result..."):
                    result = pg.Completion.create(
                        model="Nous-Hermes-Llama2-13B",
                        prompt=qa_prompt.format(context=info_context, query=latest_message)
                    )
                    completion = result['choices'][0]['text'].split('#')[0].strip()
            
            # Otherwise try an agentic approach.
            else:
                with st.spinner("Trying to find an answer with an agent..."):
                    try:
                        completion = agent.run(latest_message)
                    except:
                        completion = "Sorry, I didn't find an answer. Could you rephrase the question?" 
                    if "Agent stopped" in completion:
                        completion = "Sorry, I didn't find an answer. Could you rephrase the question?"

        elif code == "yes":

            # Handle the code generation request.
            with st.spinner("Generating code..."):
                result = pg.Completion.create(
                    model="WizardCoder",
                    prompt=code_prompt.format(query=latest_message),
                    max_tokens=500
                )
                completion = result['choices'][0]['text']

        else:

            # contruct prompt
            few_shot_prompt = FewShotPromptTemplate(
                examples=examples,
                example_prompt=demo_prompt,
                example_separator="",
                prefix="The following is a conversation between an AI assistant and a human user. The assistant is helpful, creative, clever, and very friendly.\n",
                suffix="\nHuman: {human}\nAssistant: ",
                input_variables=["human"],
            )

            prompt = few_shot_prompt.format(human=latest_message)

            # generate response
            with st.spinner("Generating chat response..."):
                result = pg.Completion.create(
                    model="Nous-Hermes-Llama2-13B",
                    prompt=prompt,
                )
                completion = result['choices'][0]['text']

        # Print out the response.
        completion = completion.split("Human:")[0].strip()
        completion = completion.split("H:")[0].strip()
        completion = completion.split('#')[0].strip()
        for token in completion.split(" "):
            full_response += " " + token
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.075)
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})