import streamlit  as  st
from langchain_ollama import ChatOllama
import langchain_core.output_parsers as StrOutputParsers


from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)



# Custom CSS styling
st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)



st.title("DEEPSEEK")
st.caption("sample deepseek ")

###create sidebar
with st.sidebar:
    st.header("MODELS")


    select_model = st.selectbox(
        "Choose a model",
        ["deepseek-r1:1.5b", "deepseek-r2:7b"],
        index=0
    )


    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai) | [langchain](https://python.langchian.com/)")



    #####initoate chat

    llm_engine = ChatOllama(
        model = select_model,
        base_url = "http://localhost:11434",
        temperature = 0.3,
        
    )
    ##how llm model should react
    
    system_prompt = SystemMessagePromptTemplate.from_template(
        "you are an ai coding assistant. Provide concise,correct solutions"
        "with strategic print statements for debugging,always respond in english"
    )

    #seesion state management

    if "message_log" not in st.session_state:
        st.session_state.message_log =[{"role":"ai" ,"content":"How can i helpe you"}]


##chat container
chat_container = st.container()

##display the chat message

with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
             st.markdown(message["content"])




#chat input and processing

user_query = st.chat_input("type your coding question here")


def genereate_ai_response(prompt_chain):

    processing_pipeline = prompt_chain | llm_engine | StrOutputParsers()
    return processing_pipeline.invoke({})   


def build_prompt_chain():
     prompt_sequence = [system_prompt]

     for msg in st.session_state.message_log:

        if msg["role"] == "user": 
            prompt_sequence.append(
                HumanMessagePromptTemplate.from_template(msg["content"]).text

            )

        elif msg["role"] == "ai":

            prompt_sequence.append(
                AIMessagePromptTemplate.from_template(msg["content"])
           
           
            )

        return ChatPromptTemplate.from_template(prompt_sequence)    

            
if user_query:

    #Add user message to log

    st.session_state.message_log.append({"role":"user", "content":user_query})

    #generate ai response
    print(user_query)
    with st.spinner("processing....."):
          prompt_chain = build_prompt_chain()
          ai_response =genereate_ai_response(prompt_chain)

          print(ai_response)
    st.session_state.message_log.append({"role":"ai","content":ai_response})
    #return to update chat display
    st.rerun()

 