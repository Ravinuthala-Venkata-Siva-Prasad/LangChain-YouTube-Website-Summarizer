import validators
import streamlit as st
from urllib.parse import urlparse
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_groq import ChatGroq

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URLs")

with st.sidebar:
    groq_api_key = st.text_input("GROQ API KEY", value="", type="password")

generic_url = st.text_input("URL", placeholder="Enter YouTube or Website URL")

prompt_template = """
Provide a summary of the following content in approximately 300 words:

Content: {text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):

    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the API key and URL to get started.")

    else:
        try:
            # Add HTTPS if missing
            generic_url = generic_url.strip()

            if not generic_url.startswith(("http://", "https://")):
                generic_url = "https://" + generic_url

            if not validators.url(generic_url):
                st.error("Please enter a valid YouTube or Website URL.")

            else:
                with st.spinner("Generating Summary..."):

                    llm = ChatGroq(
                        model="openai/gpt-oss-20b",
                        api_key=groq_api_key.strip()
                    )

                    # Identify YouTube URL
                    hostname = (urlparse(generic_url).hostname or "").lower()

                    if hostname in ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"):
                        loader = YoutubeLoader.from_youtube_url(
                            generic_url,
                            add_video_info=False
                        )

                    else:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=True,
                            headers={"User-Agent": "Mozilla/5.0"}
                        )

                    docs = loader.load()

                    if not docs:
                        st.error("No content found at the provided URL.")

                    else:
                        chain = load_summarize_chain(
                            llm,
                            chain_type="stuff",
                            prompt=prompt
                        )

                        output_summary = chain.invoke({
                            "input_documents": docs
                        })

                        st.subheader("Generated Summary")
                        st.success(output_summary["output_text"])

        except Exception as e:
            st.exception(e)