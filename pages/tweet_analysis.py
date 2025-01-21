import streamlit as st
import os
import tweepy
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TWITTER_API_KEY"] = os.getenv("TWITTER_API_KEY")
os.environ["TWITTER_API_SECRET"] = os.getenv("TWITTER_API_SECRET")
os.environ["TWITTER_ACCESS_TOKEN"] = os.getenv("TWITTER_ACCESS_TOKEN")
os.environ["TWITTER_ACCESS_TOKEN_SECRET"] = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

# Twitter API setup
auth = tweepy.OAuthHandler(
    os.environ["TWITTER_API_KEY"], os.environ["TWITTER_API_SECRET"]
)
auth.set_access_token(
    os.environ["TWITTER_ACCESS_TOKEN"], os.environ["TWITTER_ACCESS_TOKEN_SECRET"]
)
api = tweepy.API(auth)

# Groq LLM setup
groq_llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=os.environ["GROQ_API_KEY"])

# Prompt Template (for single tweet analysis)
single_tweet_prompt = PromptTemplate(
    input_variables=["tweet"],
    template="""
    You are an expert in analyzing customer feedback. Analyze the following tweet and provide:

    1. Identify the main topic/subject of the tweet.
    2. Determine the sentiment expressed in the tweet (positive, negative, or neutral).
    3. Extract any specific pros or cons mentioned.
    4. If there is a comparison with other brands, highlight it.
    5. Identify the dominant emotion expressed in the tweet (choose from: happy, sad, angry, frustrated, excited, sarcastic, or neutral).
    6. Provide a concise summary of the tweet's key message.
    7. Based on the sentiment and emotion, generate a short and clear response as if you are a company's Twitter bot.

    Tweet:
    {tweet}

    Output format:
    Topic: ...
    Emotion: ...
    Bot Response: ...
    Sentiment: ...
    Pros: ...
    Cons: ...
    Comparisons: ...
    Summary: ...
    """,
)
single_tweet_chain = single_tweet_prompt | groq_llm


def get_tweet_by_id(tweet_id):
    try:
        tweet = api.get_status(tweet_id, tweet_mode="extended")
        return tweet.full_text
    except tweepy.TweepyException as e:
        st.error(f"Error fetching tweet: {e}")
        return None


def analyze_single_tweet(tweet_text):
    try:
        analysis_result = single_tweet_chain.invoke({"tweet": tweet_text})
        return analysis_result
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        return "An error occurred during analysis."


# Streamlit app
st.title("Single Tweet Analyzer")
st.write("Analyze a single tweet or provide a tweet link.")

input_type = st.radio("Input Type:", ("Tweet Text", "Tweet Link"))

if input_type == "Tweet Text":
    tweet_input = st.text_area("Enter the tweet text:")
    if st.button("Analyze Tweet"):
        if tweet_input.strip():
            with st.spinner("Analyzing tweet..."):
                analysis = analyze_single_tweet(tweet_input)
                st.write(analysis.content)
        else:
            st.warning("Please enter tweet text.")

elif input_type == "Tweet Link":
    tweet_link = st.text_input("Enter the tweet link:")
    if st.button("Analyze Tweet"):
        if tweet_link.strip():
            try:
                tweet_id = int(tweet_link.split("/")[-1])  # Extract tweet ID from link
                with st.spinner("Fetching and analyzing tweet..."):
                    tweet_text = get_tweet_by_id(tweet_id)
                    if tweet_text:
                        analysis = analyze_single_tweet(tweet_text)
                        st.write(analysis)
                    else:
                        st.error("Could not retrieve tweet from the provided link.")
            except ValueError:
                st.error("Invalid tweet link format.")
            except IndexError:
                st.error("Invalid tweet link format.")
        else:
            st.warning("Please enter a tweet link.")
