import streamlit as st

col1, col2, col3 = st.columns([1, 2, 1])

with col2 :
    st.image("image.png",  use_container_width=True)
    st.title("FeedBot")
    st.header("Turning Customer Feedback into Actionable Insights!")
