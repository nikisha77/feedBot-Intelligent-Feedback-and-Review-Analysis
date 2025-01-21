import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


# Load the tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


tokenizer, model = load_model()
set_seed(0)

# Streamlit app layout
st.title("Customer Feedback Analyzer")
st.write(
    "Analyze customer feedback for topic, sentiment, pros, cons, comparisons, and summary."
)

# Input text box for the review
review_text = st.text_area("Enter the review text:", height=200)

# Analyze button
if st.button("Analyze"):
    if review_text.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        # Format the system and user messages
        messages = [
            {
                "role": "system",
                "content": """You are an expert in analyzing customer feedback. Given the following input from the user, 
                provide the following information in the specified output format and do not generate anything 
                else by yourself except what is given by the user:

                output : 
                Topic: ...
                Sentiment: ...
                Pros: ...
                Cons: ...
                Comparisons: ...
                Summary:

                As a response, give me only the output and nothing else. If a given parameter is not present, 
                give it as N/A""",
            },
            {"role": "user", "content": review_text},
        ]

        formatted_messages = [message["content"] for message in messages]

        # Tokenize input
        model_inputs = tokenizer(
            formatted_messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to("cuda")

        input_length = model_inputs.input_ids.shape[1]

        # Generate response
        with st.spinner("Analyzing..."):
            generated_ids = model.generate(
                model_inputs.input_ids, do_sample=True, max_new_tokens=200
            )
            generated_response = tokenizer.batch_decode(
                generated_ids[:, input_length:], skip_special_tokens=True
            )[0]

        # Display the result
        st.subheader("Analysis Result")
        st.text(generated_response)
