import streamlit as st

# Define the pages
main_page = st.Page("main.py", title="Dataset", icon="🗂️")
page_2 = st.Page("EDA.py", title="EDA", icon="📊")
page_3 = st.Page("Supervised.py", title="Supervised", icon="🤖")
page_4 = st.Page("UnSupervised.py", title="UnSupervised", icon="🧩")

# Set up navigation
pg = st.navigation([main_page, page_2, page_3, page_4])

# Run the selected page
pg.run()