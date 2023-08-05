import streamlit as st
import requests
import datetime as dt




def main():
    #basic configuration of dashboard
    st.set_page_config(
        page_title="Interactive Dashboard",
        page_icon=":)",
        layout="wide",
    )
   # page title
    st.title("Intelligent Network expansion")
    st.sidebar.write("4G traffic Forcast")
    st.sidebar.button("daily basis")
    # top level filter 
    col1, col2 = st.columns(2)
    col1.write("User parameters")
    col2.write("prediction result")
if __name__ == "__main__":   
    main()

