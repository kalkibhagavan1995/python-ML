import streamlit as st
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


st.set_page_config(layout="wide")

file_upload = st.file_uploader("upload csv file")
if file_upload:
    csv_data = pd.read_csv(file_upload)

    data_df = csv_data
    st.dataframe(data_df)
    user_output_col = st.text_input("Enter your output column from data frame")

    cat_cols = data_df.select_dtypes(include="object").columns
    num_cols = data_df.select_dtypes(exclude="object").columns

    cat_col, num_col = st.columns(2)
    cat_col.title("list of categorical columns")
    for each in list(cat_cols):
        cat_col.markdown("- "+each)

    remove_unwanted_cat_cols = cat_col.text_input("type unwanted column to remove seperated with comma")
    copy_cat_cols = list(cat_cols)[:]
    if remove_unwanted_cat_cols:
        for each in remove_unwanted_cat_cols.strip().split(','):
            copy_cat_cols.remove(each)
    else:
        copy_cat_cols = list(cat_cols)[:]
    cat_col.selectbox("list of columns for plots",copy_cat_cols)
    check_cat_radio = cat_col.radio('select plot', ['None', 'bar_chart','pie_chart'], horizontal=True)
    if check_cat_radio != 'None':
        selected_cat_col = cat_col.selectbox(f"select column for {check_cat_radio}", copy_cat_cols)
    if check_cat_radio == "bar_chart":
        cat_col.title(f"bar chart for {selected_cat_col}")
        unique_keys = data_df[selected_cat_col].unique()
        val_count = []
        for i in unique_keys:
            con =  data_df[selected_cat_col] == i
            val_count.append(len(data_df[con]))
        temp_df = pd.DataFrame(zip(unique_keys,val_count), columns=['first','count'])
        plt.xlabel(selected_cat_col)
        plt.ylabel("Count")
        plt.bar('first', 'count',data=temp_df)
        cat_col.pyplot(plt)
    if check_cat_radio == 'pie_chart':
        cat_col.title(f"pie chart for {selected_cat_col}")
        col_details = data_df[selected_cat_col].value_counts()
        keys=col_details.keys()
        value=col_details.values
        plt.pie(value, labels=keys,autopct='%0.2f')
        cat_col.pyplot(plt)

    num_col.title("list of numerical columns")
    for each in list(num_cols):
        num_col.markdown("- "+each)

    if num_col.checkbox("check for description of data"):
        num_col.title("description of data")
        ans = data_df.describe()
        num_col.dataframe(ans)

    check_radio = num_col.radio('select plot', ['None','histogram_plot', 'distribution_plot'], horizontal=True)
    if check_radio != "None":
        selected_num_col = num_col.selectbox(f"select column for {check_radio}",list(num_cols))
    if check_radio == 'histogram_plot':
        selected_plot_module = num_col.selectbox("select from module", ['matplotlib', 'seaborn'])
        num_col.title(f"histogram plot for {selected_num_col}")
        if selected_plot_module == 'matplotlib':
            plt.hist(data_df[selected_num_col])
            plt.xlabel(selected_num_col)
            plt.ylabel("count")
            plt.title(f"histogram plot")
            num_col.pyplot(plt)
        elif selected_plot_module == 'seaborn':
            sns.histplot(data_df[selected_num_col])
            plt.xlabel(selected_num_col)
            plt.ylabel("count")
            plt.title(f"histogram plot")
            num_col.pyplot(plt)

    elif check_radio == "distribution_plot":
        sns.displot(data_df[selected_num_col])
        plt.xlabel(selected_num_col)
        plt.ylabel("count")
        plt.title(f"distribution plot")
        num_col.pyplot(plt)

    elif check_radio == 'None':
        pass
    check_radio_correlation = num_col.checkbox("display correlation:")
    if check_radio_correlation:
        corr_matrix = data_df.select_dtypes(exclude='object').corr()
        plt.title("heatmap plot")
        sns.heatmap(corr_matrix, cmap='RdYlGn_r', annot=True)
        num_col.pyplot(plt)
