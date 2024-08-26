import streamlit as st
import streamlit
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.api import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score,ConfusionMatrixDisplay,roc_curve,auc
from imblearn.over_sampling import SMOTE

global p_value, processed_df
st.set_page_config(layout="wide")

file_upload = st.file_uploader("upload csv file")
if file_upload:
    csv_data = pd.read_csv(file_upload)
    data_df = csv_data
    st.dataframe(data_df)
    if st.checkbox("if NA present drop NA"):
        data_df.dropna(inplace=True)
        data_df.reset_index(inplace=True)
    user_output_col = st.text_input("Enter your output column from data frame")
    basic_eda, desc_eda, feature_selection, ml_algo = st.tabs(["Basic EDA", "Description df", "Feature Selection", "ML Algo"])
    with basic_eda:
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

    with desc_eda:
        from streamlit import runtime
        if runtime.exists():
            if st.checkbox("handle null values if has"):
                if st.checkbox("check data has null values:"):
                    null_cols = data_df.isnull().any().any()
                    if null_cols:
                        st.dataframe({'columns':data_df.isnull().sum().index, 'values':data_df.isnull().sum().values})
                        if st.checkbox("do you want to delete null values?"):
                            null_cols = st.text_input("which columns want to delete provide with comma seperate")
                            if null_cols:
                                data_df.dropna(subset=[each for each in null_cols.strip().split(',')], inplace=True)
                        if st.checkbox("fill null values with mean?"):
                            null_cols = st.text_input("which columns want to fill provide with comma seperate")
                            if null_cols:
                                for each in null_cols.strip().split(','):
                                    mean_val = data_df[each].mean()[0]
                                    data_df[each].fillna(mean_val)
                        if st.checkbox("fill null values with mode?"):
                            null_cols = st.text_input("which columns want to fill provide with comma seperate")
                            if null_cols:
                                for each in null_cols.strip().split(','):
                                    mean_val = data_df[each].mode()[0]
                                    data_df[each].fillna(mean_val)
                        if st.checkbox("fill null values with previous one?"):
                            data_df.fillna(method='pad')
                        if st.checkbox("fill null values with next one?"):
                            data_df.fillna(method='bfill')
                        if st.checkbox("fill coulmn values with specific value"):
                            null_cols = st.text_input("enter each values with comma separated Ex a=10,b=20")
                            if null_cols:
                                for each in null_cols.strip().split(','):
                                    col_name = each.split('=')[0]
                                    value = each.split('=')[1]
                                    data_df[col_name].fillna(value)

                    else:
                        st.markdown("Data frame doesnot have null values")

    with feature_selection:
        check_feat_sele = st.radio('select feature selection method', ['None', 'P-value', 'VIF', 'Dropping constants'], horizontal=True)
        if check_feat_sele == 'P-value':
            alpha_value = float(st.text_input('enter alpha value'))
            new_df = pd.get_dummies(data_df)
            X = new_df.drop(user_output_col, axis=1)
            y = new_df[user_output_col]
            # todo : select random state and test_size custom in feature
            X_train, X_test,y_train,y_test = train_test_split(X, y, random_state=123, test_size=0.2)
            model = OLS(y_train.astype(float),X_train.astype(float)).fit()
            d = {}
            for i in X_train.columns.tolist():
                if model.pvalues[i] > alpha_value:
                    d[f'{i}'] = model.pvalues[i]
                print(d)
            high_p_value_cols = list(d.keys())
            processed_df = data_df.drop(high_p_value_cols, axis=1)
            p_value = True
        if check_feat_sele == 'VIF':
            X = data_df.drop(user_output_col, axis=1)
            y = data_df[user_output_col]
            vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif_dataframe = pd.DataFrame(vif, index=[X.columns], columns=['VIF'])
            vif_custom = eval(st.text_input("vif value"))
            print(vif_dataframe)
            selected_columns = vif_dataframe[vif_dataframe['VIF'] < vif_custom]
            selected_cols_list = [i[0] for i in selected_columns.index]
            new_vif_data_df = data_df[selected_cols_list]
            print(new_vif_data_df.head())
            ss = StandardScaler()
            for each in num_cols:
                data_df[each] = ss.fit_transform(data_df[[each]])
            processed_df = pd.DataFrame(data_df, columns=selected_cols_list)
            processed_df[user_output_col] = data_df[user_output_col]
    with ml_algo:
        selected_ml = st.radio("select required ml model",['None', 'Linear model', 'Logistic model','KNN model', 'Naivebayes model','DecisionTree model'])
        # todo: convert output col to numeric if it is categorical type
        X = processed_df.drop(user_output_col, axis=1)
        y = processed_df[user_output_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if selected_ml == 'None':
            pass
        elif selected_ml == 'Linear model':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            print(predictions)
        elif selected_ml == 'Logistic model':
            from sklearn.linear_model import LogisticRegression
            lg_model = LogisticRegression(solver='saga', max_iter=1000, random_state=43)
            lg_model.fit(X_train, y_train)
            y_pred_log = lg_model.predict(X_test)
        elif selected_ml == 'KNN model':
            from sklearn.neighbors import KNeighborsClassifier
            k = 5
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            knn_classifier.fit(X_train, y_train)
            y_pred = knn_classifier.predict(X_train)
        elif selected_ml == 'Naivebayes model':
            from sklearn.naive_bayes import GaussianNB
            nb_classifier = GaussianNB()
            nb_classifier.fit(X_train, y_train)
            y_pred = nb_classifier.predict(X_test)
        elif selected_ml == 'DecisionTree model':
            from sklearn.tree import DecisionTreeClassifier
            dt_clf = DecisionTreeClassifier()
            dt_clf.fit(X_train, y_train)
            y_pred_dt = dt_clf.predict(X_test)
