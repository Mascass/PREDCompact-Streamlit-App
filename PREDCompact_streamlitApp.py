import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


st.write("Hello World")
### Uploading file
uploaded_file = st.file_uploader("Upload file", type=["xlsm", "xlsx", "csv"])
if uploaded_file is not None:
    try:
        data_raw = pd.read_excel(uploaded_file,sheet_name='raw')

        ### Cleaning data to extract ingredients
        data_cleaned = data_raw['Ingrédients'].str.split(',').explode().reset_index()
        data_cleaned['Ingredients_number'] = data_cleaned.groupby('index').cumcount()
        data_cleaned


        ### Name standardisation
        for element in data_cleaned['Ingrédients']:
            if ("SYNTHETIC FLUORPHLOGOPITE" in element) & (element != "SYNTHETIC FLUORPHLOGOPITE"):
                print(element)
                data_cleaned['Ingrédients'].replace(element,"SYNTHETIC FLUORPHLOGOPITE", inplace=True)
        data_cleaned


        ### Dataframe of all the ingredients per product
        data_cleaned_df = data_cleaned.pivot(
            index='index',
            columns='Ingredients_number',
            values='Ingrédients'
        )
        data_cleaned_df.insert(0,"Product",data_raw["Nom"])
        pd.set_option('display.max_columns', None)


        ### Plotting for brands and group
        data_raw.rename(columns={'Groupe(s) / Société(s) cosmétique(s)':'Group'},inplace=True)

        # Brands
        brand_counts = data_raw['Marque'].value_counts()
        st.bar_chart(data=brand_counts,horizontal=True,sort="count",y="count")


        # Groups
        brand_counts = data_raw['Group'].value_counts()

        st.bar_chart(data=brand_counts,horizontal=True,sort="count")

        # Get the full list of unique ingredients
        ingr_list = []
        for i in data_cleaned_df.index:
            for j in data_cleaned_df:
                if j=="Product":
                    continue
                if not data_cleaned_df[j].iloc[i] in ingr_list:
                    ingr_list.append(data_cleaned_df[j].iloc[i])


        ingredient_matrix = pd.DataFrame(0, index=data_cleaned_df["Product"], columns=ingr_list)
        ingredient_matrix = ingredient_matrix.reset_index().rename(columns={"index": "Product"})


        count=0
        for i, row in ingredient_matrix.iterrows():
            for ing in ingr_list:
                if ing in data_cleaned_df.loc[count].values:
                    ingredient_matrix.at[i, ing] = 1
            count=count+1

        ingredient_occurence_matrix = pd.DataFrame(0, index=ingr_list, columns=ingr_list)
        ingredient_occurence_matrix = ingredient_occurence_matrix.reset_index().rename(columns={"index": "Ingredients"})

        # Cooccurence matrix
        ingredients_only = ingredient_matrix.drop(columns=["Product"])
        M = ingredients_only.values.astype(int)
        cooccurrence = M.T @ M
        ingredient_occurence_matrix = pd.DataFrame(
            cooccurrence,
            index=ingredients_only.columns,
            columns=ingredients_only.columns
        )
        ingredient_occurence_matrix.columns = ingredient_occurence_matrix.columns.fillna("N/A")
        ingredient_occurence_matrix.index = ingredient_occurence_matrix.index.fillna("N/A")

        st.dataframe(ingredient_occurence_matrix)

    except Exception as e:
        st.error(f"Failed to read file: {e}. Make sure it contains a sheet named 'raw'.")
