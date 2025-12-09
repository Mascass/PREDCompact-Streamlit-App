import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import squarify  #algorithm for treemap
from pypalettes import load_cmap

st.subheader("PREDCompact app")
### Uploading file
uploaded_file = st.file_uploader("Upload file :", type=["xlsm", "xlsx", "csv"])
if uploaded_file is not None:
    try:
        data_raw = pd.read_excel(uploaded_file,sheet_name='raw')

        ### Cleaning data to extract ingredients
        try:
            data_raw = data_raw[ data_raw['Inclusao'] == 1 ]
            data_raw = data_raw.reset_index(drop=True)
        except Exception as e:
            st.error(f"Failed to find column 'Inclusao'. {e}.")  
        data_cleaned = data_raw['Ingrédients'].str.split(',').explode().reset_index()
        data_cleaned['Ingredients_number'] = data_cleaned.groupby('index').cumcount()



        ### Dataframe of all the ingredients per product
        data_cleaned_df = data_cleaned.pivot(
            index='index',
            columns='Ingredients_number',
            values='Ingrédients'
        )
        data_cleaned_df.insert(0,"Product",data_raw["Nom"])
        pd.set_option('display.max_columns', None)

        
        ### Name standardisation
        names_dict = set(['SYNTHETIC FLUORPHLOGOPITE','MICA','TALC','CELLULOSE','ZEA MAYS (CORN) STARCH',
                          'CAPRYLIC/CAPRIC TRIGLYCERIDE','ISOSTEARYL NEOPENTANOATE',
                          'CALCIUM ALUMINUM BOROSILICATE','ALUMINA','ISOEICOSANE','ALARIA ESCULENTA EXTRACT','CYCLOPENTASILOXANE'])
        for element in data_cleaned_df[0]:
            for element_replace in names_dict:
                # Replace INCI elements
                if ( element_replace in element ) & (element not in names_dict):
                    print(element)
                    data_cleaned_df[0].replace(element, element_replace, inplace=True)
            # remove May Contain
            # if ( element in "MAY CONTAIN"):
                # replace element
                

        
        st.subheader("Products and their ingredients")
        st.write("A datagram where each product from the file has a list of their ingredients displayed.")
        st.dataframe(data_cleaned_df)


        ### Plotting for brands and group
        data_raw.rename(columns={'Groupe(s) / Société(s) cosmétique(s)':'Group'},inplace=True)

        # Brands
        # st.subheader("Products and their Brands")
        # st.write("A chart to visualize what brands are represented by amount of products.")
        # brand_counts = data_raw['Marque'].value_counts()


        brands_groups = data_raw.groupby('Group')['Marque'].value_counts().reset_index()
        cmap = load_cmap("Acadia")
        category_codes, unique_categories = pd.factorize(brands_groups['Group'])
        colors = [cmap(code) for code in category_codes]

        # customize the labels
        labels = [
            f"{name} ({parent}) {value}"
            for name, parent, value in zip(brands_groups['Marque'], brands_groups['Group'],brands_groups['count'] )
        ]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_axis_off()
        squarify.plot(
            sizes=brands_groups['count'],
            label=labels,
            color=colors,
            text_kwargs={"color": "white"},
            pad=True,
            ax=ax,
        )
        st.pyplot(fig)

        st.dataframe(brands_groups)

        # Groups
        st.write("Products and their Groups")
        st.write("A chart to visualize what groups are represented by amount of products.")
        brand_counts = data_raw['Group'].value_counts()
        st.bar_chart(data=brand_counts,horizontal=True,sort="count",x_label="Sum of groups")

        # Get the full list of unique ingredients
        ingr_list = []
        for i in data_cleaned_df.index:
            for j in data_cleaned_df:
                if j=="Product":
                    continue
                if j=="NaN":
                    continue
                if not data_cleaned_df[j].iloc[i] in ingr_list:
                    ingr_list.append(data_cleaned_df[j].iloc[i])


        ingredient_matrix = pd.DataFrame(0, index=data_cleaned_df["Product"], columns=ingr_list)
        ingredient_matrix = ingredient_matrix.reset_index().rename(columns={"index": "Product"})
        ingredient_matrix = ingredient_matrix.drop(columns=np.nan)


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

        
        st.header("Coocurrence Matrix")
        st.markdown( '''Coocurrence Matrix = M.T @ M   
        where M is a Matrix where each row is a product and each column an ingredient values are :  
        1 ingredient composes this product or 0 if not. ''')
        st.dataframe(ingredient_occurence_matrix)

    except Exception as e:
        st.error(f"Failed to read file: {e}")
