import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules

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

                

        
        st.subheader("Products and their ingredients")
        st.write("A datagram where each product from the file has a list of their ingredients displayed.")
        st.dataframe(data_cleaned_df)


        ### Plotting for brands and group
        data_raw.rename(columns={'Groupe(s) / Société(s) cosmétique(s)':'Group'},inplace=True)

        # Brands
        st.subheader("Products and their Brands")
        st.write("A chart to visualize what brands are represented by amount of products.")


        brands_groups = data_raw.groupby('Group')['Marque'].value_counts().reset_index()
        cmap = load_cmap("Acadia")
        category_codes, unique_categories = pd.factorize(brands_groups['Group'])
        colors = sns.color_palette('viridis',n_colors=len(brands_groups['Marque']))

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
            text_kwargs={"color": "black"},
            pad=True,
            ax=ax,
        )
        st.pyplot(fig)


        # Group tree map
        groups = data_raw.groupby('Group')['Group'].value_counts().reset_index()
        cmap = load_cmap("Acadia")
        category_codes, unique_categories = pd.factorize(groups['Group'])
        colors = sns.color_palette('viridis',n_colors=len(groups['Group']))

        # customize the labels
        labels = [
            f"{name}  {value}"
            for name, value in zip(groups['Group'],groups['count'] )
        ]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_axis_off()
        squarify.plot(
            sizes=groups['count'],
            label=labels,
            color=colors,
            text_kwargs={"color": "black"},
            pad=True,
            ax=ax,
        )
        st.pyplot(fig)

        st.dataframe(brands_groups)

        # Groups
        st.write("Products and their Groups")
        st.write("A chart to visualize what groups are represented by amount of products.")
        brand_counts = data_raw['Group'].value_counts()

        fig, ax = plt.subplots(figsize=(10, 12))
        ax.barh(
            brand_counts.index,
            brand_counts.values
        )
        ax.set_xlabel("Sum of groups")
        ax.set_ymargin(0)
        
        st.pyplot(fig)

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
        ingredient_matrix = ingredient_matrix.drop(columns="+/- (MAY CONTAIN)")


        # count=0
        # for i, row in ingredient_matrix.iterrows():
        #     for ing in ingr_list:
        #         if ing in data_cleaned_df.loc[count].values:
        #             ingredient_matrix.at[i, ing] = 1
        #     count=count+1

        # ingredient_occurence_matrix = pd.DataFrame(0, index=ingr_list, columns=ingr_list)
        # ingredient_occurence_matrix = ingredient_occurence_matrix.reset_index().rename(columns={"index": "Ingredients"})

        # Cooccurence matrix
    #     ingredients_only = ingredient_matrix.drop(columns=["Product"])
    #     M = ingredients_only.values.astype(int)
    #     cooccurrence = M.T @ M
    #     ingredient_occurence_matrix = pd.DataFrame(
    #         cooccurrence,
    #         index=ingredients_only.columns,
    #         columns=ingredients_only.columns
    #     )
    #     ingredient_occurence_matrix.columns = ingredient_occurence_matrix.columns.fillna("N/A")
    #     ingredient_occurence_matrix.index = ingredient_occurence_matrix.index.fillna("N/A")

        
    #     st.header("Coocurrence Matrix")
    #     st.markdown( '''Coocurrence Matrix = M.T @ M   
    #     where M is a Matrix where each row is a product and each column an ingredient values are :  
    #     1 ingredient composes this product or 0 if not. ''')
    #     st.dataframe(ingredient_occurence_matrix)

    
        # Association rules based on data_cleaned_df
        data_cleaned_df = data_cleaned_df.drop(columns="Product")
        st.dataframe(data_cleaned_df)

        # Find the most used ingredients by sorting by number of occurences
        transaction = []
        for i in range(0, data_cleaned_df.shape[0]):
            for j in range(0, data_cleaned_df.shape[1]):
                transaction.append(data_cleaned_df.values[i,j])
        transaction = np.array(transaction)
        df = pd.DataFrame(transaction, columns=["items"]) 
        df["incident_count"] = 1
        indexNames = df[df['items'] == "nan" ].index
        df.drop(indexNames , inplace=True)
        df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()

        # Association rules
        transaction = []
        for i in range(data_cleaned_df.shape[0]):
            transaction.append([str(data_cleaned_df.values[i,j]) for j in range(data_cleaned_df.shape[1])])
            
        transaction = np.array(transaction)
        print(transaction)

        te = TransactionEncoder()
        te_ary = te.fit(transaction).transform(transaction)
        dataset = pd.DataFrame(te_ary, columns=te.columns_)
        # Descending order of the most used ingredients
        first50 = df_table["items"].head(50).values
        dataset = dataset.loc[:,first50]

        # Convert dataset into 1-0 encoding
        def encode_units(x):
            if x == False:
                return 0 
            if x == True:
                return 1
            
        dataset = dataset.applymap(encode_units)
        st.dataframe(dataset)


        frequent_itemsets = apriori(dataset, min_support=0.2, use_colnames=True)
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        st.write("Number of rules : ",frequent_itemsets.shape[0])
        frequent_itemsets[(frequent_itemsets['length'] > 2)]

    except Exception as e:
        st.error(f"Failed to read file: {e}")



