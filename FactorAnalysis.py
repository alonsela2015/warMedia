from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from EDA import reverse_hebrew_text
from factor_analyzer import FactorAnalyzer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


def scaling_data(dataframe):
    """Accepts a dataframe and returns a dataframe with scaled data"""
    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(dataframe)
    scaled_data = scaler.transform(dataframe)

    return scaled_data


def rename_dataframe_from_csv(df, csv_file_path):
    # Read the CSV file
    mapping_df = pd.read_csv(csv_file_path, encoding='utf-8')

    # Apply reverse_hebrew_text to each value in the 'Questions' column
    mapping_df['Questions'] = mapping_df['Questions'].apply(reverse_hebrew_text)

    # Create a dictionary with 'Variable' as keys and 'Questions' as values
    mapping_dict = pd.Series(mapping_df.Questions.values, index=mapping_df.Variable).to_dict()

    # Rename the DataFrame columns
    df_renamed = df.rename(columns=mapping_dict)

    return df_renamed


def factor_analysis(dataframe: pd.DataFrame, df_name: str, n_components: int = 3) -> None:
    fa = FactorAnalyzer(n_factors=n_components, method='principal')  # TODO: check hyperparameters
    fa.fit_transform(dataframe)

    # Get variance of all factors
    eigen_values, vectors = fa.get_eigenvalues()
    _, prop_var, cum_var = fa.get_factor_variance()
    print(f"{df_name} prop variance:\n", prop_var)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(hspace=1)

    # First subplot for cumulative explained variance
    axs[0].plot(range(1, len(dataframe.columns) + 1), eigen_values.cumsum())
    axs[0].set_title("Cumulative scree plot")
    axs[0].set_xlabel('Factors')
    axs[0].set_ylabel('EigenValue')
    axs[0].set_xticks(range(1, len(dataframe.columns) + 1))
    axs[0].grid()

    # Second subplot for individual explained variance
    axs[1].plot(range(1, len(dataframe.columns) + 1), eigen_values)
    axs[1].set_title("Individual scree plot")
    axs[1].set_xlabel('Factors')
    axs[1].set_ylabel('EigenValue')
    axs[1].set_xticks(range(1, len(dataframe.columns) + 1))
    axs[1].grid()

    # Show the plot
    plt.show()

    # Save loadings of n_components to CSV file
    loadings = fa.loadings_
    print(f"{df_name} loadings:\n", loadings)
    loadings_df = pd.DataFrame(loadings, columns=[f"Factor{i}" for i in range(1, n_components + 1)])
    loadings_df.index = [reverse_hebrew_text(idx) for idx in dataframe.columns]

    loadings_df.to_csv(f'{LOADINGS_PATH}/{df_name}_loadings.csv', encoding='utf-8-sig')

def corr_heatmap(dataframes: list, df_names: list) -> None:
    plt.figure(figsize=(15, 10))
    corrs = pd.concat(dataframes, axis=1).corr()
    mask = np.triu(np.ones_like(corrs))
    sns.heatmap(corrs, cmap='Spectral', mask=mask, vmin=-1, vmax=1, annot=True)
    plt.title(f"Correlation between {df_names}")
    plt.show()


def hierarchical_clustering(dataframes: list, df_names: list) -> None:
    # # Performing hierarchical clustering on qol
    # cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=10, linkage='ward')
    # cluster.fit_predict(np.concatenate((qol_scaled_data, resilience_scaled_data, phone_scaled_data), axis=1))
    # print(cluster.labels_)
    #
    # Z = linkage(qol_scaled_data, method='ward')
    #
    # # Plot the dendrogram
    # plt.figure(figsize=(15, 10))
    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Sample index')
    # plt.ylabel('Distance')
    # dendrogram(Z)
    # plt.show()
    pass


def main():
    df = pd.read_excel(FILE_PATH)

    qol_df = df.loc[:, df.columns.str.contains('y')]
    resilience_df = df.loc[:, df.columns.str.contains('z')]
    media_df = df.loc[:, df.columns.str.contains('x')]
    phone_df = df.loc[:, df.columns.str.contains('w')]
    # stress_df = df.loc[:, df.columns.str.contains('u')]
    # support_df = df.loc[:, df.columns.str.contains('v')]

    # Changing columns names back to the questions
    qol_df = rename_dataframe_from_csv(qol_df, f"{maps_path}/y_mapping.csv")
    resilience_df = rename_dataframe_from_csv(resilience_df, f"{maps_path}/z_mapping.csv")
    media_df = rename_dataframe_from_csv(media_df, f"{maps_path}/x_mapping.csv")
    phone_df = rename_dataframe_from_csv(phone_df, f"{maps_path}/w_mapping.csv")
    # stress_df = rename_dataframe_from_csv(stress_df, f"{maps_path}/u_mapping.csv")
    # support_df = rename_dataframe_from_csv(support_df, f"{maps_path}/v_mapping.csv")

    # FA on resilience
    resilience_scaled_data = scaling_data(resilience_df)
    nan_rows = np.isnan(resilience_scaled_data).any(axis=1)
    resilience_scaled_data = resilience_scaled_data[~nan_rows]
    resilience_scaled_data = pd.DataFrame(resilience_scaled_data, columns=resilience_df.columns)
    resilience_scaled_data.dropna(inplace=True)

    # FA on qol
    qol_scaled_data = scaling_data(qol_df)
    qol_scaled_data = qol_scaled_data[~nan_rows]
    qol_scaled_data = pd.DataFrame(qol_scaled_data, columns=qol_df.columns)
    factor_analysis(qol_scaled_data, 'QOL', n_components=5)

    # FA on phone
    phone_scaled_data = scaling_data(phone_df)
    phone_scaled_data = phone_scaled_data[~nan_rows]
    phone_scaled_data = pd.DataFrame(phone_scaled_data, columns=phone_df.columns)
    factor_analysis(phone_scaled_data, 'phone', n_components=1)

    # FA on media
    media_scaled_data = scaling_data(media_df)
    media_scaled_data = pd.DataFrame(media_scaled_data, columns=media_df.columns)
    media_scaled_data['.םיילארשי -- םירז םיצורעהמ יתושדח עדימ ת/לבקמ ךנה המכ דע ןייצ'].fillna(7, inplace=True)
    factor_analysis(media_scaled_data, 'media', n_components=4)


if __name__ == "__main__":
    FILE_PATH = "C:/Users/bug32/Desktop/Media and Resilience/12.12.respondents_processed_data.xlsx"
    maps_path = "C:/Users/bug32/Desktop/Media and Resilience/sem_variables_maps"
    LOADINGS_PATH = "C:/Users/bug32/Desktop/Media and Resilience/loadings"
    main()
