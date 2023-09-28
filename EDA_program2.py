# IMPORTS 
import streamlit as st
import pandas as pd
from tabulate import tabulate
import numpy as np
from scipy.stats import linregress
from scipy.stats import stats
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from itertools import combinations

# SESSION
st.session_state['answer'] = ''

realans = ['', 'abc', 'edf']

if  st.session_state['answer'] in realans:
    answerStat = "correct"
elif st.session_state['answer'] not in realans:
    answerStat = "incorrect"
# PREDEF FUNCTIONS ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# function that makes interaction terms for regression table
def create_and_append_interaction_terms(data, independent_vars, interaction_terms):
    for interaction in interaction_terms:
        var1, var2 = interaction.split(":")
        interaction_name = f"{var1}:{var2}"
        data[interaction_name] = data[var1] * data[var2]
    return data

# function that performs linear regression
def perform_linear_regression(data, dependent_var, independent_vars):

    X = data[independent_vars]
    X = sm.add_constant(X)
    y = data[dependent_var]
    model = sm.OLS(y, X).fit()
    return model


def filter_data(data, selected_column, filter_option, filter_value):
    filtered_data = data
    if filter_option == "Categorical Filter":
        filtered_data = filtered_data[filtered_data[selected_column] == filter_value]
    elif filter_option == "Numerical Filter":
        if filter_value == "Above Mean":
            filtered_data = filtered_data[filtered_data[selected_column] > filtered_data[selected_column].mean()]
        elif filter_value == "Below Mean":
            filtered_data = filtered_data[filtered_data[selected_column] < filtered_data[selected_column].mean()]
        elif filter_value == "Above Median":
            filtered_data = filtered_data[filtered_data[selected_column] > filtered_data[selected_column].median()]
        elif filter_value == "Below Median":
            filtered_data = filtered_data[filtered_data[selected_column] < filtered_data[selected_column].median()]
        elif filter_value == "Within 1 STD":
            std_dev = filtered_data[selected_column].std()
            filtered_data = filtered_data[(filtered_data[selected_column] >= filtered_data[selected_column].mean() - std_dev) &
                                          (filtered_data[selected_column] <= filtered_data[selected_column].mean() + std_dev)]
        elif filter_value == "Within 2 STD":
            std_dev = filtered_data[selected_column].std()
            filtered_data = filtered_data[(filtered_data[selected_column] >= filtered_data[selected_column].mean() - 2 * std_dev) &
                                          (filtered_data[selected_column] <= filtered_data[selected_column].mean() + 2 * std_dev)]
        elif filter_value == "Within 3 STD":
            std_dev = filtered_data[selected_column].std()
            filtered_data = filtered_data[(filtered_data[selected_column] >= filtered_data[selected_column].mean() - 3 * std_dev) &
                                          (filtered_data[selected_column] <= filtered_data[selected_column].mean() + 3 * std_dev)]
        elif filter_value == "Above 25th Quantile":
            quantile_25 = filtered_data[selected_column].quantile(0.25)
            filtered_data = filtered_data[filtered_data[selected_column] > quantile_25]
        elif filter_value == "Below 25th Quantile":
            quantile_25 = filtered_data[selected_column].quantile(0.25)
            filtered_data = filtered_data[filtered_data[selected_column] < quantile_25]
        elif filter_value == "Above 75th Quantile":
            quantile_75 = filtered_data[selected_column].quantile(0.75)
            filtered_data = filtered_data[filtered_data[selected_column] > quantile_75]
        elif filter_value == "Below 75th Quantile":
            quantile_75 = filtered_data[selected_column].quantile(0.75)
            filtered_data = filtered_data[filtered_data[selected_column] < quantile_75]
        elif filter_value == "0":
            return data.loc[data[selected_column] == 0]
        elif filter_value == "1":
            return data.loc[data[selected_column] == 1]
        elif filter_value == ">0":
            return data.loc[data[selected_column] > 0]
    return filtered_data

date_columns = ["Year", "Date", "date", 
                "year","timeperiod","released_year", 
                "time", "Time", "TIME",]  # Add other similar column names here

def treat_date_columns(data):
    for column in date_columns:
        if column in data.columns:
            if column == "Year":
                # Treat "Year" as an integer representing the year
                data[column] = pd.to_datetime(data[column], format='%Y', errors="coerce")  # Convert to datetime with year only
            else:
                data[column] = pd.to_datetime(data[column], errors="coerce")  # Convert to datetime, handle errors
    return data

def identify_x_axis(data):
    for column in date_columns:
        if column in data.columns:
            return column

    return st.warning("No time series data found")  # No suitable x-axis column found

# Function to perform K-means clustering and return results
def perform_clustering(data_for_clustering, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    data_for_clustering['Cluster'] = kmeans.fit_predict(data_for_clustering)
    return data_for_clustering, kmeans

# Function to calculate silhouette score for clustering quality
def calculate_silhouette_score(data_for_clustering, kmeans):
    return silhouette_score(data_for_clustering, kmeans.labels_)

def custom_accuracy_score(data, testing_column, cluster_column):
    # Ensure both columns exist in the DataFrame
    if testing_column not in data.columns or cluster_column not in data.columns:
        raise ValueError("Both testing_column and cluster_column must be present in the DataFrame.")

    # Calculate the percentage of matching values
    matching_count = (data[testing_column] == data[cluster_column]).sum()
    total_count = len(data)
    accuracy = (matching_count / total_count)

    return accuracy

def ratio_calculator(df, col1, col2):
    col_name = f'{col1}/{col2}'
    df[col_name] = df[col1]/df[col2]

def calculate_regression_metrics(data, x_column, y_column, degrees, test_size, random_state=42):
    """
    Finds the optimal degree that minimizes both %diff in MSE and R2
    """
    results = {
        "Degree": [],
        "MSE_diff": [],
        "R2_diff": [],
        "opt": []
    }

    X = data[x_column].values.reshape(-1, 1)
    Y = data[y_column].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        reg = LinearRegression().fit(X_train_poly, Y_train)
        Y_pred_train = reg.predict(X_train_poly)
        Y_pred_test = reg.predict(X_test_poly)

        mse_train = mean_squared_error(Y_train, Y_pred_train)
        mse_test = mean_squared_error(Y_test, Y_pred_test)
        r2_train = r2_score(Y_train, Y_pred_train)
        r2_test = r2_score(Y_test, Y_pred_test)

        mse_diff = abs(mse_test - mse_train)
        r2_diff = abs(r2_test - r2_train)
        opt = (mse_diff + r2_diff - abs(mse_diff - r2_diff))/2

        results["Degree"].append(degree)
        results["MSE_diff"].append(mse_diff)
        results["R2_diff"].append(r2_diff)
        results["opt"].append(opt)

    results_df = pd.DataFrame(results)
    min_opt_degree = results_df.loc[results_df["opt"].idxmin()]["Degree"]

    return int(min_opt_degree)

# Function to create stringed integers for categorical columns
def create_stringed_integers(data):
    categorical_columns = data.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        unique_values = data[column].unique()
        value_mapping = {value: str(i) for i, value in enumerate(unique_values)}
        data[column + "_num"] = data[column].map(value_mapping)
    return data

# Function to display explanation tables for categorical columns
def display_categorical_explanations(data):
    categorical_columns = data.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        st.subheader(f"Explanation for '{column}'")
        value_mapping = {str(i): value for i, value in enumerate(data[column].unique())}
        explanation_data = pd.DataFrame(list(value_mapping.items()), columns=["Number", "Original Value"])
        st.table(explanation_data)

# MAIN APP ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    # Main Title and descriptions
    st.title("Exploratory Data Analysis and Model Building")
    st.text("Upload a CSV to get started")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Radio to toggle between different analysis types
        pages = ["Exploratory Data Analysis", "Data Visualizations", "Regression Modeling","KMeans Clustering","Categircal to Numerical Dictionary"]
        analysis_type = st.selectbox("Select Analysis Type", pages)

        # Checkbox for dropping NaN values
        st.sidebar.title("Options")
        drop_nan = st.sidebar.checkbox("Drop Nan Values")
        if drop_nan:
            data = data.dropna()

        # check box for ratio columns
        ratios = st.sidebar.checkbox("Include Ratios for Column Combos")
        if ratios:
            columns = data.select_dtypes(include=[np.number]).columns
            combinations_list = list(combinations(columns, 2))
            for col1, col2 in combinations_list:
                ratio_calculator(data, col1, col2)

        # Checkbox to create stringed integers for categorical columns
        create_stringed_int = st.sidebar.checkbox("Assign Numbers for categorical columns")
        
        # Check if the checkbox is enabled
        if create_stringed_int:
            data = create_stringed_integers(data)
          
        if analysis_type == "Categorical To Numerical Dictionary":
            display_categorical_explanations(create_stringed_integers(data))
        
        st.sidebar.divider()

        # Inseting None Column
        data.insert(loc = 0,
          column = 'None',
          value = '')

        st.sidebar.title("Narrow down columns")
        selected_columns = st.sidebar.multiselect("Select Columns to Keep", data.columns)
        if selected_columns:
            data = data[selected_columns]
        st.sidebar.divider()

        st.sidebar.title("Filters, Cleansing, and Conditions")
        st.sidebar.subheader("Filter 1")
        selected_column_1 = st.sidebar.selectbox("Select a column to filter by", data.columns, key='sb1')
        column_type_1 = data[selected_column_1].dtype

        if column_type_1 == "object":  # Categorical column
            filter_option_1 = "Categorical Filter"
            filter_value_1 = st.sidebar.selectbox(f"Select a value from '{selected_column_1}'", data[selected_column_1].unique(), key='sb2')
        elif np.issubdtype(column_type_1, np.number):  # Numerical column
            if selected_column_1 in date_columns:
                filter_option_1 = "Categorical Filter"
                filter_value_1 = st.sidebar.selectbox(f"Select a value from '{selected_column_1}'", data[selected_column_1].unique(), key='sb2.5')
            else:
                filter_option_1 = "Numerical Filter"
                numerical_filter_options_1 = ["None","1 (for dummies)","0 (for dummies)",">0","Above Mean", "Below Mean", "Above Median", "Below Median",
                                            "Within 1 STD", "Within 2 STD", "Within 3 STD", "Above 25th Quantile", "Below 25th Quantile",
                                             "Above 75th Quantile", "Below 75th Quantile" ]
                filter_value_1 = st.sidebar.selectbox(f"Select a filter option for '{selected_column_1}'", numerical_filter_options_1, key='sb3')

        st.sidebar.subheader("Filter 2")
        selected_column_2 = st.sidebar.selectbox("Select a column to filter by", data.columns, key='sb4')
        column_type_2 = data[selected_column_2].dtype

        if column_type_2 == "object":  # Categorical column
            filter_option_2 = "Categorical Filter"
            filter_value_2 = st.sidebar.selectbox(f"Select a value from '{selected_column_2}'", data[selected_column_2].unique(), key='sb5')
        elif np.issubdtype(column_type_2, np.number):  # Numerical column
            if selected_column_2 in date_columns:
                filter_option_2 = "Categorical Filter"
                filter_value_2 = st.sidebar.selectbox(f"Select a value from '{selected_column_2}'", data[selected_column_2].unique(), key='sb5.5')
            else:
                filter_option_2 = "Numerical Filter"
                numerical_filter_options_2 = ["None","1 (for dummies)","0 (for dummies)","Above Mean", "Below Mean", "Above Median", "Below Median",
                                            "Within 1 STD", "Within 2 STD", "Within 3 STD", "Above 25th Quantile", "Below 25th Quantile",
                                             "Above 75th Quantile", "Below 75th Quantile" ]
                filter_value_2 = st.sidebar.selectbox(f"Select a filter option for '{selected_column_2}'", numerical_filter_options_2, key='sb6')

        st.sidebar.subheader("Filter 3")
        selected_column_3 = st.sidebar.selectbox("Select a column to filter by", data.columns, key='sb7')
        column_type_3 = data[selected_column_3].dtype

        if column_type_3 == "object":  # Categorical column
            filter_option_3 = "Categorical Filter"
            filter_value_3 = st.sidebar.selectbox(f"Select a value from '{selected_column_3}'", data[selected_column_3].unique(), key='sb8')
        elif np.issubdtype(column_type_3, np.number):  # Numerical column
            if selected_column_3 in date_columns:
                filter_option_3 = "Categorical Filter"
                filter_value_3 = st.sidebar.selectbox(f"Select a value from '{selected_column_2}'", data[selected_column_2].unique(), key='sb8.5')
            else:
                filter_option_3 = "Numerical Filter"
                numerical_filter_options_3 = ["None","1 (for dummies)","0 (for dummies)","Above Mean", "Below Mean", "Above Median", "Below Median",
                                            "Within 1 STD", "Within 2 STD", "Within 3 STD", "Above 25th Quantile", "Below 25th Quantile",
                                             "Above 75th Quantile", "Below 75th Quantile" ]
            filter_value_3 = st.sidebar.selectbox(f"Select a filter option for '{selected_column_3}'", numerical_filter_options_3, key='sb9')

        # Apply All Filters
        if st.sidebar.checkbox("Active Filters", key='b0'):
            # Filter the data using the filter_data function if a filter value is provided
            data = filter_data(data, selected_column_1, filter_option_1, filter_value_1)
            data = filter_data(data, selected_column_2, filter_option_2, filter_value_2)
            data = filter_data(data, selected_column_3, filter_option_3, filter_value_3)

        st.sidebar.divider()

        # Display the final filtered data
        st.subheader("Filtered Data")
        st.write(data)
    #----------------------------------------------------------EDA Section-----------------------------------------------------------------------------
        # Exploratory Data Analysis
        if analysis_type == "Exploratory Data Analysis":
            st.subheader("Exploratory Data Analysis")
            st.text("Table of summary statistics")
            st.table(data.describe().applymap(lambda x: f"{x:0.2f}"))

                # Iterate through columns to find summary statistics

            # function that calculates statistics for the relationship between variables
            def calculate_correlations(df):
                correlation_data = []

                # Iterate through pairs of numerical columns
                numerical_columns = df.select_dtypes(include=[np.number]).columns
                for i in range(len(numerical_columns)):
                    for j in range(i + 1, len(numerical_columns)):
                        column1 = numerical_columns[i]
                        column2 = numerical_columns[j]

                        
                        filtered_df = df[[column1, column2]].dropna()

                        if len(filtered_df) > 0:
                            
                            slope, intercept, r_value, p_value, std_err = linregress(filtered_df[column1], filtered_df[column2])

                            
                            correlation_data.append({
                                'Column1': column1,
                                'Column2': column2,
                                'Correlation': r_value,  
                                'R-squared': r_value ** 2,
                                'Beta Coefficient': slope,
                                'P-value': p_value,
                                'Standard Error': std_err
                            })

                return correlation_data

            # Filter Correlations
            st.subheader("Filter Correlations")
            min_corr = st.number_input("Minimum Correlation (Absolute Value):", -1.0, 1.0, -1.0, 0.01)
            min_r_squared = st.number_input("Minimum R-squared:", 0.0, 1.0, 0.0, 0.01)
            min_beta_coefficient = st.number_input("Minimum Beta Coefficient:", -1000.0, 1000.0, -1000.0)
            max_p_value = st.number_input("Maximum P-value:", 0.0, 1.0, 1.0, 0.01)

            if st.button("Apply Correlation Filters"):
                correlation_data = calculate_correlations(data)

                # Apply user-defined filters
                filtered_correlation_data = [
                    corr for corr in correlation_data
                    if abs(corr['Correlation']) >= min_corr
                    and abs(corr['R-squared']) >= min_r_squared
                    and abs(corr['Beta Coefficient']) >= min_beta_coefficient
                    and corr['P-value'] <= max_p_value
                    and corr['Column1'] != corr['Column2']  # Exclude self-correlations
                    and (abs(corr['Correlation']) >= min_corr or corr['Correlation'] < 0)  # Filter by minimum correlation in both directions
                ]

                if filtered_correlation_data:
                    st.subheader("Filtered Correlation Statistics:")
                    st.table(pd.DataFrame(filtered_correlation_data))
                else:
                    st.warning("No correlations found that meet the specified criteria.")
            
            # putting the Table into streamlit
            correlation_data = calculate_correlations(data)
            st.table(pd.DataFrame(correlation_data))

    #---------------------------------------------------Regression Analysis Section-------------------------------------------------------------------
        elif analysis_type == "Regression Modeling":
            st.header("Regression Analysis")        
            st.subheader("Scatter Plot and Regression Analysis")
            st.text("Select 2 columns from your data to run a regression. The check boxes below give you options to \nobserve regressions of different degrees and test out the accuracy of the\nmodel doing a train-test split")

            # Plotting X and Y columns
            x_column = st.selectbox("Select X-column variable", data.columns,key='sfodkn')
            y_column = st.selectbox("Select Y-column variable", data.columns,key='dfhkh')
            
            # Checkbox to enable regression line
            plot_regression = st.checkbox("Plot Regression Line",key='prl')
            tts_box = st.checkbox("Train-Test Split",key='tts')
            degree = st.slider("Select Polynomial Degree", min_value=1, max_value=10, value=1)
            
            # Create a scatter plot using Plotly Graph Objects
            fig = go.Figure()
            
            scatter_trace = go.Scatter(
                x=data[x_column],
                y=data[y_column],
                mode='markers',
                opacity=0.6,
                name="Scatter Plot"
            )
            
            fig.add_trace(scatter_trace)
            
            # Perform regression and plot the regression line if enabled
            if plot_regression:
                X = data[x_column].values.reshape(-1, 1)
                Y = data[y_column].values
            
                if degree > 1:
                    poly_features = PolynomialFeatures(degree=degree)
                    X_poly = poly_features.fit_transform(X)
                    reg = LinearRegression().fit(X_poly, Y)
                    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                    X_plot_poly = poly_features.transform(X_plot)
                    Y_plot = reg.predict(X_plot_poly)

                    # Get the coefficients for the polynomial equation
                    coefficients = reg.coef_
                    intercept = reg.intercept_
                    
            
                    # Build the polynomial equation as a string dynamically
                    equation = f"Y = "
                    for i, coef in enumerate(coefficients):
                        if i == 0:
                            equation += f"{coef:.4f}"
                        else:
                            equation += f" + {coef:.4f}X^{i}"
            
                    equation += f" + {intercept:.4f}"
                    
                    # Calculate R-squared
                    Y_pred = reg.predict(X_poly)
                    r_squared = r2_score(Y, Y_pred)

                    st.write("Equation: " + equation)
                    st.write("R-squared: " + f"{r_squared:.4f}")
                    
                else:
                    reg = LinearRegression().fit(X, Y)
                    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                    Y_plot = reg.predict(X_plot)

                    # Calculate the regression equation
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x=data[x_column], y=data[y_column])
                    equation = f"Y = {slope:.2f}X + {intercept:.2f}"
                    R_squared = f"{r_value**2:.4f}"
                    P_value = f"{p_value:.4f}"
                    

                    st.write("Equation: " + equation)
                    st.write("R-squared: " + R_squared)
                    st.write("P-value: " + P_value)
                    
            
                # Add the regression line as a trace to the figure
                regression_trace = go.Scatter(
                    x=X_plot.flatten(),
                    y=Y_plot,
                    mode='lines',
                    name="Regression Line",
                    line=dict(color='red')
                )
            
                fig.add_trace(regression_trace)
            
            # Update layout to include labels
            fig.update_layout(
                xaxis_title=x_column,
                yaxis_title=y_column,
                title="Scatter Plot with Regression Line"
            )

            if tts_box:
                # Input for train-test split ratio
                test_size = st.slider("Select Test Size Ratio", min_value=0.1, max_value=0.5, step=0.05, value=0.2)

                # Perform train-test split
                X = data[x_column].values.reshape(-1, 1)
                Y = data[y_column].values
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

                degrees = [1,2,3,4,5,6,7,8,9,10]
                optimal_deg = calculate_regression_metrics(data, x_column, y_column, degrees, test_size)
                st.write(f'Optimal Degree to regress on: {optimal_deg}')
                st.text("See bottom of the page for more information on optimization technique")

                # Calculate model accuracy metrics
                if degree > 1:
                    # For polynomial regression
                    poly_features = PolynomialFeatures(degree=degree)
                    X_train_poly = poly_features.fit_transform(X_train)
                    X_test_poly = poly_features.transform(X_test)
                    reg = LinearRegression().fit(X_train_poly, Y_train)
                    Y_pred_train = reg.predict(X_train_poly)
                    Y_pred_test = reg.predict(X_test_poly)
                else:
                    # For linear regression
                    reg = LinearRegression().fit(X_train, Y_train)
                    Y_pred_train = reg.predict(X_train)
                    Y_pred_test = reg.predict(X_test)
                
                # Calculate MSE and R-squared for train and test sets
                mse_train = mean_squared_error(Y_train, Y_pred_train)
                mse_test = mean_squared_error(Y_test, Y_pred_test)
                r_squared_train = r2_score(Y_train, Y_pred_train)
                r_squared_test = r2_score(Y_test, Y_pred_test)
                mse_percent_error = abs(((mse_test - mse_train)/mse_train))
                r_squared_percent_error = abs(((r_squared_test - r_squared_train)/r_squared_train))

                                # Create a dictionary with placeholders for MSE and R-squared values
                results = {
                    "MSE": [mse_train, mse_test, mse_percent_error],
                    "R-Squared": [r_squared_train, r_squared_test, r_squared_percent_error]
                }
                
                # Define the row index (row headers)
                index = ["Train", "Test", "% Difference"]
                
                # Create the DataFrame
                results_df = pd.DataFrame(results, index=index)

                st.table(results_df)
            
            # Show the chart
            st.plotly_chart(fig)

            st.divider()

            #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            st.subheader("OLS Regression Table")
            st.text("Below is a feature that can produce a multiple regression table, \nwhich includes one dependent variable, several independent variables, \nand optional interaction terms.")
            
            dependent_var = st.selectbox("Select dependent variable", data.columns)
            independent_vars = st.multiselect("Select independent variables", data.columns)

            # Allow users to select interaction terms
            interaction_terms = st.multiselect("Select interaction terms", [f"{var1}:{var2}" for var1 in independent_vars for var2 in independent_vars if var1 != var2], key="interaction_terms")

            if st.button("Run Regression"):
                try:

                    # Check if interaction terms need to be created and appended
                    if interaction_terms:
                        data = create_and_append_interaction_terms(data, independent_vars, interaction_terms)

                    model = perform_linear_regression(data, dependent_var, independent_vars + interaction_terms)
                    st.subheader("Regression Summary")

                    # Generate a basic text-based regression table using tabulate
                    regression_summary = tabulate(model.summary().tables[1], headers='keys', tablefmt='plain')
                    st.text(regression_summary)

                    # Statistical significance section
                    significance_level = 0.05
                    p_values = model.pvalues[1:]  # Exclude the constant
                    significant_vars = [var for var, p_value in zip(independent_vars, p_values) if p_value <= significance_level]

                    st.subheader("Statistical Significance")
                    st.write("Significance level:", significance_level)
                    st.write("Significant variables:", ", ".join(significant_vars))
                except:
                    st.text("There is an error with your regression.\nNote that you cannot regress on categorical variables, string variables, and \nmissing values")
            st.divider()
            st.subheader("Note on optimization technique")
            st.write("""
            Only degrees 1-10 are considered in this function, which should be a large enough range to minimize most functions. However, this may not always be the
            case, and the user should attempt other optimization techniques. 

            The first step to finding the the optimal degree is finding the MSE and the R-squared values from both the train and test groups. Next, find the percent
            difference between the train and test groups. The closer to zero this difference is, the more accurate the trained model is at predicting values in the 
            test group. The optimal degree is going to be the degree that corresponds with the absolute minimum between both of the following functions:

            MSE_diff(degree) & R2_diff(degree)

            Then, using the following formula at each degree 1-10 to combine the functions:

            (f(x) + g(x) - |f(x) - g(x)|)/2

            We simply find the degree that has the lowest value. 
            """)
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif analysis_type == "Data Visualizations":

            # Header for visualization page
            st.header("Data Visualizations")
            
            # Function used to call histograms and scatter plots
            def visualize_data(data):

                # Histogram-------------------------------------------------------------------------------------------------------------------------------------------------------------
                st.subheader("Histogram")
                selected_column = st.selectbox("Select a column for the histogram", data.columns)
                num_bins = st.number_input("Number of Bins", min_value=1, max_value=100, value=20, step=1)
                fig = px.histogram(data, x=selected_column, nbins=num_bins)
                st.plotly_chart(fig)
                
                st.divider()
                # Scatter Plot----------------------------------------------------------------------------------------------------------------------------------------------------------
                
                st.subheader("Scatter Plot")
                x_column = st.selectbox("Select X-axis column", data.columns)
                y_column = st.selectbox("Select Y-axis column", data.columns)
                
                # Create a scatter plot using Plotly Graph Objects
                fig = go.Figure()
                
                scatter_trace = go.Scatter(
                    x=data[x_column],
                    y=data[y_column],
                    mode='markers',
                    name="Scatter Plot"
                )
                
                fig.add_trace(scatter_trace)

                fig.update_layout(
                    xaxis_title=x_column,    # X-axis title
                    yaxis_title=y_column,    # Y-axis title
                    title= f"Scatter of {x_column} and {y_column}"     # Plot title
                )

                st.plotly_chart(fig)
                    
                st.divider()
                
                # Bar Chart-------------------------------------------------------------------------------------------------------------------------------------------------------------

                # Function to check if a column is categorical (object dtype)
                def is_categorical_column(column):
                    return column.dtype == "object"
                
                st.subheader("Bar Chart")
                x_column = st.selectbox("Select X-axis column (binary or categorical)", data.columns)
                y_column = st.selectbox("Select Y-axis column (numerical)", data.select_dtypes(include=['number']).columns)
                aggregation_type = st.selectbox("Select Aggregation Type", ["Sum", "Mean", "Median", "Count"])
                
                if aggregation_type == "Sum":
                    y_data = data.groupby(x_column)[y_column].sum().reset_index()
                elif aggregation_type == "Mean":
                    y_data = data.groupby(x_column)[y_column].mean().reset_index()
                elif aggregation_type == "Median":
                    y_data = data.groupby(x_column)[y_column].median().reset_index()
                elif aggregation_type == "Count":
                    y_data = data.groupby(x_column)[y_column].count().reset_index()
                fig = px.bar(y_data, x=x_column, y=y_column)
                st.plotly_chart(fig)

                st.divider()

                # Time Series-----------------------------------------------------------------------------------------------------------------------------------------------------------
                # Identify the x-axis column
                x_axis_column = identify_x_axis(data)
                
                # Allow the user to select the y-axis column for the main line
                y_axis_column = st.selectbox("Select Y-axis column", data.columns, key="y_axis_select")
                
                # Allow the user to choose "Mean" or "Sum" with a unique key for the main line
                aggregation_type = st.selectbox("Select Aggregation Type", ["Mean", "Sum"], key="aggregation_select")
                
                # Allow the user to input the number of periods for aggregation
                num_periods = st.number_input("Number of Periods", min_value=1, max_value=len(data), value=52, key="num_periods_input")
                
                # Perform aggregation based on the user's selection for the main line
                if aggregation_type == "Mean":
                    y_data = data.groupby(x_axis_column)[y_axis_column].mean().reset_index()
                elif aggregation_type == "Sum":
                    y_data = data.groupby(x_axis_column)[y_axis_column].sum().reset_index()
                
                # Sort the data by the x-axis column in ascending order
                y_data = y_data.sort_values(by=[x_axis_column])
                
                # Calculate the number of data points to include in each period
                period_size = len(y_data) // num_periods
                
                # Create aggregated data by grouping into periods and applying the selected aggregation function
                aggregated_data = y_data.groupby(y_data.index // period_size)[y_axis_column].agg('mean').reset_index()
                aggregated_data[x_axis_column] = y_data.groupby(y_data.index // period_size)[x_axis_column].first()
                
                # Create a line chart for the main line
                fig = go.Figure()
                
                # Add the time series line trace
                fig.add_trace(go.Scatter(x=aggregated_data[x_axis_column], y=aggregated_data[y_axis_column], mode='lines', name=f"{aggregation_type} {y_axis_column}"))
                
                # Specify axis labels and title
                fig.update_layout(
                    xaxis_title=x_axis_column,
                    yaxis_title=y_axis_column,
                    title=f"{aggregation_type} {y_axis_column} over Time (Aggregated)"
                )
                
                # Specify that the x-axis is a date axis
                fig.update_xaxes(type='date')
                
                # Show the chart
                st.plotly_chart(fig)
                
            visualize_data(data)

        if analysis_type == "KMeans Clustering":
            # Header for clustering options
            st.subheader("K-means Clustering Options")
        
            # Select features for clustering
            selected_features = st.multiselect("Select Features for Clustering", data.columns)
        
            # Number of clusters (K)
            num_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)
        
            # Select the testing column
            testing_column = st.selectbox("Select Testing Column", data.columns)
        
            # Perform clustering when the user clicks the "Run Clustering" button
            if st.button("Run Clustering"):
                if len(selected_features) < 2:
                    st.warning("Please select at least two features for clustering.")
                elif testing_column == "":
                                        # Prepare the data for clustering
                    data_for_clustering = data[selected_features]
        
                    # Perform K-means clustering
                    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
                    data['Cluster'] = kmeans.fit_predict(data_for_clustering)
        
                    # Visualize the clusters using a scatter plot with Plotly
                    fig = px.scatter(data, x=selected_features[0], y=selected_features[1], color='Cluster',
                                    title="K-means Clustering")
                    st.plotly_chart(fig)
        
                    st.text("Remember that this only plots the first two dimensions, but the \namount of dimensions depends on the amount of features listed")
        
                    # Display clustering results
                    st.subheader("Clustering Results")
                    st.write(data[['Cluster'] + selected_features])
        
                    # Show cluster statistics
                    cluster_counts = data['Cluster'].value_counts().reset_index()
                    cluster_counts.columns = ['Cluster', 'Count']
                    st.subheader("Cluster Counts")
                    st.write(cluster_counts)
        
                    # Display centroids
                    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=selected_features)
                    st.subheader("Cluster Centroids")
                    st.write(centroids)
        
                    # Calculate and display inertia (within-cluster sum of squares)
                    inertia = kmeans.inertia_
                    st.subheader("Inertia (Within-Cluster Sum of Squares)")
                    st.write(inertia)
        
                    # Calculate and display silhouette score
                    silhouette_avg = silhouette_score(data_for_clustering, kmeans.labels_)
                    st.subheader("Silhouette Score")
                    st.write(silhouette_avg)
                    
                else:
                    # Prepare the data for clustering
                    data_for_clustering = data[selected_features]
        
                    # Perform K-means clustering
                    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
                    data['Cluster'] = kmeans.fit_predict(data_for_clustering)
        
                    # Evaluate clustering
                    clustering_accuracy = custom_accuracy_score(data, testing_column, 'Cluster')
        
                    # Visualize the clusters using a scatter plot with Plotly
                    fig = px.scatter(data, x=selected_features[0], y=selected_features[1], color='Cluster',
                                    title="K-means Clustering")
                    st.plotly_chart(fig)
        
                    st.text("Remember that this only plots the first two dimensions, but the \namount of dimensions depends on the amount of features listed")
        
                    # Display clustering results
                    st.subheader("Clustering Results")
                    st.write(data[['Cluster'] + selected_features])
        
                    # Show cluster statistics
                    cluster_counts = data['Cluster'].value_counts().reset_index()
                    cluster_counts.columns = ['Cluster', 'Count']
                    st.subheader("Cluster Counts")
                    st.write(cluster_counts)
        
                    # Display centroids
                    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=selected_features)
                    st.subheader("Cluster Centroids")
                    st.write(centroids)
        
                    # Calculate and display inertia (within-cluster sum of squares)
                    inertia = kmeans.inertia_
                    st.subheader("Inertia (Within-Cluster Sum of Squares)")
                    st.write(inertia)
        
                    # Calculate and display silhouette score
                    silhouette_avg = silhouette_score(data_for_clustering, kmeans.labels_)
                    st.subheader("Silhouette Score")
                    st.write(silhouette_avg)
        
                    # Display clustering evaluation
                    st.subheader("Clustering Evaluation")
                    st.text(f"Clustering Accuracy (w.r.t. {testing_column}): {clustering_accuracy:.2f}") 
            
    else:
        st.subheader("About the app")
        st.write("This app was created by Seth Ruffing (TAMU Co'23) in order to perform exploratory analysis on CSV files. Current analyses include bivariate analyses, data visualizations, in-depth regression model building, and KMeans clustering")
        st.write("I made this app because I often performed EDA on new datasets to find variables that had sometimes of relationship, and I thought I could speed up the process by having a prewritten algorithm that finds bivariate relationships with my input parameters.")
        st.write("From there, more and more features were added to incrementally increase the usability of this app. I also had been wanting to learn how to apply ML model into my work so I use this app as the opportunity")


#-------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
