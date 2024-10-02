import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Load dataset
day_data = pd.read_csv("data/day.csv")
hourly_data = pd.read_csv("data/hour.csv")
day_data['dteday'] = pd.to_datetime(day_data['dteday'])
hourly_data['dteday'] = pd.to_datetime(hourly_data['dteday'])

# Sidebar for Analysis Type
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type:",
    options=["Seasonal Analysis", "Correlation Analysis", "Weekly Usage Patterns", "Predictive Modeling"]
)

# Sidebar for year and date filtering
year_options = st.sidebar.multiselect(
    "Select Year(s) to Analyze:",
    options=[2011, 2012],
    default=[2011, 2012]
)

# Custom date range filtering
start_date = st.sidebar.date_input('Start Date', min_value=datetime(2011, 1, 1), max_value=datetime(2012, 12, 31), value=datetime(2011, 1, 1))
end_date = st.sidebar.date_input('End Date', min_value=start_date, max_value=datetime(2012, 12, 31), value=datetime(2012, 12, 31))

# Convert start_date and end_date to datetime
start_datetime = pd.to_datetime(start_date)
end_datetime = pd.to_datetime(end_date)

# Filter data by year and date
filtered_day_data = day_data[(day_data['yr'].isin([year - 2011 for year in year_options])) & (day_data['dteday'].between(start_datetime, end_datetime))]
filtered_hourly_data = hourly_data[(hourly_data['yr'].isin([year - 2011 for year in year_options])) & (hourly_data['dteday'].between(start_datetime, end_datetime))]


# Main Dashboard Title
st.title("Enhanced Bike Sharing Data Analysis Dashboard")

# Section dividers for better readability
st.markdown("---")

# Seasonal Analysis ###
if analysis_type == "Seasonal Analysis":
    st.header("Seasonal Analysis")
    st.write("### Examine the Effect of Seasonality on Rental Counts")

    # Select season from Sidebar
    season_option = st.sidebar.selectbox(
        "Select Season:",
        options=["Spring", "Summer", "Fall", "Winter"]
    )

    # Season map
    season_map = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
    filtered_season_data = filtered_day_data[filtered_day_data['season'] == season_map[season_option]]

    # Casual vs Registered Users Plot
    st.subheader(f"Casual vs Registered Users in {season_option}")
    avg_casual = filtered_season_data['casual'].mean()
    avg_registered = filtered_season_data['registered'].mean()

    fig = go.Figure(data=[
        go.Bar(name='Casual Users', x=['Casual'], y=[avg_casual], marker_color='#FFDDC1'),
        go.Bar(name='Registered Users', x=['Registered'], y=[avg_registered], marker_color='#7CB9E8')
    ])
    fig.update_layout(barmode='group', title_text=f"Average Casual vs Registered Users in {season_option}",
                      xaxis_title="User Type", yaxis_title="Number of Users")
    st.plotly_chart(fig)

    # Rental count by day line plot
    st.subheader(f"Daily Rental Count in {season_option}")
    line_fig = px.line(filtered_season_data, x='dteday', y='cnt', title=f"Rental Count Over Time in {season_option}",
                       labels={'dteday': 'Date', 'cnt': 'Rental Count'})
    line_fig.update_xaxes(showspikes=True, spikemode="across")
    line_fig.update_yaxes(showspikes=True)
    st.plotly_chart(line_fig)

    # Downloadable data feature
    st.download_button(label="Download Filtered Data as CSV", data=filtered_season_data.to_csv(), file_name='filtered_data.csv')

# Correlation Analysis 
elif analysis_type == "Correlation Analysis":
    st.header("Correlation Analysis")
    st.write("### Analyze the Correlation Between Different Variables")

    # Select variable for correlation analysis
    corr_var = st.sidebar.selectbox(
        "Select Variable for Correlation:",
        options=["temp", "hum", "windspeed"]
    )

    # Compute and display correlation
    st.subheader(f"Correlation between Rental Count and {corr_var.capitalize()}")
    correlation = filtered_day_data[['cnt', corr_var]].corr().iloc[0, 1]
    st.metric(label=f"Correlation with {corr_var.capitalize()}", value=round(correlation, 2))

    # Scatter plot for correlation analysis
    scatter_fig = px.scatter(filtered_day_data, x=corr_var, y='cnt', title=f"{corr_var.capitalize()} vs Rental Count",
                             labels={corr_var: corr_var.capitalize(), 'cnt': 'Rental Count'})
    st.plotly_chart(scatter_fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_matrix = filtered_day_data[['cnt', 'temp', 'hum', 'windspeed']].corr()
    heatmap_fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
    st.plotly_chart(heatmap_fig)

# Weekly Usage Patterns 
elif analysis_type == "Weekly Usage Patterns":
    st.header("Weekly Usage Patterns")
    st.write("### Analyze Bicycle Use Patterns During the Week")

    # Pivot and heatmap for hourly usage by day
    pivot_df = filtered_hourly_data.pivot_table(values='cnt', index='hr', columns='weekday', aggfunc='mean')
    heatmap_fig = px.imshow(
    pivot_df,
    color_continuous_scale='Blues',
    title="Bike Usage by Hour and Day",
    labels={'weekday': 'Day of the Week', 'hr': 'Hour'},
    width=1800,  # Adjust width
    height=1400  # Adjust height
    )
    
    heatmap_fig.update_layout(
    xaxis_title='Day of the Week',
    yaxis_title='Hour',
    xaxis=dict(tickmode='array', tickvals=list(range(len(pivot_df.columns))), ticktext=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']),
    yaxis=dict(tickmode='array', tickvals=list(range(len(pivot_df.index))), ticktext=pivot_df.index)
    )


    st.plotly_chart(heatmap_fig)

# Predictive Modeling 
elif analysis_type == "Predictive Modeling":
    st.header("Predictive Modeling")
    st.write("### Forecast Future Bike Usage Based on Historical Data")

    # Allow users to select multiple features
    features = ['temp', 'atemp', 'hum', 'windspeed']  
    selected_features = st.multiselect("Select Features for Prediction", features, default=['temp'])

    if selected_features:
        # Prepare data for modeling
        X = filtered_day_data[selected_features].values
        y = filtered_day_data['cnt'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Predictive plot
        if len(selected_features) == 1:
            # Fit the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.subheader("Model Performance")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
            
            # If there's only one feature, create a scatter plot
            pred_fig = px.scatter(x=X_test.flatten(), y=y_test, labels={'x': selected_features[0], 'y': 'Actual Count'},
                                  title="Actual vs Predicted Rental Counts")
            pred_fig.add_traces(go.Scatter(x=X_test.flatten(), y=y_pred, mode='lines', name='Predicted Count',
                                             line=dict(color='firebrick', width=2)))
        elif len(selected_features) == 2:
            # Create polynomial features
            degree = 2  
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            # Fit the model
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            y_pred = model.predict(X_test_poly)
            
            # Create a grid for the surface plot
            x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
            y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
            x_grid, y_grid = np.meshgrid(x_range, y_range)

            # Prepare input for predictions
            grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
            grid_poly = poly.transform(grid_points)

            # Predict on the grid
            z_pred = model.predict(grid_poly).reshape(x_grid.shape)

            # Create a 3D surface plot
            pred_fig = go.Figure(data=[
                go.Surface(z=z_pred, x=x_grid, y=y_grid, colorscale='Blues', opacity=0.8)
            ])
            pred_fig.update_layout(title='Predicted Counts Surface Plot',
                                   scene=dict(
                                       xaxis_title=selected_features[0],
                                       yaxis_title=selected_features[1],
                                       zaxis_title='Predicted Count'),
                                   width=800,
                                   height=600)

        else:
            # Create polynomial features
            degree = 2 
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            # Fit the model
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            y_pred = model.predict(X_test_poly)
            
            # For more than two features, show coefficients
            st.warning("With more than two features selected, the model is polynomial regression. "
                       "Interpret results with caution as relationships may be complex.")
            st.write("Model coefficients:")
            coefficients = pd.DataFrame(model.coef_, index=poly.get_feature_names_out(selected_features), columns=["Coefficient"])
            st.write(coefficients)
            pred_fig = None

        
    else:
        st.warning("Please select at least one feature for prediction.")

    # Only plot if pred_fig is defined
    if pred_fig is not None:
        st.plotly_chart(pred_fig)
