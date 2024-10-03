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
    options=["Weather Analysis", "Weekly Usage Analysis","Seasonal Analysis", "Correlation Analysis", "Weekly Usage Patterns", "Predictive Modeling"],
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

# Weather Analysis
if analysis_type == "Weather Analysis":
    st.header("Weather Analysis")
    st.write("### Examine the Effect of Weather on Rental Counts")

    # Allow users to select multiple weather conditions
    weather_options = ['Clear', 'Mist', 'Light Snow/Rain', 'Heavy Snow/Rain']
    selected_weather = st.multiselect("Select Weather Conditions for Analysis", weather_options, default=['Clear', 'Mist', 'Light Snow/Rain','Heavy Snow/Rain'])

    # Map the selected weather conditions to numeric values
    weather_map = {"Clear": 1, "Mist": 2, "Light Snow/Rain": 3, "Heavy Snow/Rain": 4}
    selected_weather_codes = [weather_map[weather] for weather in selected_weather]

    # Filter data based on selected weather conditions
    filtered_weather_data = filtered_day_data[filtered_day_data['weathersit'].isin(selected_weather_codes)]

    ### Average Users by Weather (Daily Data) ###
    st.subheader("Daily Average Users by Weather Condition")

    # Group the data by weather condition and calculate the mean for casual, registered, and total users
    weather_agg_day = filtered_weather_data.groupby('weathersit').agg({
        'casual': 'mean',
        'registered': 'mean',
        'cnt': 'mean'
    }).reset_index()

    # Create a grouped bar plot using Plotly
    fig1 = go.Figure()

    fig1.add_trace(go.Bar(
        x=[weather_options[code-1] for code in weather_agg_day['weathersit']], 
        y=weather_agg_day['casual'], 
        name='Casual Users'
    ))

    fig1.add_trace(go.Bar(
        x=[weather_options[code-1] for code in weather_agg_day['weathersit']], 
        y=weather_agg_day['registered'], 
        name='Registered Users'
    ))

    fig1.add_trace(go.Bar(
        x=[weather_options[code-1] for code in weather_agg_day['weathersit']], 
        y=weather_agg_day['cnt'], 
        name='Total Users'
    ))

    # Update layout
    fig1.update_layout(
        title='Impact of Weather Conditions on Daily Bike Usage',
        xaxis_title='Weather Condition',
        yaxis_title='Average Number of Users',
        barmode='group'
    )

    st.plotly_chart(fig1)

    ### Average Users by Weather (Hourly Data) ###
    st.subheader("Hourly Average Users by Weather Condition")

    # Filter hourly data based on selected weather conditions
    filtered_weather_hourly = filtered_hourly_data[filtered_hourly_data['weathersit'].isin(selected_weather_codes)]

    # Group the data by weather condition and calculate the mean for casual, registered, and total users
    weather_agg_hourly = filtered_weather_hourly.groupby('weathersit').agg({
        'casual': 'mean',
        'registered': 'mean',
        'cnt': 'mean'
    }).reset_index()

    # Create a grouped bar plot for hourly data
    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=[weather_options[code-1] for code in weather_agg_hourly['weathersit']], 
        y=weather_agg_hourly['casual'], 
        name='Casual Users'
    ))

    fig2.add_trace(go.Bar(
        x=[weather_options[code-1] for code in weather_agg_hourly['weathersit']], 
        y=weather_agg_hourly['registered'], 
        name='Registered Users'
    ))

    fig2.add_trace(go.Bar(
        x=[weather_options[code-1] for code in weather_agg_hourly['weathersit']], 
        y=weather_agg_hourly['cnt'], 
        name='Total Users'
    ))

    # Update layout
    fig2.update_layout(
        title='Impact of Weather Conditions on Hourly Bike Usage',
        xaxis_title='Weather Condition',
        yaxis_title='Average Number of Users',
        barmode='group'
    )

    st.plotly_chart(fig2)

    ###  Box Plot  ###
    ### Daily ###
    st.subheader("Box Plot of Total Users by Weather Condition (Daily Data)")

    # Create a box plot for total users by weather condition
    fig_box_daily = go.Figure()

    for weather_code in selected_weather_codes:
        fig_box_daily.add_trace(go.Box(
            y=filtered_weather_data[filtered_weather_data['weathersit'] == weather_code]['cnt'],
            name=weather_options[weather_code - 1],
            boxmean='sd'  # show mean and standard deviation
        ))

    fig_box_daily.update_layout(
        title='Boxplot of Total Users by Weather Conditions on Daily Bike Usage',
        xaxis_title='Weather Condition',
        yaxis_title='Number of Total Users'
    )

    st.plotly_chart(fig_box_daily)
    
    ###Hourly###
    st.subheader("Box Plot of Total Users by Weather Condition (Hourly Data)")

    # Create a box plot for total users by weather condition
    fig_box_hourly = go.Figure()

    for weather_code in selected_weather_codes:
        fig_box_hourly.add_trace(go.Box(
            y=filtered_weather_hourly[filtered_weather_hourly['weathersit'] == weather_code]['cnt'],
            name=weather_options[weather_code - 1],
            boxmean='sd'  # show mean and standard deviation
        ))

    fig_box_hourly.update_layout(
        title='Boxplot of Total Users by Weather Conditions on Hourly Bike Usage',
        xaxis_title='Weather Condition',
        yaxis_title='Number of Total Users'
    )

    st.plotly_chart(fig_box_hourly)

    ### Violin Plot ###
    ### Daily ###
    st.subheader("Violin Plot of Total Users by Weather Condition (Daily Data)")

    # Create a violin plot for total users by weather condition
    fig_violin_daily = go.Figure()

    for weather_code in selected_weather_codes:
        fig_violin_daily.add_trace(go.Violin(
            y=filtered_weather_data[filtered_weather_data['weathersit'] == weather_code]['cnt'],
            name=weather_options[weather_code - 1],
            box_visible=True,  # display box inside violin
            meanline_visible=True  # display mean line
        ))

    fig_violin_daily.update_layout(
        title='Violin Plot of Total Users by Weather Conditions on Daily Bike Usage',
        xaxis_title='Weather Condition',
        yaxis_title='Number of Total Users'
    )

    st.plotly_chart(fig_violin_daily)

    ### Hourly Data ###
    st.subheader("Violin Plot of Total Users by Weather Condition (Hourly Data)")

    # Filter hourly data based on selected weather conditions
    filtered_weather_hourly = filtered_hourly_data[filtered_hourly_data['weathersit'].isin(selected_weather_codes)]

    # Create a violin plot for hourly data
    fig_violin_hourly = go.Figure()

    for weather_code in selected_weather_codes:
        fig_violin_hourly.add_trace(go.Violin(
            y=filtered_weather_hourly[filtered_weather_hourly['weathersit'] == weather_code]['cnt'],
            name=weather_options[weather_code - 1],
            box_visible=True,  # display box inside violin
            meanline_visible=True  # display mean line
        ))

    fig_violin_hourly.update_layout(
        title='Violin Plot of Total Users by Weather Conditions on Hourly Bike Usage',
        xaxis_title='Weather Condition',
        yaxis_title='Number of Total Users'
    )

    st.plotly_chart(fig_violin_hourly)
    
    ###Line Plot for User Trends Over Time by Weather Condition ###
    ### Daily ###
    st.subheader("User Trends Over Time by Weather Condition on Daily Bike Usage")

    # Define the mapping for weather conditions
    weather_mapping = {1: 'Clear', 2: 'Mist', 3: 'Light Snow/Rain', 4: 'Heavy Snow/Rain'}
    
    # Map the weather condition numeric values to descriptive labels
    filtered_weather_data['weathersitmap'] = filtered_weather_data['weathersit'].map(weather_mapping)

    # Create a line plot for total users over time by weather condition
    fig_line = px.line(
        filtered_weather_data,
        x='dteday',
        y='cnt',
        color='weathersitmap',
        title='Total Users Over Time by Weather Conditions',
        labels={'dteday': 'Date', 'cnt': 'Number of Total Users', 'weathersitmap': 'Weather Condition'},
        template='plotly_dark'
    )

    # Update layout for better readability
    fig_line.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Total Users',
        legend_title='Weather Condition'
    )

    st.plotly_chart(fig_line)
    
    st.write('''
    ### Insights:
    - The bar plots show that bike usage is highest during clear weather, for both casual and registered users. This indicates that good weather conditions are a significant motivator for riding bikes.
    - There is a noticeable drop in the number of total users when the weather shifts to mist or light snow/rain. This trend is observed in both daily and hourly data.
    - Casual users are more sensitive to bad weather compared to registered users. While registered users’ usage decreases slightly in misty or rainy conditions, casual users show a sharper decline, possibly due to the less essential nature of their trips.
    - The boxplots and violin plots indicate that clear weather has a higher range and density of bike usage, with more extreme high values in total users, especially for casual users.
    - The line plot visualizes how bike usage fluctuates over time across different weather conditions. Clear weather is consistently associated with peaks in usage, while bad weather shows flatter trends with fewer users.
    - The weather has a significant impact on bike-sharing usage. Clear weather sees the highest number of users, while snowy or rainy conditions result in lower usage.
    - Casual users are more affected by weather conditions than registered users, particularly in harsher weather like rain or snow. This trend is visible in both daily and hourly data.
    - User activity peaks during certain times of day, and this pattern is consistent across different weather conditions.
    ''')
    
# Weekly Usage Analysis
elif analysis_type == "Weekly Usage Analysis":
    st.header("Weekly Usage Analysis")
    st.write("### Examine the Effect of Day on Rental Counts")

    # Allow users to select multiple days of the week
    day_options = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    selected_days = st.multiselect("Select Days for Analysis", day_options, default=day_options)

    # Map the selected days to their corresponding numbers
    day_map = {"Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6}
    selected_day_codes = [day_map[day] for day in selected_days]

    # Filter data based on selected days
    filtered_weekly_data = filtered_day_data[filtered_day_data['weekday'].isin(selected_day_codes)]

    ### Average Users by Day (Daily Data) ###
    st.subheader("Daily Average Users by Day of the Week")

    # Group the data by day of the week and calculate the mean for casual, registered, and total users
    day_agg_day = filtered_weekly_data.groupby('weekday').agg({
        'casual': 'mean',
        'registered': 'mean',
        'cnt': 'mean'
    }).reset_index()

    # Create a grouped bar plot using Plotly
    fig1 = go.Figure()

    fig1.add_trace(go.Bar(
        x=[day_options[code] for code in day_agg_day['weekday']], 
        y=day_agg_day['casual'], 
        name='Casual Users'
    ))

    fig1.add_trace(go.Bar(
        x=[day_options[code] for code in day_agg_day['weekday']], 
        y=day_agg_day['registered'], 
        name='Registered Users'
    ))

    fig1.add_trace(go.Bar(
        x=[day_options[code] for code in day_agg_day['weekday']], 
        y=day_agg_day['cnt'], 
        name='Total Users'
    ))

    # Update layout
    fig1.update_layout(
        title='Impact of Day of the Week on Daily Bike Usage',
        xaxis_title='Day of the Week',
        yaxis_title='Average Number of Users',
        barmode='group'
    )

    st.plotly_chart(fig1)

    ### Average Users by Day (Hourly Data) ###
    st.subheader("Hourly Average Users by Day of the Week")

    # Filter hourly data based on selected days
    filtered_weekly_hourly = filtered_hourly_data[filtered_hourly_data['weekday'].isin(selected_day_codes)]

    # Group the data by day of the week and calculate the mean for casual, registered, and total users
    day_agg_hourly = filtered_weekly_hourly.groupby('weekday').agg({
        'casual': 'mean',
        'registered': 'mean',
        'cnt': 'mean'
    }).reset_index()

    # Create a grouped bar plot for hourly data
    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=[day_options[code] for code in day_agg_hourly['weekday']], 
        y=day_agg_hourly['casual'], 
        name='Casual Users'
    ))

    fig2.add_trace(go.Bar(
        x=[day_options[code] for code in day_agg_hourly['weekday']], 
        y=day_agg_hourly['registered'], 
        name='Registered Users'
    ))

    fig2.add_trace(go.Bar(
        x=[day_options[code] for code in day_agg_hourly['weekday']], 
        y=day_agg_hourly['cnt'], 
        name='Total Users'
    ))

    # Update layout
    fig2.update_layout(
        title='Impact of Day of the Week on Hourly Bike Usage',
        xaxis_title='Day of the Week',
        yaxis_title='Average Number of Users',
        barmode='group'
    )

    st.plotly_chart(fig2)

    ### Box Plot for Daily Data ###
    st.subheader("Box Plot of Total Users by Day of the Week (Daily Data)")

    # Create a box plot for total users by day of the week
    fig_box_daily = go.Figure()

    for day_code in selected_day_codes:
        fig_box_daily.add_trace(go.Box(
            y=filtered_weekly_data[filtered_weekly_data['weekday'] == day_code]['cnt'],
            name=day_options[day_code],
            boxmean='sd'  # show mean and standard deviation
        ))

    fig_box_daily.update_layout(
        title='Boxplot of Total Users by Day of the Week on Daily Bike Usage',
        xaxis_title='Day of the Week',
        yaxis_title='Number of Total Users'
    )

    st.plotly_chart(fig_box_daily)
    
    ### Box Plot for Hourly Data ###
    st.subheader("Box Plot of Total Users by Day of the Week (Hourly Data)")

    # Create a box plot for total users by day of the week for hourly data
    fig_box_hourly = go.Figure()

    for day_code in selected_day_codes:
        fig_box_hourly.add_trace(go.Box(
            y=filtered_weekly_hourly[filtered_weekly_hourly['weekday'] == day_code]['cnt'],
            name=day_options[day_code],
            boxmean='sd'  # show mean and standard deviation
        ))

    fig_box_hourly.update_layout(
        title='Boxplot of Total Users by Day of the Week on Hourly Bike Usage',
        xaxis_title='Day of the Week',
        yaxis_title='Number of Total Users'
    )

    st.plotly_chart(fig_box_hourly)

    ### Violin Plot for Daily Data ###
    st.subheader("Violin Plot of Total Users by Day of the Week (Daily Data)")

    # Create a violin plot for total users by day of the week for daily data
    fig_violin_daily = go.Figure()

    for day_code in selected_day_codes:
        fig_violin_daily.add_trace(go.Violin(
            y=filtered_weekly_data[filtered_weekly_data['weekday'] == day_code]['cnt'],
            name=day_options[day_code],
            box_visible=True,  # display box inside violin
            meanline_visible=True  # display mean line
        ))

    fig_violin_daily.update_layout(
        title='Violin Plot of Total Users by Day of the Week on Daily Bike Usage',
        xaxis_title='Day of the Week',
        yaxis_title='Number of Total Users'
    )

    st.plotly_chart(fig_violin_daily)

    ### Violin Plot for Hourly Data ###
    st.subheader("Violin Plot of Total Users by Day of the Week (Hourly Data)")

    # Create a violin plot for total users by day of the week for hourly data
    fig_violin_hourly = go.Figure()

    for day_code in selected_day_codes:
        fig_violin_hourly.add_trace(go.Violin(
            y=filtered_weekly_hourly[filtered_weekly_hourly['weekday'] == day_code]['cnt'],
            name=day_options[day_code],
            box_visible=True,  # display box inside violin
            meanline_visible=True  # display mean line
        ))

    fig_violin_hourly.update_layout(
        title='Violin Plot of Total Users by Day of the Week on Hourly Bike Usage',
        xaxis_title='Day of the Week',
        yaxis_title='Number of Total Users'
    )

    st.plotly_chart(fig_violin_hourly)
    
    ### Line Plot for User Trends Over Time by Day of the Week ###
    st.subheader("User Trends Over Time by Day of the Week on Daily Bike Usage")

    day_mapping = {
    0:'Sunday',
    1:'Monday',
    2:'Tuesday',
    3:'Wednesday',
    4:'Thursday',
    5:'Friday',
    6:'Saturday'
    }
    
    # Map the day of the week numbers to descriptive labels
    filtered_weekly_data['daymap'] = filtered_weekly_data['weekday'].map(day_mapping)

    # Create a line plot for total users over time by day of the week
    fig_line = px.line(
        filtered_weekly_data,
        x='dteday',
        y='cnt',
        color='daymap',
        title='Total Users Over Time by Day of the Week',
        labels={'dteday': 'Date', 'cnt': 'Number of Total Users', 'daymap': 'Day of the Week'},
        template='plotly_dark'
    )

    # Update layout for better readability
    fig_line.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Total Users',
        legend_title='Day of the Week'
    )

    st.plotly_chart(fig_line)
    
    st.write('''
    ### Insights:
    - From the bar plots showing the average number of casual, registered, and total users across different days, we see a clear distinction between weekdays (Monday to Friday) and weekends (Saturday and Sunday).
    - Casual users tend to increase significantly on weekends, indicating that casual riders likely use the service for leisure activities during their free time.
    - Registered users show a more consistent usage throughout the week, with a slight dip on weekends, suggesting their usage is likely more related to commuting or routine trips during workdays.
    - The boxplots and violin plots reveal that weekend usage for casual users has a wider distribution, especially on weekday and tighter in weekend. This supports the assumption that bike-sharing services are popular for weekend leisure activities.
    - For registered users, the distributions are tighter, particularly on weekdays, confirming the routine and predictable nature of their usage patterns.
    ''')

    
    

# Seasonal Analysis ###
elif analysis_type == "Seasonal Analysis":
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
