import feedparser
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from textblob import TextBlob  # Import TextBlob for sentiment analysis
import geopandas as gpd
import plotly.express as px
from wordcloud import WordCloud
from scipy.interpolate import UnivariateSpline
import re

# Define top 3 news sources for each country
country_sources = {
    'United States': ['cnn.com', 'nytimes.com', 'reuters.com'],
    'United Kingdom': ['bbc.com', 'theguardian.com', 'telegraph.co.uk'],
    'Germany': ['faz.net', 'spiegel.de', 'deutschewelle.com'],
    'France': ['lemonde.fr', 'france24.com', 'lefigaro.fr'],
    'India': ['thehindu.com', 'timesofindia.indiatimes.com', 'ndtv.com'],
    'Australia': ['abc.com.au', 'theage.com.au', 'smh.com.au'],
    'Canada': ['cbc.ca', 'globesandmail.com', 'torontostar.com'],
    'Brazil': ['globo.com', 'estadão.com.br', 'folha.uol.com.br'],
    'Russia': ['rt.com', 'ria.ru', 'sovietnews.net'],
    'Spain': ['elpais.com', 'abc.es', 'marca.com'],
    'Mexico': ['eluniversal.com.mx', 'milenio.com', 'reforma.com'],
    'Japan': ['japantimes.co.jp', 'nhk.or.jp', 'asahi.com'],
    'South Korea': ['koreaherald.com', 'chosun.com', 'joongang.co.kr'],
    'Italy': ['corriere.it', 'repubblica.it', 'ilmessaggero.it'],
    'China': ['china.org.cn', 'scmp.com', 'globaltimes.cn'],
    'South Africa': ['timeslive.co.za', 'sabcnews.com', 'news24.com'],
    'Egypt': ['almasryalyoum.com', 'egyptindependent.com', 'dailynewsegypt.com'],
    'United Arab Emirates': ['thenationalnews.com', 'khaleejtimes.com', 'gulfnews.com'],
    'Saudi Arabia': ['arabnews.com', 'saudigovt.com', 'alriyadh.com'],
    'Turkey': ['hurriyet.com.tr', 'milliyet.com.tr', 'sozcu.com.tr'],
    'Indonesia': ['jakartapost.com', 'thejakartaglobe.com', 'tempo.co.id'],
    'Argentina': ['clarin.com', 'infobae.com', 'lanacion.com.ar'],
    'Nigeria': ['punchng.com', 'theguardian.ng', 'vanguardngr.com'],
    'Pakistan': ['dawn.com', 'tribune.com.pk', 'thenews.com.pk'],
    'Colombia': ['eltiempo.com', 'elnuevodia.com.co', 'colombiareports.com'],
    'Chile': ['emol.com', 'latercera.com', 'biobiochile.cl'],
    'Vietnam': ['vietnamnet.vn', 'tuoitrenews.vn', 'dantri.com.vn'],
    'Philippines': ['rappler.com', 'abs-cbn.com', 'philstar.com'],
    'Thailand': ['bangkokpost.com', 'nationthailand.com', 'thaipbsworld.com'],
    'Malaysia': ['theedgemarkets.com', 'malaysiakini.com', 'thestar.com.my'],
    'Singapore': ['straitstimes.com', 'todayonline.com', 'channelnewsasia.com'],
    'New Zealand': ['stuff.co.nz', 'nzherald.co.nz', 'tvnz.co.nz'],
    'Belgium': ['lecho.be', 'rtbf.be', 'demorgen.be'],
    'Netherlands': ['nos.nl', 'telegraaf.nl', 'fd.nl'],
    'Sweden': ['svt.se', 'aftonbladet.se', 'expressen.se'],
    'Finland': ['yle.fi', 'hs.fi', 'ilta-sanomat.fi'],
    'Norway': ['nrk.no', 'vg.no', 'aftenposten.no'],
    'Denmark': ['dr.dk', 'berlingske.dk', 'jyllands-posten.dk']
}

# DEFINE FUNCTION TO FETCH NEWS
def fetch_bayer_articles_from_source(source_url, country):
    url = f"https://news.google.com/news/rss/search/section/q/Bayer+{source_url}/{source_url}?hl=en"
    feed = feedparser.parse(url)
    articles = []
    
    for entry in feed.entries:
        published_date = entry.get('published', '')
        try:
            published_datetime = datetime.strptime(published_date, '%a, %d %b %Y %H:%M:%S %Z')
        except ValueError:
            continue
        
        if published_datetime.year >= 2025:
            # Exclude football-related news based on title or summary
            title = entry.get('title', '').lower()
            summary = entry.get('summary', '').lower()
            if "football" in title or "soccer" in title or "bayer 04" in title or "soccer" in summary:
                continue  # Skip this article
            
            # Safely access Summary (use content or title if missing)
            summary = entry.get('summary', entry.get('content', [{}])[0].get('value', entry.get('title', '')))

            article = {
                'Title': entry.get('title', ''),
                'Link': entry.get('link', ''),
                'Published': published_date,
                'Summary': summary,
                'Authors': entry.get('author', ''),
                'Tags': entry.get('tags', ''),
                'Updated': entry.get('updated', ''),
                'Content': entry.get('content', ''),
                'Language': entry.get('language', ''),
                'Source': entry.get('source', ''),
                'ID': entry.get('id', ''),
                'Country': country  # Add the country for each article
            }
            articles.append(article)
    return articles

# Function to calculate sentiment using TextBlob (returns sentiment score as a number)
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Sentiment polarity score (-1 to 1)
    return round(sentiment, 2)  # Round to 2 decimal places

# Fetch Bayer-related articles from multiple countries
articles_all = []
for country, sources in country_sources.items():
    for source in sources[:3]:  # Limit to 3 sources per country
        articles_all.extend(fetch_bayer_articles_from_source(source, country))

# Create DataFrame from fetched articles
df = pd.DataFrame(articles_all)

# st.write(df)

###################### GOOD SO FAR


# Create df_copy by copying df and retaining only the Title, Published, Source, Country columns
df_copy = df[['Title', 'Published', 'Source', 'Country']].copy()

# Format the Published column to remove weekday and time info (keep only the date)
df_copy['Published'] = pd.to_datetime(df_copy['Published']).dt.strftime('%Y-%m-%d')

# Extract URL from the 'Source' column
def extract_url_from_source(source):
    if isinstance(source, dict):  # Check if the source is a dictionary
        return source.get('href', None)  # Return the 'href' value (URL)
    return None  # If not a dictionary, return None

# Apply the function to the 'Source' column to extract the URLs
df_copy['Source'] = df_copy['Source'].apply(extract_url_from_source)

# Safely calculate sentiment based on Summary, or fallback to Title if Summary is empty or NaN
df_copy['Sentiment'] = df_copy.apply(
    lambda row: get_sentiment(row.get('Summary', '')) if row.get('Summary', '') else get_sentiment(row['Title']),
    axis=1
)

# st.write(df_copy)

####################################### GOOD SO FAR

# Check if 'selected_section' exists in session state, if not, initialize it to "Home"
if 'selected_section' not in st.session_state:
    st.session_state.selected_section = "Home"  # Default section

# Home button: Default content
home_button = st.sidebar.button("Home", key="home_button")
if home_button:
    st.session_state.selected_section = "Home"  # Set section to Home when button is clicked

# Set default dates
date_filter_end = datetime.today()
date_filter_start = datetime.today() - timedelta(days=7)

# Sentiment per country button
sentiment_per_country_button = st.sidebar.button("Sentiment per country", key="sentiment_per_country")
if sentiment_per_country_button:
    st.session_state.selected_section = "Sentiment per country"

# Sentiment world map button
sentiment_world_map_button = st.sidebar.button("Sentiment world map", key="sentiment_world_map")
if sentiment_world_map_button:
    st.session_state.selected_section = "Sentiment world map"

# Sentiment trend button
sentiment_trend_button = st.sidebar.button("Sentiment trend", key="sentiment_trend")
if sentiment_trend_button:
    st.session_state.selected_section = "Sentiment trend"

# Word cloud button
word_cloud_button = st.sidebar.button("Word cloud", key="word_cloud")
if word_cloud_button:
    st.session_state.selected_section = "Word cloud"

time_window_option = st.sidebar.selectbox(
    "Choose a time period for analysis",
    options=["Last 7 days", "Last 30 days", "Last 90 days", "All Time", "Custom Date Range"]
)

# Filter data based on selected time window
if time_window_option == "Last 7 days":
    date_filter_start = datetime.today() - timedelta(days=7)
elif time_window_option == "Last 30 days":
    date_filter_start = datetime.today() - timedelta(days=30)
elif time_window_option == "Last 90 days":
    date_filter_start = datetime.today() - timedelta(days=90)
elif time_window_option == "All Time":
    date_filter_start = datetime.min  # No start date, so include all articles
else:  # Custom Date Range
    start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=30))  # Now in the sidebar
    end_date = st.sidebar.date_input("End Date", datetime.today())  # Now in the sidebar
    date_filter_start = datetime.combine(start_date, datetime.min.time())
    date_filter_end = datetime.combine(end_date, datetime.max.time())

################################### GOOD SO FAR

# Filter the df_copy based on the time window
df_copy_filtered = df_copy[df_copy['Published'].apply(pd.to_datetime) >= date_filter_start]

# st.write(df_copy_filtered)

# Filter the df_copy based on the time window
df_copy_filtered = df_copy[df_copy['Published'].apply(pd.to_datetime) >= date_filter_start]

# df_agg: stores filtered news per country, has Country, Nr. Articles, Avg. Sentiment 
df_agg = df_copy_filtered.groupby('Country').agg(
    num_articles=('Title', 'count'),  # Count the number of articles per country
    avg_sentiment=('Sentiment', 'mean')  # Calculate the average sentiment per country
).reset_index()

# Rename columns to display custom names
df_agg = df_agg.rename(columns={
    'num_articles': 'Nr. News',
    'avg_sentiment': 'Avg. Sentiment'
})

# Round the avg_sentiment to 2 decimal places
df_agg['Avg. Sentiment'] = df_agg['Avg. Sentiment'].round(2)

################################### GOOD SO FAR

# Content display based on selected section
if st.session_state.selected_section == "Home":
    st.title("Welcome to Bayer News Sentiment Analysis Project")
    st.subheader("Introduction")
    st.write("""
        This app collects news articles related to **Bayer** from multiple global sources starting from **2025**.
             """)

    st.write("""
    The app then performs the following tasks:
    - Calculates the sentiment of each article based on its summary.
    - Displays sentiment scores on a world map.
    - Generates a sentiment trend over time.
    - Creates a word cloud for the most frequently reported topics.
""")
    
    st.write("""
    Click the buttons in the navigation sidebar to explore the different sections of the app and view the content.
""")
    
    st.subheader("About me")
    st.write("""
    I’m **Hemei Wang**, a Senior Product Designer with a strong passion for building user-centric experiences in tech. 
    Over the years, I’ve honed my skills in designing intuitive web applications, focusing on making complex processes simple and accessible.

    As part of my personal growth, I’ve recently embarked on learning **Artificial Intelligence (AI)**. 
    Although my background is in design, I believe AI is an exciting field with immense potential, and I wanted to prove to myself that I could successfully learn and apply AI skills. 
    This project is a way for me to foster my ability to grasp and implement AI concepts despite my design background.

    **Note**: The app is using real data to show sentiment trends, word clouds, and sentiment on the world map. However, due to limited funding (I’m working on this project with my own interest and budget) and the fact that I’m still learning, there are definitely areas that can be improved. There are opportunities for future enhancements as I continue developing my AI skills.

    If you’d like to connect or learn more about my work and journey in product design and AI, feel free to reach out!
    
    **Contact**: 
    - Email: hemei.wang@bayer.com
""")

################################## GOOD SO FAR

# Display content for sentiment per country
elif st.session_state.selected_section == "Sentiment per country":
    st.subheader("Sentiment per Country")
    st.write(f"For news from {date_filter_start.strftime('%Y-%m-%d')} to {date_filter_end.strftime('%Y-%m-%d')}")
    
    # # Filter the df_copy based on the time window
    # df_copy_filtered = df_copy[df_copy['Published'].apply(pd.to_datetime) >= date_filter_start]

    # # Group by Country and calculate the average sentiment for each country
    # # df_agg: stores filtered news per country, has Country, Nr. Articles, Avg. Sentiment 
    # df_agg = df_copy_filtered.groupby('Country').agg(
    #     num_articles=('Title', 'count'),  # Count the number of articles per country
    #     avg_sentiment=('Sentiment', 'mean')  # Calculate the average sentiment per country
    # ).reset_index()

    # # Rename columns to display custom names
    # df_agg = df_agg.rename(columns={
    #     'num_articles': 'Nr. News',
    #     'avg_sentiment': 'Avg. Sentiment'
    # })

    # # Round the avg_sentiment to 2 decimal places
    # df_agg['Avg. Sentiment'] = df_agg['Avg. Sentiment'].round(2)

    st.write(df_agg)

################################### GOOD SO FAR

# Display content for sentiment world map
elif st.session_state.selected_section == "Sentiment world map":
   st.subheader("Sentiment World Map")
   st.write(f"For news from {date_filter_start.strftime('%Y-%m-%d')} to {date_filter_end.strftime('%Y-%m-%d')}")

   # Load the Natural Earth shapefile manually
   shapefile_path = r"C:\Users\GDJUX\AppData\Local\Programs\Python\Python313\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"  # Update this path
   world = gpd.read_file(shapefile_path)
 
   # Merge the world map with the sentiment data
   world['Country'] = world['NAME']
   df_agg = df_agg.rename(columns={'Country': 'Country'})  # Ensure column names match for merging
   world = world.merge(df_agg[['Country', 'Avg. Sentiment']], on='Country', how='left')

   # Plotting the map with Plotly
   fig = px.choropleth(world,
                       locations='Country',
                       locationmode='country names',
                       color='Avg. Sentiment',
                       hover_name='Country',
                       color_continuous_scale='Cividis',  # Default color scale (red to green)
                       labels={'Avg. Sentiment': 'Average Sentiment'},
                       #title="Sentiment Towards Bayer by Country"
                       )

    # Update map's layout to default settings
   fig.update_geos(
       showcoastlines=True, 
       coastlinecolor="Black", 
       showland=True, 
       landcolor="whitesmoke",  # Default land color
       )

    # Resize the map to make it larger
   fig.update_layout(
       autosize=True,
       height=600,  # Default height for better visibility
       width=1200,  # Default width for better visibility
       )

    # Display the updated map in Streamlit
   st.plotly_chart(fig)

#####################################################################


# # Display content for sentiment trend  
# elif st.session_state.selected_section == "Sentiment trend":
#     st.subheader("Sentiment Trend")
#     st.write(f"For news from {date_filter_start.strftime('%Y-%m-%d')} to {date_filter_end.strftime('%Y-%m-%d')}")

# # Display content for word cloud  
# elif st.session_state.selected_section == "Word cloud":
#     st.subheader("Word Cloud")
#     st.write(f"For news from {date_filter_start.strftime('%Y-%m-%d')} to {date_filter_end.strftime('%Y-%m-%d')}")
 


######################################################################################



# # Filter the df_copy based on the time window
# df_copy_filtered = df_copy[df_copy['Published'].apply(pd.to_datetime) >= date_filter_start]

# # Group by Country and calculate the average sentiment for each country
# # df_agg: stores filtered news per country, has Country, Nr. Articles, Avg. Sentiment 
# df_agg = df_copy_filtered.groupby('Country').agg(
#     num_articles=('Title', 'count'),  # Count the number of articles per country
#     avg_sentiment=('Sentiment', 'mean')  # Calculate the average sentiment per country
# ).reset_index()

# # Rename columns to display custom names
# df_agg = df_agg.rename(columns={
#     'num_articles': 'Nr. News',
#     'avg_sentiment': 'Avg. Sentiment'
# })

# # Round the avg_sentiment to 2 decimal places
# df_agg['Avg. Sentiment'] = df_agg['Avg. Sentiment'].round(2)

# ## NAVI SIDEBAR

# if st.sidebar.button('Home'):
#     st.session_state.page = "home"










# if st.sidebar.button('Sentiment per Country'):
#     st.session_state.page = "sentiment_country"

# if st.sidebar.button('Sentiment Trend'):
#     st.session_state.page = "sentiment_trend"

# if st.sidebar.button('Sentiment World Map'):
#     st.session_state.page = "sentiment_map"

# if st.sidebar.button('Word Cloud'):
#     st.session_state.page = "word_cloud"

# if st.sidebar.button('Filtered News'):
#     st.session_state.page = 'filtered_news'

# if st.sidebar.button('Project Background'):
#     st.session_state.page = "background"

# if 'page' not in st.session_state:
#     st.session_state.page = "home"  # Default page






# # Sidebar Navigation













### DEFINE PAGE CONTENTS

# if st.session_state.page == "home":        
#     # Sentiment per country section

#     st.title("Bayer News Sentiment")

#     #################################
#     st.subheader("About the app")

#     st.write("""
#              This app collects news articles related to **Bayer** from multiple global sources starting from **2025**.
#              """)

#     st.write("""
#         The app performs the following tasks:
#         - Calculates the sentiment of each article based on its summary.
#         - Displays sentiment scores on a world map.
#         - Generates a sentiment trend over time.
#         - Creates a word cloud for the most frequently reported topics.
#     """)

#     st.write("""
#         Click the buttons in the navigation sidebar to explore the different sections of the app and view the content.
#     """)

#     ###############################

#     st.subheader("About me")

#     st.write("""
#     Hi, I’m **Hemei Wang**, a Senior Product Designer in UX field with a strong passion for creating user-centric experiences. 
#     Over the years, I’ve honed my skills in designing intuitive web app and mobile apps, focusing on making complex processes simple and accessible.

#     As part of my personal growth, I’ve recently embarked on learning **Artificial Intelligence (AI)**. 
#     Although my background is in design, I believe AI is an exciting field with immense potential, and I wanted to prove to myself that I could successfully learn and apply AI skills. 
#     This project is a way for me to further develop my ability to grasp and implement AI concepts despite my design background.

#     **Note**: The app is using real data to show sentiment trends, word clouds, and sentiment on the world map. However, due to limited funding (I’m working on this project with my own interest and budget) and the fact that I’m still learning, there are definitely areas that can be improved. There are opportunities for future enhancements as I continue developing my AI skills.
             
#     If you'd like to learn more about my journey, or if you're interested in collaborating, feel free to reach out!

#     **Contact**: 
#     - Email: hemei.wang@bayer.com
# """)


# # Define sections based on selected page
# if st.session_state.page == "sentiment_country":        
#     # Sentiment per country section
#     st.title("Sentiment Per Country")
#     st.write("Here is the aggregated sentiment per country")  # Your content for sentiment per country
#     st.write(df_agg)

# # Function to display sentiment trend
# def display_sentiment_trend(df_copy_filtered):
#     #st.title("Sentiment Trend Over Time")

#     # List of countries for user selection
#     countries = df_copy_filtered['Country'].unique()
#     selected_countries = st.multiselect("Select Countries", options=countries, default=countries)

#     # Filter data for selected countries
#     sentiment_trend = df_copy_filtered[df_copy_filtered['Country'].isin(selected_countries)]

#     # Plotting the trend
#     fig, ax = plt.subplots(figsize=(12, 6))

#     for country in selected_countries:
#         country_data = sentiment_trend[sentiment_trend['Country'] == country]
#         # Convert dates to numeric values for spline fitting
#         x_vals = mdates.date2num(country_data['Published'])
#         y_vals = country_data['Sentiment']
        
#         # Apply spline smoothing
#         spline = UnivariateSpline(x_vals, y_vals, s=0.5)  # Adjust 's' to control smoothness
#         smoothed_y = spline(x_vals)  # Generate the smoothed sentiment values

#         ax.plot(country_data['Published'], smoothed_y, label=country)

#     ax.set_title("Sentiment Trend per Country", fontsize=14)
#     ax.set_xlabel("Date", fontsize=12)
#     ax.set_ylabel("Sentiment Score", fontsize=12)
    
#     ax.xaxis.set_major_locator(mdates.WeekdayLocator())
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     ax.xaxis.set_minor_locator(mdates.DayLocator())
#     plt.xticks(rotation=45)
    
#     plt.grid(True)
#     plt.tight_layout()
    
#     # Display the legend
#     ax.legend(title="Countries", loc='upper left')
    
#     st.pyplot(fig)
#     st.title("Sentiment Trend Over Time")

#     # List of countries for user selection
#     countries = df_copy_filtered['Country'].unique()
#     selected_countries = st.multiselect("Select Countries", options=countries, default=countries)

#     # Filter data for selected countries
#     sentiment_trend = df_copy_filtered[df_copy_filtered['Country'].isin(selected_countries)]

#     # Plotting the trend
#     fig, ax = plt.subplots(figsize=(12, 6))

#     for country in selected_countries:
#         country_data = sentiment_trend[sentiment_trend['Country'] == country]
#         # Convert dates to numeric values for spline fitting
#         x_vals = mdates.date2num(country_data['Published'])
#         y_vals = country_data['Sentiment']
        
#         # Apply spline smoothing
#         spline = UnivariateSpline(x_vals, y_vals, s=0.5)  # Adjust 's' to control smoothness
#         smoothed_y = spline(x_vals)  # Generate the smoothed sentiment values

#         ax.plot(country_data['Published'], smoothed_y, label=country)

#     ax.set_title("Sentiment Trend per Country", fontsize=14)
#     ax.set_xlabel("Date", fontsize=12)
#     ax.set_ylabel("Sentiment Score", fontsize=12)
    
#     ax.xaxis.set_major_locator(mdates.WeekdayLocator())
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     ax.xaxis.set_minor_locator(mdates.DayLocator())
#     plt.xticks(rotation=45)
    
#     plt.grid(True)
#     plt.tight_layout()
    
#     # Display the legend
#     ax.legend(title="Countries", loc='upper left')
    
#     st.pyplot(fig)    

# Check which page user selected


# ##########################################################################################

# ### SENTIMENT TREND
# if st.session_state.page == "sentiment_trend":
    
#     # Sentiment per country section
#     st.title("Sentiment Trend")
#     st.write("Here is the aggregated sentiment develops overtime...")  # Your content for sentiment per country
#     # Display sentiment trend with smoothed lines
#     display_sentiment_trend(df_copy_filtered)
   
# if st.session_state.page == "sentiment_map":
#     # Sentiment world map section
#     st.title("Sentiment World Map")
#     st.write("Here is the sentiment world map...")  # Your content for the map

#     # Load the Natural Earth shapefile manually
#     shapefile_path = r"C:\Users\GDJUX\AppData\Local\Programs\Python\Python313\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"  # Update this path
#     world = gpd.read_file(shapefile_path)

#     # Check the column names of the world GeoDataFrame
#     # st.write("Shapefile columns:", world.columns)

#     # Merge the world map with the sentiment data
#     world['Country'] = world['NAME']
#     df_agg = df_agg.rename(columns={'Country': 'Country'})  # Ensure column names match for merging
#     world = world.merge(df_agg[['Country', 'Avg. Sentiment']], on='Country', how='left')

#     st.subheader("Sentiment by Country")

#     # Plotting the map with Plotly
#     fig = px.choropleth(world,
#                         locations='Country',
#                         locationmode='country names',
#                         color='Avg. Sentiment',
#                         hover_name='Country',
#                         color_continuous_scale='Cividis',  # Default color scale (red to green)
#                         labels={'Avg. Sentiment': 'Average Sentiment'},
#                         #title="Sentiment Towards Bayer by Country"
#                         )

#     # Update map's layout to default settings
#     fig.update_geos(
#         showcoastlines=True, 
#         coastlinecolor="Black", 
#         showland=True, 
#         landcolor="whitesmoke",  # Default land color
#     )

#     # Resize the map to make it larger
#     fig.update_layout(
#         autosize=True,
#         height=600,  # Default height for better visibility
#         width=1200,  # Default width for better visibility
#     )

#     # Display the updated map in Streamlit
#     st.plotly_chart(fig)

# if st.session_state.page == "word_cloud":
#     # Word cloud section
#     st.title("Word Cloud")
#     st.write("Here is the word cloud...")  # Your content for the word cloud

#     # Function to clean and preprocess the text
#     STOPWORDS = set([
#         'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 
#         'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
#         'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
#         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
#         'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
#         'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 
#         'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
#         'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
#         'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
#     ])

#     # Function to clean text (remove non-alphabetic characters and HTML tags)
#     def clean_text(text):
#         # Remove HTML tags
#         text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
#         # Remove HTML entities (e.g., &nbsp;, &lt;, &gt;, etc.)
#         text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)  # Remove HTML entities
#         # Remove non-alphabetic characters (keeping spaces and Chinese characters)
#         text = re.sub(r'[^a-zA-Z\s\u4e00-\u9fa5]', '', text)
#         # Convert to lowercase
#         text = text.lower()
#         # Remove specific irrelevant words (e.g., Bayer, Bayern, etc.)
#         unwanted_words = ['title', 'says', 'sbnation', 'targetblank', 'font', 'nbsp', 'football', 'soccer', 'bayer', 'bayern','bayerns', 'bundesliga','leverkusen', 'munich','chosunbiz','mainz','could']
#         text = ' '.join([word for word in text.split() if word not in unwanted_words])
#         return text

#     def process_text(text):
#         words = text.split()
#         words = [word for word in words if word not in STOPWORDS]
#         return ' '.join(words)

#     # Generate Word Cloud
#     def generate_wordcloud(text_data):
#         wordcloud = WordCloud(
#             stopwords=STOPWORDS,
#             background_color='white',
#             width=800,
#             height=400,
#             max_words=200,
#             contour_color='black',
#             contour_width=1
#         ).generate(' '.join(text_data))

#         # Display the word cloud using Streamlit
#         plt.figure(figsize=(10, 5))
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.axis('off')
#         st.pyplot(plt)

#     # Filter df based on the selected date range
#     df_cloud = df.copy()
#     df_cloud['Published'] = pd.to_datetime(df_cloud['Published'])  # Ensure Published is in datetime format
#     df_cloud = df_cloud[(df_cloud['Published'] >= date_filter_start) & (df_cloud['Published'] <= date_filter_end)]

#     # Clean and process the 'Summary' column
#     df_cloud['Cleaned_Summary'] = df_cloud['Summary'].apply(lambda x: clean_text(x))
#     df_cloud['Processed_Summary'] = df_cloud['Cleaned_Summary'].apply(lambda x: process_text(x))

#     # Generate word cloud from filtered and processed summaries
#     generate_wordcloud(df_cloud['Processed_Summary'])


# ### FILTERED NEWS
# if st.session_state.page == "filtered_news":
#     st.title("News in selected time window")
#     st.write(df_copy_filtered)


# if st.session_state.page == "background":
#     # Background section
#     st.title("Project Background")
#     st.write("""
#         This is a sentiment analysis project I developed to track and analyze news sentiment about Bayer.
#         Through this project, I gained experience in data collection, sentiment analysis, and visualizations.
#     """)  # Your project background content
