# --- Imports ---
import feedparser
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from textblob import TextBlob  # Import TextBlob for sentiment analysis
import geopandas as gpd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from scipy.interpolate import UnivariateSpline
import re
import pycountry

import nltk
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "data", "corpora"))

import warnings
warnings.filterwarnings("ignore")

# --- Function definition ---
## Define top 3 news sources for each country
country_sources = {
    'United States': ['reuters.com', 'bloomberg.com', 'nytimes.com'],
    'United Kingdom': ['bbc.com', 'theguardian.com', 'ft.com'],
    'Germany': ['faz.net', 'spiegel.de', 'handelsblatt.com'],
    'France': ['lemonde.fr', 'lesechos.fr'],
    'India': ['thehindu.com', 'business-standard.com'],
    'Australia': ['abc.net.au', 'afr.com'],
    'Canada': ['cbc.ca', 'financialpost.com'],
    'Brazil': ['globo.com', 'valor.globo.com'],
    'Russia': ['rt.com', 'ria.ru', 'vedomosti.ru'],
    'Spain': ['elpais.com', 'expansion.com', 'abc.es'],
    'Mexico': ['eluniversal.com.mx', 'milenio.com', 'reforma.com'],
    'Japan': ['japantimes.co.jp', 'nhk.or.jp'],
    'South Korea': ['koreaherald.com', 'koreatimes.co.kr'],
    'Italy': ['corriere.it', 'repubblica.it', 'sole24ore.com'],
    'China': ['scmp.com', 'china.org.cn'],
    'South Africa': ['news24.com', 'businesslive.co.za'],
    'Egypt': ['egyptindependent.com', 'dailynewsegypt.com'],
    'United Arab Emirates': ['thenationalnews.com', 'gulfnews.com'],
    'Saudi Arabia': ['arabnews.com', 'saudigazette.com.sa'],
    'Turkey': ['hurriyet.com.tr', 'dunya.com'],
    'Indonesia': ['jakartapost.com', 'tempo.co.id'],
    'Argentina': ['clarin.com', 'lanacion.com.ar'],
    'Nigeria': ['theguardian.ng', 'businessday.ng'],
    'Pakistan': ['dawn.com', 'thenews.com.pk'],
    'Colombia': ['eltiempo.com', 'portafolio.co'],
    'Chile': ['latercera.com', 'emol.com'],
    'Vietnam': ['vietnamnet.vn', 'tuoitrenews.vn'],
    'Philippines': ['rappler.com', 'philstar.com'],
    'Thailand': ['bangkokpost.com', 'nationthailand.com'],
    'Malaysia': ['malaysiakini.com', 'thestar.com.my'],
    'Singapore': ['straitstimes.com', 'channelnewsasia.com'],
    'New Zealand': ['stuff.co.nz', 'nzherald.co.nz'],
    'Belgium': ['lecho.be', 'rtbf.be'],
    'Netherlands': ['nos.nl', 'fd.nl'],
    'Sweden': ['svt.se', 'di.se'],
    'Finland': ['yle.fi', 'hs.fi'],
    'Norway': ['nrk.no', 'aftenposten.no'],
    'Denmark': ['dr.dk', 'berlingske.dk']
}

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

## --- Fetch news ---
# 🔍 Targeted search query
base_query = (
    'intitle:"Bayer" (pharma OR Monsanto OR agriculture OR chemical OR crop OR ESG OR lawsuit) '
    '-football -soccer -Leverkusen -match -team -bundesliga'
)

# 🌐 Build global feed URLs using your country_sources
def build_global_feed_urls(country_sources):
    query_encoded = base_query.replace(" ", "%20")
    feed_urls = []

    for country, sources in country_sources.items():
        for source in sources:
            url = f"https://news.google.com/rss/search?q={query_encoded}+site:{source}&hl=en"
            feed_urls.append((country, url))
    
    return feed_urls

def is_relevant_article(title, summary):
        text = (title + ' ' + summary).lower()
        sports_keywords = [
            'bayer 04', 'leverkusen', 'football', 'soccer', 'bundesliga',
            'match', 'goal', 'coach', 'team', 'player', 'transfer', 'cup',
            'league', 'hertha', 'berlin', 'fc', 'striker','world cup'
        ]
        return not any(keyword in text for keyword in sports_keywords)

# 📥 Fetch all news articles for the specified time window
def fetch_global_bayer_articles(start_date, country_sources):

    articles = []

    for country, sources in country_sources.items():
        for source in sources:
            url = f"https://news.google.com/rss/search?q=Bayer+site:{source}&hl=en"
            try:
                feed = feedparser.parse(url)

                if not hasattr(feed, 'entries') or not feed.entries:
                    #st.write(f"⚠️ No entries found for {source}")
                    continue

                for entry in feed.entries:
                    published_date_raw = entry.get('published', '')
                    try:
                        published_datetime = datetime.strptime(published_date_raw, '%a, %d %b %Y %H:%M:%S %Z')
                        published_date = published_datetime.date()
                    except Exception as e:
                        st.write(f"⚠️ Failed to parse published date: {published_date_raw} ({e})")
                        continue

                    if published_date < start_date:
                        continue

                    # Extract and lowercase title and summary
                    title = entry.get('title', '')
                    summary = entry.get('summary', entry.get('content', [{}])[0].get('value', title))

                    title_lc = title.lower()
                    summary_lc = summary.lower()

                    # ✅ Require "bayer" in title or summary
                    if "bayer" not in title_lc and "bayer" not in summary_lc:
                        continue

                    # ❌ Filter out sports, football, and common Bayer 04 terms
                    skip_terms = ["bayer 04", "football", "soccer", "bundesliga", "leverkusen", "bayern"]
                    if any(term in title_lc or term in summary_lc for term in skip_terms):
                        continue

                    articles.append({
                        'Title': title,
                        'Link': entry.get('link', ''),
                        'Published': published_datetime.strftime('%a, %d %b %Y %H:%M:%S GMT'),
                        'Summary': summary,
                        'Source': entry.get('source', ''),
                        'Country': country
                    })

            except Exception as e:
                st.write(f"❌ Error parsing feed for {source}: {e}")
                continue

    # Create, clean, and sort DataFrame
    df = pd.DataFrame(articles)
    df["Published"] = pd.to_datetime(df["Published"], errors="coerce")
    df = df.dropna(subset=["Published"])
    df = df.sort_values("Published")

    st.success(f"✅ {len(df)} articles loaded from {start_date} onward.")

    return df



def fetch_all_articles(start_date):
    return fetch_global_bayer_articles(start_date, country_sources)

def get_date_range():
    if st.session_state.time_window == "Custom":
        return st.session_state.custom_start, st.session_state.custom_end
    days = int(re.findall(r'\d+', st.session_state.time_window)[0])
    end = datetime.today().date()
    start = end - timedelta(days=days)
    return start, end

## --- Calculate sentiment using TextBlob (returns sentiment score as a number) ---
def get_sentiment(text):
    blob = TextBlob(text)
    return round(blob.sentiment.polarity, 2)

## --- Extract URL from the 'Source' column ---
def extract_url_from_source(source):
    if isinstance(source, dict):
        return source.get('href', None)
    return None

## --- Prep df_filtered used in Sentiment trend ---
def prepare_df_filtered(df, time_window=None):
    # Filter and transform df to df_filter
    df_filtered = df[['Title', 'Published', 'Source', 'Country','Summary']].copy()
    df_filtered['Published'] = pd.to_datetime(df_filtered['Published']).dt.strftime('%Y-%m-%d')
    df_filtered['Source'] = df_filtered['Source'].apply(extract_url_from_source)
    df_filtered['Sentiment'] = df_filtered.apply(
        lambda row: get_sentiment(row.get('Summary', '')) if row.get('Summary', '') else get_sentiment(row['Title']),axis=1
        )
    df_filtered = df_filtered[["Title", "Published", "Source", "Country", "Sentiment"]]
    return df_filtered

## --- Aggregate sentiment by country ---
def aggregate_sentiment_by_country(df_filtered):
    df_agg = df_filtered.groupby('Country').agg(
        Nr_News=('Title', 'count'),
        Avg_Sentiment=('Sentiment', 'mean')
    ).reset_index()
    df_agg['Avg_Sentiment'] = df_agg['Avg_Sentiment'].round(2)
    return df_agg

### --- Function stubs used in defining df_cloud ---
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    # Remove HTML entities (e.g., &nbsp;, &lt;, &gt;, etc.)
    text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)  # Remove HTML entities
    # Remove non-alphabetic characters (keeping spaces and Chinese characters)
    text = re.sub(r'[^a-zA-Z\s\u4e00-\u9fa5]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove specific irrelevant words (e.g., Bayer, Bayern, etc.)
    unwanted_words = ['dortmund','chosunbizchosunbiz','title', 'says', 'sbnation', 'targetblank', 'font', 'nbsp', 'football', 'soccer', 'bayer', 'bayern','bayerns', 'bundesliga','leverkusen', 'munich','chosunbiz','mainz','could']
    
    return ' '.join([word for word in text.split() if word not in unwanted_words])

def process_text(text):
    # Tokenize and POS tag
    blob = TextBlob(text)
    nouns = [word for word, tag in blob.tags if tag.startswith('NN')]
    # Remove stopwords
    nouns = [word for word in nouns if word not in STOPWORDS]
    return ' '.join(nouns)

## --- Prepare df_cloud for word cloud ---
def prepare_wordcloud_df(df, start_date, end_date):
    from datetime import datetime

    # Convert start_date and end_date to datetime for safe comparison
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.max.time())

    df_cloud = df.copy()
    df_cloud['Published'] = pd.to_datetime(df_cloud['Published'])

    df_cloud = df_cloud[
        (df_cloud['Published'] >= start_date) &
        (df_cloud['Published'] <= end_date)
    ]

    df_cloud['Cleaned_Summary'] = df_cloud['Summary'].apply(clean_text)
    df_cloud['Processed_Summary'] = df_cloud['Cleaned_Summary'].apply(process_text)

    return df_cloud

# Helper to convert country names to ISO Alpha-3
def get_country_code(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None

def plot_world_map(df_agg):
    # Add ISO Alpha-3 codes for Plotly
    df_agg['iso_alpha'] = df_agg['Country'].apply(get_country_code)
    df_agg = df_agg.dropna(subset=['iso_alpha'])

    # Create the choropleth map
    fig = px.choropleth(
        df_agg,
        locations="iso_alpha",
        color="Avg_Sentiment",
        hover_name="Country",
        hover_data={"Nr_News": True, "Avg_Sentiment": True},
        color_continuous_scale=px.colors.diverging.RdYlGn,
        range_color=[-1, 1],
        labels={"Avg_Sentiment": "Avg. Sentiment"},
        title=""
    )

    # Transparent + modern with country borders
    fig.update_geos(
        projection_type="natural earth",
        scope="world",
        showcountries=True,
        showcoastlines=True,
        showframe=True
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(
            title="Avg. Sentiment",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["Very Neg", "-0.5", "Neutral", "+0.5", "Very Pos"]
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Plot sentiment trend ---
def plot_sentiment_trend_per_country(df_filtered):
    # Ensure datetime format
    df_filtered['Published'] = pd.to_datetime(df_filtered['Published'])

    # Group by Country and Date to get daily average sentiment
    df_trend = df_filtered.groupby(['Country', 'Published']).agg(
        Avg_Sentiment=('Sentiment', 'mean')
    ).reset_index()

    df_trend = df_trend.sort_values(by=['Country', 'Published'])

    # Get countries that have actual sentiment data
    countries_with_data = df_trend['Country'].unique()

    for country in countries_with_data:
        country_data = df_trend[df_trend['Country'] == country]

        if country_data.empty:
            continue  # Skip if no data

        fig = px.line(
            country_data,
            x='Published',
            y='Avg_Sentiment',
            title=f"{country} - Sentiment Trend",
            labels={'Published': 'Date', 'Avg_Sentiment': 'Average Sentiment'},
            line_shape='spline',
            markers=True  # ✅ Show individual sentiment points
        )

        fig.update_layout(
            height=300,
            yaxis=dict(range=[-1, 1]),
            xaxis=dict(tickformat='%b %d', tickangle=-45),
            margin=dict(l=0, r=0, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

# --- Show word cloud ---
def show_word_cloud(df, start_date, end_date):
    df = df.copy()
    df['Published'] = pd.to_datetime(df['Published'])
    df = df[(df['Published'] >= pd.to_datetime(start_date)) & (df['Published'] <= pd.to_datetime(end_date))]

    # Clean and process text
    df['Cleaned_Summary'] = df['Summary'].apply(clean_text)
    df['Nouns_Only'] = df['Cleaned_Summary'].apply(process_text)
    combined_text = ' '.join(df['Nouns_Only'].dropna().astype(str))

    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(combined_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


# --- Session State ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# Optional: time window control (used by df_filtered)
if "time_window" not in st.session_state:
    st.session_state.time_window = "last 7 days"  # or allow user input

# --- Sidebar Navigation ---
with st.sidebar:
   
    if st.button("Home"):
        st.session_state.current_page = "home"
    if st.button("Sentiment per Country"):
        st.session_state.current_page = "sentiment_country"
    if st.button("Sentiment World Map"):
        st.session_state.current_page = "sentiment_map"
    if st.button("Sentiment Trend"):
        st.session_state.current_page = "sentiment_trend"
    if st.button("Word Cloud"):
        st.session_state.current_page = "word_cloud"

    time_window_options = ["Last 7 days", "Last 30 days", "Last 90 days", "Custom"]
    selected_window = st.selectbox("Select time window", time_window_options)

    # Initialize in session_state
    if "time_window" not in st.session_state:
        st.session_state.time_window = "Last 7 days"

    if "custom_start" not in st.session_state:
        st.session_state.custom_start = datetime.today() - timedelta(days=7)
    if "custom_end" not in st.session_state:
        st.session_state.custom_end = datetime.today()

    # Show date inputs if custom selected
    if selected_window == "Custom":
        start_date = st.date_input("Start date", st.session_state.custom_start)
        end_date = st.date_input("End date", st.session_state.custom_end)

        # Save to session state
        if start_date != st.session_state.custom_start:
            st.session_state.custom_start = start_date
            st.session_state.df = pd.DataFrame()  # trigger update

        if end_date != st.session_state.custom_end:
            st.session_state.custom_end = end_date
            st.session_state.df = pd.DataFrame()  # trigger update
    else:
        if selected_window != st.session_state.time_window:
            st.session_state.time_window = selected_window
            st.session_state.df = pd.DataFrame()  # trigger update

# --- Main Page Logic ---
if st.session_state.current_page == "home":
    st.title("Welcome to Bayer News Sentiment Analysis Project")
    st.subheader("Introduction")

    st.write("""
             This app collects and analyzes news articles related to **Bayer** from leading global media sources, starting from 2025.
             It provides an interactive view of how Bayer is portrayed in the press through the following features:
             - Sentiment analysis of each article based on its summary
             - Geographic visualization of average sentiment scores by country
             - Time-series trend showing how media sentiment evolves over time
             - Word cloud generation to highlight the most frequently reported topics

            *Note: This analysis reflects media narratives, not direct public or governmental attitudes.*
             """)
    
    st.write("""
             Click the buttons in the navigation sidebar to explore the different sections of the app and view the content.
             """)
    
    st.subheader("About me")
    st.write("""
             I’m **Tianhemei Wang**, a Senior Product Designer with a strong passion for building user-centric experiences in tech. 
             Over the years, I’ve honed my skills in designing intuitive web applications, focusing on making complex processes simple and accessible.

             As part of my personal growth, I’ve recently embarked on learning **Artificial Intelligence (AI)**. 
             Although my background is in design, I believe AI is an exciting field with immense potential, and I wanted to prove to myself that I could successfully learn and apply AI skills. 
             This project is a way for me to foster my ability to grasp and implement AI concepts despite my design background.

             **Note**: The app is using real data to show sentiment trends, word clouds, and sentiment on the world map. However, due to limited funding (I’m working on this project with my own interest and budget) and the fact that I’m still learning, there are definitely areas that can be improved. There are opportunities for future enhancements as I continue developing my AI skills.

             If you’d like to connect or learn more about my work and journey in product design and AI, feel free to reach out!
            
             **Contact**: 
             - Email: hemei.wang@bayer.com
             """)


elif st.session_state.current_page in ["sentiment_country", "sentiment_map", "sentiment_trend", "word_cloud"]:
    if st.session_state.df.empty:
        start_date, _ = get_date_range()
        with st.spinner("Fetching news..."):
            df = fetch_all_articles(start_date)
        st.session_state.df = df

# --- Sentiment per Country ---
if st.session_state.current_page == "sentiment_country":
    df = st.session_state.df
    start_date, end_date = get_date_range()

    st.markdown(f"🗓️ **News from `{start_date.strftime('%Y-%m-%d')}` to `{end_date.strftime('%Y-%m-%d')}`**")

    if df.empty:
        st.warning("⚠️ No news articles found for the selected time window.")
    elif not all(col in df.columns for col in ['Title', 'Published', 'Source', 'Country', 'Summary']):
        st.error("⚠️ Fetched data is missing required columns. Please try refreshing or check feed source.")
    else:
               
        df_filtered = prepare_df_filtered(df, st.session_state.time_window)
        df_agg = aggregate_sentiment_by_country(df_filtered)

        st.title("Aggregated Media Sentiment per Country")
        st.markdown("Sentiment scores range from -1 to 1, where -1 indicates very negative sentiment, 0 is neutral, and 1 indicates very positive sentiment.")

        st.info(f"📊 A total of **{len(df_filtered)}** news articles from the selected date range were analyzed for country-level sentiment.")

        st.dataframe(df_agg)

    st.title("Fetched News – Original")
    st.markdown("Click on any news title to open the original article:")

    df_display = df.copy()

    # Clean up 'Source' column (extract href if it's a dict)
    df_display["Source"] = df_display["Source"].apply(
        lambda s: s.get("href") if isinstance(s, dict) and "href" in s else s
    )

    # Create a new DataFrame with selected and formatted columns
    df_cleaned = pd.DataFrame({
        "Title": df_display.apply(
            lambda row: f"[{row['Title']}]({row['Link']})", axis=1
        ),
        "Published": df_display["Published"],
        "Source": df_display["Source"],
        "Country": df_display["Country"]
    })

    # Display the cleaned DataFrame as markdown
    st.markdown(df_cleaned.to_markdown(index=False), unsafe_allow_html=True)


# --- Sentiment World Map ---
elif st.session_state.current_page == "sentiment_map":
    
    df = st.session_state.df

    start_date, end_date = get_date_range()
    st.markdown(f"🗓️ **News from `{start_date.strftime('%Y-%m-%d')}` to `{end_date.strftime('%Y-%m-%d')}`**")

    df_filtered = prepare_df_filtered(st.session_state.df, st.session_state.time_window)
    df_agg = aggregate_sentiment_by_country(df_filtered)

    st.title("Media Sentiment World Map")
    st.markdown("This world map visualizes the average sentiment of news articles mentioning Bayer across medien in different countries within the selected time window.")

    st.info(f"📊 A total of **{len(df_filtered)}** news articles from the selected date range were analyzed for world map.")

    plot_world_map(df_agg)

# --- Sentiment Trend ---
elif st.session_state.current_page == "sentiment_trend":
    
    df = st.session_state.df

    start_date, end_date = get_date_range()
    st.markdown(f"🗓️ **News from `{start_date.strftime('%Y-%m-%d')}` to `{end_date.strftime('%Y-%m-%d')}`**")

    df_filtered = prepare_df_filtered(df, st.session_state.time_window)

    st.title("Media Sentiment Trend Over Time")

    st.info(f"📊 A total of **{len(df_filtered)}** news articles from the selected date range were analyzed for country-level sentiment.")

    plot_sentiment_trend_per_country(df_filtered)


# --- Word Cloud ---
elif st.session_state.current_page == "word_cloud":
    df = st.session_state.df
    start_date, end_date = get_date_range()

    st.markdown(f"🗓️ **News from `{start_date.strftime('%Y-%m-%d')}` to `{end_date.strftime('%Y-%m-%d')}`**")

    if df.empty:
        with st.spinner("Fetching news..."):
            df = fetch_all_articles(start_date)
            st.session_state.df = df

    # ✅ Prepare word cloud dataframe
    df_cloud = prepare_wordcloud_df(df, start_date, end_date)

    st.title("Word Cloud for Topics")

    # ✅ Summary
    st.info(f"☁️ A total of **{len(df_cloud)}** news articles from the selected date range were processed to generate the word cloud.")

    show_word_cloud(df, start_date, end_date)