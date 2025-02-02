import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import streamlit as st
import urllib
import numpy as np

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = '#0E1117'
plt.rcParams['axes.facecolor'] = '#0E1117'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

# Dataset paths (update these with your local paths)
DATASET_PATH = r'C:\Users\Rafli\Documents\Data-Analysis Dicoding\Dashboard\all_data_df.csv'
GEOLOCATION_PATH = r'C:\Users\Rafli\Documents\Data-Analysis Dicoding\Dashboard\geolocation.csv'
LOGO_PATH = r'C:\Users\Rafli\Documents\Data-Analysis Dicoding\Dashboard\png logo data analysis.jpg'

# Load data
datetime_cols = ["order_approved_at", "order_delivered_carrier_date", 
                 "order_delivered_customer_date", "order_estimated_delivery_date",
                 "order_purchase_timestamp", "shipping_limit_date"]

all_df = pd.read_csv(DATASET_PATH)
all_df.sort_values(by="order_approved_at", inplace=True)
all_df.reset_index(inplace=True)

geolocation = pd.read_csv(GEOLOCATION_PATH)
data = geolocation.drop_duplicates(subset='customer_unique_id')  

for col in datetime_cols:
    all_df[col] = pd.to_datetime(all_df[col])

min_date = all_df["order_approved_at"].min()
max_date = all_df["order_approved_at"].max()

# Classes
class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def create_daily_orders_df(self):
        daily_orders_df = self.df.resample(rule='D', on='order_approved_at').agg({
            "order_id": "nunique",
            "payment_value": "sum"
        }).reset_index()
        return daily_orders_df.rename(columns={
            "order_id": "order_count",
            "payment_value": "revenue"
        })
    
    def create_sum_spend_df(self):
        return self.df.resample(rule='D', on='order_approved_at').agg({
            "payment_value": "sum"
        }).reset_index().rename(columns={"payment_value": "total_spend"})

    def create_sum_order_items_df(self):
        sum_order_items_df = self.df.groupby("product_category_name_english")["product_id"].count().reset_index()
        return sum_order_items_df.rename(columns={"product_id": "product_count"}
                                    ).sort_values(by='product_count', ascending=False)

    def review_score_df(self):
        review_scores = self.df['review_score'].value_counts().sort_values(ascending=False)
        return review_scores, review_scores.idxmax()

    def create_bystate_df(self):
        bystate_df = self.df.groupby(by="customer_state").customer_id.nunique().reset_index()
        bystate_df = bystate_df.rename(columns={"customer_id": "customer_count"}
                                  ).sort_values(by='customer_count', ascending=False)
        return bystate_df, bystate_df.loc[bystate_df['customer_count'].idxmax(), 'customer_state']

    def create_order_status(self):
        order_status_df = self.df["order_status"].value_counts().sort_values(ascending=False)
        return order_status_df, order_status_df.idxmax()

class BrazilMapPlotter:
    def __init__(self, data, plt, mpimg, st):
        self.data = data
        self.plt = plt
        self.mpimg = mpimg
        self.st = st

    def plot(self):
        # Use direct URL for Brazil map image
        url = 'https://i.pinimg.com/originals/3a/0c/e1/3a0ce18b3c842748c255bc0aa445ad41.jpg'
        
        # Load image from URL
        with urllib.request.urlopen(url) as response:
            brazil_img = self.mpimg.imread(response, format='jpg')
        
        # Create plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        self.data.plot(kind="scatter", 
                      x="geolocation_lng", 
                      y="geolocation_lat",
                      ax=ax,
                      alpha=0.3,
                      s=0.3,
                      c='#FF4B4B')
        ax.axis('off')
        ax.imshow(brazil_img, extent=[-73.98283055, -33.8, -33.75116944, 5.4])
        self.st.pyplot(fig)

# Sidebar
with st.sidebar:
    st.image(LOGO_PATH, width=120)
    st.write("""### Date Range Filter""")
    start_date, end_date = st.date_input(
        label="Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

# Main content
main_df = all_df[(all_df["order_approved_at"] >= str(start_date)) & 
                 (all_df["order_approved_at"] <= str(end_date))]

function = DataAnalyzer(main_df)
map_plot = BrazilMapPlotter(data, plt, mpimg, st)

# Generate DataFrames
daily_orders_df = function.create_daily_orders_df()
sum_spend_df = function.create_sum_spend_df()
sum_order_items_df = function.create_sum_order_items_df()
review_score, common_score = function.review_score_df()
state, most_common_state = function.create_bystate_df()
order_status, common_status = function.create_order_status()

# Dashboard
st.title("📈 E-Commerce Analytics Dashboard")
st.markdown("##")

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Orders", f"{daily_orders_df['order_count'].sum():,}")
col2.metric("Total Revenue", f"R${daily_orders_df['revenue'].sum():,.2f}")
col3.metric("Average Review Score", f"{review_score.mean():.1f}/5.0")

st.markdown("---")

# Order Trends
st.subheader("📆 Daily Orders Trend")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=daily_orders_df, x="order_approved_at", y="order_count",
             linewidth=2.5, color='#FF4B4B', marker='o', markersize=8)
plt.title("Daily Order Volume", fontsize=16, pad=20)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Orders", fontsize=12)
st.pyplot(fig)

# Customer Spending
st.subheader("💸 Customer Spending Patterns")
fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(sum_spend_df.set_index('order_approved_at').index, 0, sum_spend_df.set_index('order_approved_at')["total_spend"], color='#00CCCC', alpha=0.4, linewidth=2)
plt.title("Daily Customer Spending", fontsize=16, pad=20)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Total Spend (R$)", fontsize=12)
st.pyplot(fig)

# Product Performance
st.subheader("📦 Product Category Performance")
fig, ax = plt.subplots(figsize=(14, 8))
sns.barplot(data=sum_order_items_df.head(10), x="product_count", y="product_category_name_english",
            palette="rocket_r", edgecolor='black')
plt.title("Top 10 Product Categories", fontsize=16, pad=20)
plt.xlabel("Number of Items Sold", fontsize=12)
plt.ylabel("Category", fontsize=12)
st.pyplot(fig)

# Review Analysis
st.subheader("⭐ Customer Reviews Analysis")
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis_r(np.linspace(0.2, 0.7, len(review_score)))
sns.barplot(x=review_score.index, y=review_score.values, palette=colors, edgecolor='black')
plt.title("Review Score Distribution", fontsize=16, pad=20)
plt.xlabel("Rating", fontsize=12)
plt.ylabel("Count", fontsize=12)
for i, v in enumerate(review_score.values):
    ax.text(i, v + 20, str(v), ha='center', va='bottom', color='white', fontsize=10)
st.pyplot(fig)

# Customer Demographics
st.subheader("🌍 Customer Geography")
tab1, tab2 = st.tabs(["State Distribution", "Geographic Density"])

with tab1:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=state.head(10), x="customer_count", y="customer_state",
                palette="viridis", edgecolor='black')
    plt.title("Top 10 States by Customer Count", fontsize=16, pad=20)
    plt.xlabel("Number of Customers", fontsize=12)
    plt.ylabel("State", fontsize=12)
    st.pyplot(fig)

with tab2:
    map_plot.plot()
    with st.expander("Analysis Insight"):
        st.markdown("""
        - **Southeast/South Dominance:** Majority of customers located in developed regions
        - **Capital Cities Focus:** Higher density around São Paulo, Rio de Janeiro
        - **Regional Potential:** Opportunities for expansion in northern regions
        """)

st.markdown("---")
st.caption("E-Commerce Analytics Dashboard | Created with Streamlit")