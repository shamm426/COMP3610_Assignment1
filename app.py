import polars as pl 
import duckdb
import requests
import os 
import streamlit as st
import plotly.express as px
import altair as alt

st.set_page_config( page_title='NYC Yellow Taxi Dashboard - January 2024', page_icon='taxi', layout='wide')

url_tripData = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
url_lookupTable = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

destination_dir = "data/raw"

file_nameData = "yellow_tripdata_2024-01.parquet"
file_nameTable = "taxi_zone_lookup.csv"

os.makedirs(destination_dir, exist_ok=True)

file_path1 = os.path.join(destination_dir, file_nameData)
file_path2 = os.path.join(destination_dir, file_nameTable)

response1 = requests.get(url_tripData)
response2 = requests.get(url_lookupTable)

with open(file_path1, 'wb') as file:
        file.write(response1.content)
with open(file_path2, 'wb') as file:
        file.write(response2.content)

df1 = pl.read_parquet(url_tripData)

df_clean = df1.drop_nulls(subset=[
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "fare_amount"
])

df_clean = df_clean.filter((pl.col('trip_distance') > 0) & (pl.col('fare_amount') > 0) & (pl.col('fare_amount') < 500))

df_clean = df_clean.filter((pl.col('tpep_pickup_datetime')) < pl.col('tpep_dropoff_datetime'))


derived_columns = df_clean.with_columns([((pl.col('tpep_dropoff_datetime') - pl.col('tpep_pickup_datetime')).dt.total_seconds() / 60).alias('trip_duration_minutes'),

                  (pl.col('trip_distance') / ((pl.col('tpep_dropoff_datetime') - pl.col('tpep_pickup_datetime')).dt.total_seconds() / 3600)).alias('trip_speed_mph'),

                  (pl.col('tpep_pickup_datetime').dt.hour()).alias('pickup_hour'),

                  (pl.col('tpep_pickup_datetime').dt.weekday()).alias('pickup_weekday')
                ])


st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=[
        derived_columns.select(pl.col("tpep_pickup_datetime").min()).item().date(),
        derived_columns.select(pl.col("tpep_pickup_datetime").max()).item().date()
    ]
)

hour_range = st.sidebar.slider("Hour Range", 0, 23, (0, 23))

payment_types = derived_columns.select("payment_type").unique().to_series().to_list()

selected_payments = st.sidebar.multiselect(
    "Payment Type",
    payment_types,
    default=payment_types
)

filtered_df = derived_columns.filter(
    (pl.col("pickup_hour") >= hour_range[0]) &
    (pl.col("pickup_hour") <= hour_range[1]) &
    (pl.col("payment_type").is_in(selected_payments))
)

filtered_pd = filtered_df.to_pandas()

if filtered_pd.empty:
    st.warning("No data available for selected filters.")
    st.stop()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Trips", len(filtered_pd))
col2.metric("Average Fare", f"${filtered_pd['fare_amount'].mean():.2f}")
col3.metric("Total Revenue", f"${filtered_pd['total_amount'].sum():,.2f}")
col4.metric("Avg Distance", f"{filtered_pd['trip_distance'].mean():.2f} mi")
col5.metric("Avg Duration", f"{filtered_pd['trip_duration_minutes'].mean():.2f} min")

zones = pl.read_csv("data/raw/taxi_zone_lookup.csv")

zones_pd = zones.to_pandas()

filtered_pd = filtered_pd.merge(
    zones_pd,
    left_on="PULocationID",
    right_on="LocationID",
    how="left"
)

top_zones = (
    filtered_pd.groupby("Zone")
    .size()
    .reset_index(name="trip_count")
    .sort_values("trip_count", ascending=False)
    .head(10)
)

fig1 = alt.Chart(top_zones).mark_bar().encode(
    y=alt.Y("Zone:O", sort='-x', title="Pickup Zone"),
    x=alt.X("trip_count:Q", title="Number of Trips"),
    tooltip=["Zone", "trip_count"]
).properties(
    title="Top 10 Busiest Pickup Zones",
    height=450
)

st.altair_chart(bar_chart, use_container_width=True)

st.markdown("""
Midtown Manhattan zones dominate pickup activity, reflecting commercial and tourism concentration.
""")

avg_fare_hour = (
    filtered_pd.groupby("pickup_hour")["fare_amount"]
    .mean()
    .reset_index()
)

fig2 = px.line(avg_fare_hour,
               x="pickup_hour",
               y="fare_amount",
               title="Average Fare by Hour")

st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
Fares peak during late evening and early morning hours.
Midday hours show more stable pricing.
""")

filtered_small = filtered_pd[filtered_pd["trip_distance"] < 20]

fig3 = alt.Chart(filtered_small).mark_bar().encode(
    x=alt.X(
        "trip_distance:Q",
        bin=alt.Bin(maxbins=40),
        title="Trip Distance (miles)"
    ),
    y=alt.Y("count()", title="Number of Trips")
).properties(
    title="Trip Distance Distribution (0–20 miles)",
    height=450
)

st.altair_chart(histogram, use_container_width=True)

st.markdown("""
Most trips are short-distance (under 5 miles).
Long-distance trips are comparativelyrare.
""")


payment_counts = (
    filtered_pd["payment_type"]
    .value_counts()
    .reset_index()
)

fig4 = px.pie(payment_counts,
              names="payment_type",
              values="count",
              title="Payment Type Breakdown")

st.plotly_chart(fig4, use_container_width=True)

st.markdown("""
Credit card payments dominate transactions.
Cash usage represents a smaller share of total trips.
""")


heatmap_data = (
    filtered_pd.groupby(["pickup_weekday", "pickup_hour"])
    .size()
    .reset_index(name="trip_count")
)

fig5 = px.density_heatmap(
    heatmap_data,
    x="pickup_hour",
    y="pickup_weekday",
    z="trip_count",
    title="Trips by Day and Hour"
)

st.plotly_chart(fig5, use_container_width=True)

st.markdown("""
Weekday rush hours show peak demand during morning and evening commuting periods.
Weekend late-night activity is significantly higher than weekday nights.
""")