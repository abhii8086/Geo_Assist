from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
from sklearn.cluster import KMeans
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

app = Flask(__name__)

data_path = "Mumbai_NaviMumbai.csv"

def load_and_clean_data():
    df = pd.read_csv(data_path)
    df = df[['Name', 'Categories', 'Latitude', 'Longitude', 'Address', 'Image_URL']].dropna()
    return df

def cluster_amenities(df, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])
    return df

def filter_by_radius(df, center, radius_km=5):
    return df[df.apply(lambda row: geodesic(center, (row['Latitude'], row['Longitude'])).km <= radius_km, axis=1)]

def generate_map(df, search_location=None):
    if search_location:
        map_center = search_location
    else:
        map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    
    my_map = folium.Map(location=map_center, zoom_start=12, tiles='cartodbpositron')
    category_colors = {
        'College': 'orange',
        'Hospital': 'green',
        'Garden': 'blue',
        'Hotel': 'red'
    }
    
    marker_cluster = MarkerCluster().add_to(my_map)
    for _, row in df.iterrows():
        color = category_colors.get(row['Categories'], 'gray')
        popup_content = f"""
        <b>{row['Name']}</b><br>
        Category: {row['Categories']}<br>
        Address: {row['Address']}<br>
        <img src='{row['Image_URL']}' width='200px'>
        """
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color=color)
        ).add_to(marker_cluster)
    
    # Highlight the searched location with a blue circle
    if search_location:
        folium.Circle(
            location=search_location,
            radius=5000,  # Adjust radius to 5km
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.3
        ).add_to(my_map)
    
    return my_map._repr_html_()

def get_coordinates(location):
    geolocator = Nominatim(user_agent="geoapi")
    loc = geolocator.geocode(location)
    return [loc.latitude, loc.longitude] if loc else None

@app.route('/')
def index():
    df = load_and_clean_data()
    amenities = df['Categories'].unique().tolist()
    return render_template('index.html', amenities=amenities)

@app.route('/get_map', methods=['POST'])
def get_map():
    df = load_and_clean_data()
    user_amenities = request.json.get('amenities', [])
    search_location = request.json.get('location', None)
    
    search_coords = get_coordinates(search_location) if search_location else None
    if not search_coords:
        return jsonify({'error': 'Invalid location entered'}), 400
    
    filtered_df = df[df['Categories'].isin(user_amenities)]
    filtered_df = filter_by_radius(filtered_df, search_coords, radius_km=5)
    
    if filtered_df.empty:
        return jsonify({'error': 'No locations found for selected amenities within the search radius'}), 400
    
    clustered_df = cluster_amenities(filtered_df)
    map_html = generate_map(clustered_df, search_coords)
    return jsonify({'map': map_html})

if __name__ == '__main__':
    app.run(debug=True)
