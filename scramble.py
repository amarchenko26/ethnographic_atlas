#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:18:01 2023

1) Identify the pairs of adjacent groups in the Murdock map with the largest 
difference in average luminosity in 1992-2020 (use the luminosity I sent you).

2) What are the strongest correlates from the Ethnographic Atlas correlates of 
these bilateral differences?

3) Let’s Scramble Africa. Recreate placebo African countries (one idea is to use 
fishnet, another is to throw random points and generate Thiessen around these 
points, etc.). Remember to keep the number of the resulting polygons/countries to 48. 
Redo the analysis of the Scramble. Test whether groups that are split by these 
pseudo-borders have more conflict (use the data from ACLED). 
What is your interpretation of the pattern found?

@author: anyamarchenko
"""

import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt


# Set the base directory and change working directory
base_directory = "/Users/anyamarchenko/Documents/Github/ethnographic_atlas"
os.chdir(base_directory)


#############################################################################
# TASK 1
#############################################################################

# Function to get mean luminosity
def get_luminosity_mean(geom, raster):
    try:
        out_image, out_transform = mask(raster, [geom], crop=True)
        return out_image[out_image > 0].mean()
    except:
        return None

# Function to process a single year
def process_year(year, countries_gdf, tribes_gdf):
    # Load the raster for the given year
    raster_file = f'data/more_night_lights/DMSP{year}_bltcfix.tif'
    with rasterio.open(raster_file) as raster:
        # Rest of the processing as before...
        intersected_gdf = gpd.overlay(tribes_gdf, countries_gdf, how='intersection')
        split_tribes = intersected_gdf.groupby('NAME').filter(lambda x: len(x) > 1)
        luminosity_data = []
        for tribe, group in split_tribes.groupby('NAME'):
            countries = group['COUNTRY'].unique()
            if len(countries) > 1:
                luminosity_means = [get_luminosity_mean(geom, raster) for geom in group.geometry]
                luminosity_diff = abs(luminosity_means[0] - luminosity_means[1])
                luminosity_data.append({
                    'year': year,
                    'tribe_name': tribe,
                    'country_1': countries[0],
                    'country_2': countries[1],
                    'mean_lights_1': luminosity_means[0],
                    'mean_lights_2': luminosity_means[1],
                    'luminosity_diff': luminosity_diff
                })
        # Create DataFrame for the year
        year_df = pd.DataFrame(luminosity_data)
        return year_df

def plot_histogram(data, title, xlabel, ylabel, filename):
    plt.figure(figsize=(15, 8))
    data.plot(kind='bar')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)    
    plt.style.use('bmh')
    plt.tight_layout()
    plt.savefig(f'output/{filename}', format='png')
    plt.show()


# Load the shapefiles
countries_gdf = gpd.read_file('data/DCW_Africa.shp')
tribes_gdf = gpd.read_file('data/Tribal_Map_Africa.shp')
ethnologue_gdf = gpd.read_file('data/Ethnologue_16_shapefile/langa_no_overlap_biggest_clean.shp')


# Main processing loop for each year
all_years_df = pd.DataFrame()
for year in range(1992, 2021):
    year_df = process_year(year, countries_gdf, tribes_gdf)
    all_years_df = pd.concat([all_years_df, year_df], ignore_index=True)


tribe_counts = all_years_df['tribe_name'].value_counts().sort_values(ascending=True)
plot_histogram(data = tribe_counts, title="Tribes with Biggest Difference in Night Lights Across Splits (1992-2020)", xlabel='Tribe', ylabel='Number of times tribe split has large lights diff', filename='tribe_hist.png')


#############################################################################
# TASK 2
#############################################################################

ea_df = pd.read_csv('data/ethnographic_atlas_africa.csv')
ea_df.rename(columns={'atlas': 'tribe_name', 'h_g': 'subsistence_hunt_gather', 'v3': 'subsistence_fish', 'v4': 'subsistence_pastoralism', 'v5': 'subsistence agriculture', 'v8': 'domestic_organization', 'v28': "intensity_ag", 'v31':'size_communities', 'v33':"heirarchy",'v34':"high_gods","v39":"animals_plow", "v42":"subsistence_economy", "v54":"sex_diff_in_ag", "v66":"class_strat", "v98":"language_fam"}, inplace=True)


# Step 1: Merge DataFrames
merged_df = pd.merge(all_years_df, ea_df, on=['tribe_name'])

# Step 2: Handle Missing Values
for col in merged_df.columns:
    if merged_df[col].dtype == 'object':  # Categorical
        merged_df[col].fillna(merged_df[col].mode()[0], inplace=True)
    else:  # Numerical
        merged_df[col].fillna(merged_df[col].median(), inplace=True)


# Columns to be removed
columns_to_remove = ['year', 'tribe_name', 'country_1', 'country_2', 'mean_lights_1',
                     'mean_lights_2', 'oid', 'continent_EA']

# Remove the specified columns
merged_df = merged_df.drop(columns=columns_to_remove)

# Step 3: Encode Categorical Variables
categorical_cols = ['subsistence_hunt_gather',
       'subsistence_fish', 'subsistence_pastoralism',
       'subsistence agriculture', 'domestic_organization', 'intensity_ag',
       'size_communities', 'heirarchy', 'high_gods', 'animals_plow',
       'subsistence_economy', 'sex_diff_in_ag', 'class_strat']  # list all categorical columns
merged_df = pd.get_dummies(merged_df, columns=categorical_cols)

# Step 4: Compute Correlations
correlation_results = merged_df.corr()['luminosity_diff']  # Assuming 'luminosity_diff' is the column name

# Print correlation results
print(correlation_results)


#############################################################################
# TASK 3
############################################################################
from shapely.geometry import Point
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import voronoi_diagram
from shapely import ops
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

# Step 1: Generate Random Points in Africa
def generate_random_points_in_africa(africa_shape, n_points=48):
    minx, miny, maxx, maxy = africa_shape.bounds
    points = []
    while len(points) < n_points:
        point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if africa_shape.contains(point):
            points.append(point)
    return points

# Load Africa's shapefile or use a suitable representation of Africa's boundaries
africa_shape = countries_gdf.unary_union  # Assuming a single, continuous shape for Africa

# Generate random points
random_points = generate_random_points_in_africa(africa_shape)

# Function to create Thiessen (Voronoi) polygons
def create_thiessen_polygons(points, africa_shape, crs):
    # Create Voronoi diagram
    x = [point.x for point in points]
    y = [point.y for point in points]
    vor = voronoi_diagram(Polygon(zip(x, y)))

    # Convert Voronoi regions to GeoDataFrame
    polygons = [geom for geom in vor.geoms if africa_shape.contains(geom)]
    thiessen_polygons = gpd.GeoDataFrame(geometry=polygons, crs=crs)

    return thiessen_polygons

# Assuming africa_gdf and tribes_gdf are already loaded and have the same CRS
thiessen_polygons = create_thiessen_polygons(random_points, africa_shape, countries_gdf.crs)

# Intersect Thiessen polygons with tribes
intersected_gdf = gpd.overlay(tribes_gdf, thiessen_polygons, how='intersection')


import chardet

# Detect encoding
with open('data/african_conflicts.csv', 'rb') as file:
    result = chardet.detect(file.read())
    encoding = result['encoding']

# Read the file with the detected encoding
conflicts_df = pd.read_csv('data/african_conflicts.csv', encoding=encoding)

conflicts_df = conflicts_df.replace('#REF!', np.nan)  # Replace '#REF!' with NaN

# Convert LONGITUDE and LATITUDE to numeric values
conflicts_df['LONGITUDE'] = pd.to_numeric(conflicts_df['LONGITUDE'], errors='coerce')
conflicts_df['LATITUDE'] = pd.to_numeric(conflicts_df['LATITUDE'], errors='coerce')

# Create the GeoDataFrame
conflicts_gdf = gpd.GeoDataFrame(conflicts_df, geometry=gpd.points_from_xy(conflicts_df.LONGITUDE, conflicts_df.LATITUDE))
conflicts_gdf = conflicts_gdf.set_crs(tribes_gdf.crs)


# intersect conflicts and thiessen
conflicts_in_polygons = gpd.sjoin(conflicts_gdf, thiessen_polygons)


import pandas as pd

# Step 1: Aggregate Conflict Data
# Count the number of conflicts in each Thiessen polygon
conflict_count = conflicts_in_polygons.groupby('index_right').size()
conflict_count.name = 'conflict_count'

print('done1')
# Join this count back to the Thiessen polygons

thiessen_polygons.reset_index(drop=True, inplace=True)

# Merge the conflict_count into thiessen_polygons
thiessen_polygons = thiessen_polygons.merge(conflict_count.to_frame(), left_index=True, right_index=True, how='left')

print('done2')

# Step 2: Determine Split and Non-Split Tribes
# Identify split tribes (tribes that intersect with more than one polygon)
split_tribes = intersected_gdf['NAME'].value_counts()
split_tribes = split_tribes[split_tribes > 1].index.tolist()

# Add a column to tribes_gdf to indicate split status
tribes_gdf['split'] = tribes_gdf['NAME'].isin(split_tribes)

# Step 3: Calculate Mean Conflicts for Split and Non-Split Groups
# Join the split status back to the intersected_gdf
intersected_gdf = intersected_gdf.merge(tribes_gdf[['NAME', 'split']], on='NAME')

# Join conflict counts to intersected_gdf
intersected_gdf = intersected_gdf.merge(thiessen_polygons[['conflict_count']], left_on='index_right', right_index=True, how='left')

# Compute mean conflicts for split and non-split tribes
mean_conflicts = intersected_gdf.groupby('split')['conflict_count'].mean()

# Print the results
print(mean_conflicts)



