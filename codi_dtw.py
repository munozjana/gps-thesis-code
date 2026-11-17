import gpxpy
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from statistics import median
from fastdtw import fastdtw
import math
import time
 
def calculate_average_track_dtw(filtered_points):
    # Extract latitude and longitude values from filtered points
    trajectories = []
    for track_points in filtered_points:
        trajectory = np.array([[point.latitude, point.longitude] for point in track_points])
        trajectories.append(trajectory)

    # Align trajectories using DTW
    aligned_trajectories = []
    max_length = max(len(trajectory) for trajectory in trajectories)
    for i in range(len(trajectories)):
        aligned_trajectory = []
        for j in range(len(trajectories)):
            if i == j:
                aligned_trajectory.append(trajectories[i])  # Add original trajectory
            else:
                # Align trajectories using DTW
                _, path = fastdtw(trajectories[i], trajectories[j]) # _ is distane
                aligned_trajectory.append(np.array([trajectories[j][p[1]] for p in path]))

        aligned_trajectories.append(aligned_trajectory)

     # Pad aligned trajectories to ensure they have the same length
    max_len = max(len(traj) for aligned_traj in aligned_trajectories for traj in aligned_traj)
    aligned_trajectories_padded = []
    for aligned_trajectory in aligned_trajectories:
        padded_trajectories = []
        for traj in aligned_trajectory:
            padded_traj = np.pad(traj, ((0, max_len - len(traj)), (0, 0)), mode='constant', constant_values=np.nan)
            padded_trajectories.append(padded_traj)
        aligned_trajectories_padded.append(padded_trajectories)

        aligned_trajectories_padded.append(padded_trajectories)

    # Calculate average trajectory
    average_trajectory = np.nanmean(aligned_trajectories_padded, axis=(0, 1))

    return average_trajectory[:, 0], average_trajectory[:, 1]
 

def analizar_archivo_gpx(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        gpx = gpxpy.parse(archivo)

        # Conteo de puntos de seguimiento
        num_puntos = sum(1 for _ in gpx.walk(only_points=True))

        # Estadísticas de altitud
        altitudes = [punto.elevation for track in gpx.tracks for segment in track.segments for punto in segment.points]
        altitud_minima = min(altitudes)
        altitud_maxima = max(altitudes)

        # Cálculo de la distancia total
        distancia_total = sum(segment.length_3d() for track in gpx.tracks for segment in track.segments)

    print(f"Archivo GPX: {ruta_archivo}")
    print(f"Número de puntos de seguimiento: {num_puntos}")
    print(f"Altitud mínima: {altitud_minima} metros")
    print(f"Altitud máxima: {altitud_maxima} metros")
    print(f"Distancia total recorrida: {distancia_total} metros")

def filter_points_in_section(gpx_files, min_lat, max_lat, min_lon, max_lon):
    filtered_points = []
    
    for gpx_file in gpx_files:
        with open(gpx_file, 'r') as file:
            gpx = gpxpy.parse(file)
            
            # Filter points within the specified bounding box
            track_points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    track_points.extend([point for point in segment.points
                                         if min_lat <= point.latitude <= max_lat
                                         and min_lon <= point.longitude <= max_lon])
            
            # Append filtered points to the result list
            filtered_points.append(track_points)
    
    return filtered_points

def count_points(tracks):
    num_points = []

    for track_points in filtered_points:
        num_points.append(len(track_points))
    return num_points
    

def calculate_distance_pairs(filtered_points):
    total_distance = 0
   
    for i in range(len(filtered_points)):
        for j in range(i + 1, len(filtered_points)):
            track1_points = filtered_points[i]
            track2_points = filtered_points[j]
            
            for point1 in track1_points:
                min_distance = float('inf')  # Initialize min_distance to infinity
                for point2 in track2_points:
                    distance = point1.distance_3d(point2)
                    if distance < min_distance:
                        min_distance = distance
                total_distance += min_distance
    
    return total_distance

def calculate_distance_to_average_track(average_track_points, equal_length_tracks):
    total_distance_to_average = 0
    num_tracks = len(equal_length_tracks)
    
    for track_points in equal_length_tracks:
        track_distance = 0
        for avg_point, track_point in zip(average_track_points, track_points):
            distance = math.sqrt((avg_point[0] - track_point.latitude) ** 2 + (avg_point[1] - track_point.longitude) ** 2)
            track_distance += distance
        total_distance_to_average += track_distance  # Total distance for this track
    
    average_distance_to_average = total_distance_to_average / num_tracks
    
    return average_distance_to_average

def calculate_distance_to_average_track_meters(average_track_points, equal_length_tracks):
    # Earth's radius in meters (approximation)
    earth_radius = 6371000
    
    total_distance_to_average = 0
    num_tracks = len(equal_length_tracks)
    
    for track_points in equal_length_tracks:
        track_distance = 0
        for avg_point, track_point in zip(average_track_points, track_points):
            # Convert degrees to radians for latitude and longitude
            avg_lat_rad = math.radians(avg_point[0])
            avg_lon_rad = math.radians(avg_point[1])
            track_lat_rad = math.radians(track_point.latitude)
            track_lon_rad = math.radians(track_point.longitude)
            
            # Haversine formula to calculate distance
            dlat = track_lat_rad - avg_lat_rad
            dlon = track_lon_rad - avg_lon_rad
            a = math.sin(dlat / 2) ** 2 + math.cos(avg_lat_rad) * math.cos(track_lat_rad) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = earth_radius * c
            
            track_distance += distance
        
        total_distance_to_average += track_distance  # Total distance for this track
    
    average_distance_to_average = total_distance_to_average / num_tracks
    
    return average_distance_to_average

def plot_tracks_on_map(gpx_files, min_lat, max_lat, min_lon, max_lon, average_track_x, average_track_y):
    plt.figure(figsize=(10, 8))
    m = Basemap(projection='merc', llcrnrlat=min_lat, urcrnrlat=max_lat, llcrnrlon=min_lon, urcrnrlon=max_lon, resolution='h')
    
    m.drawmapboundary(color='white', linewidth=1.5)

    for gpx_file in gpx_files:
        with open(gpx_file, 'r') as file:
            gpx = gpxpy.parse(file)
            for track in gpx.tracks:
                for segment in track.segments:
                    lons, lats = zip(*[(point.longitude, point.latitude) for point in segment.points
                                       if min_lat <= point.latitude <= max_lat and min_lon <= point.longitude <= max_lon])
                    x, y = m(lons, lats)
                    m.plot(x, y, 'o-', markersize=2, linewidth=1)

    # Plot average track
    x_avg, y_avg = m(average_track_y, average_track_x)
    m.plot(x_avg, y_avg, 'o-', color='black', markersize=2, linewidth=1)
    
    """
    for i, aligned_trajectory in enumerate(aligned_trajectories):
        for j, trajectory in enumerate(aligned_trajectory):
            lons, lats = zip(*[(point[1], point[0]) for point in trajectory])
            x, y = m(lons, lats)
            m.plot(x, y, '*', color='red', linewidth=0.5)
    
    for i, aligned_trajectory in enumerate(aligned_trajectories_padded):
        for j, trajectory in enumerate(aligned_trajectory):
            lons, lats = zip(*[(point[1], point[0]) for point in trajectory])
            x, y = m(lons, lats)
            m.plot(x, y, '*', color='blue', linewidth=0.5)
    """

    plt.title('Tracks within Bounding Box with Average Track using DTW')
    plt.show()


def read_file_paths(file_path):
    with open(file_path, 'r') as file:
        file_paths = [line.strip() for line in file if line.strip()]
    return file_paths

def compute_track_distance(track_points):
    # Earth's radius in meters (approximation)
    earth_radius = 6371000
    
    total_distance = 0
    
    for i in range(len(track_points) - 1):
        # Convert degrees to radians for latitude and longitude
        lat1_rad = math.radians(track_points[i][0])
        lon1_rad = math.radians(track_points[i][1])
        lat2_rad = math.radians(track_points[i+1][0])
        lon2_rad = math.radians(track_points[i+1][1])
        
        # Haversine formula to calculate distance between two points
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = earth_radius * c
        
        total_distance += distance
    
    return total_distance


if __name__ == "__main__":
    
    
    #gpx_files = read_file_paths("gpx_files_1.txt")
    #min_lat, max_lat, min_lon, max_lon = 41.769340, 41.770558, 2.467923, 2.471504
    
    #gpx_files = read_file_paths("gpx_files_2.txt")
    #min_lat, max_lat, min_lon, max_lon = 41.766670, 41.769058, 2.471346, 2.476506
    
    gpx_files = read_file_paths("gpx_files_3.txt")
    min_lat, max_lat, min_lon, max_lon = 41.781467, 41.802012, 2.383469, 2.398930
    
    filtered_points = filter_points_in_section(gpx_files, min_lat, max_lat, min_lon, max_lon)
    
    num_points = count_points(filtered_points)
    print(f"Total number of points in the bounding box: {num_points}")
    print(f"Total number of tracks: {len(filtered_points)}")

    start_time = time.time()
    
    average_track_x, average_track_y = calculate_average_track_dtw(filtered_points)
    print(f"Length of average track: {compute_track_distance(list(zip(average_track_x, average_track_y))) } meters")
    print(f"Points on average track: {len(average_track_x)}")
        
    end_time = time.time()
    
    result = calculate_distance_to_average_track(list(zip(average_track_x, average_track_y)), filtered_points)
    print(f"Total difference to average track: {result} units")  
    print(f"Total average difference to average track per point: {result / len(average_track_x)} units")  
    
    result_in_meters = calculate_distance_to_average_track_meters(list(zip(average_track_x, average_track_y)), filtered_points)
    print(f"Total average difference to average track: {result_in_meters} meters")  
    print(f"Total average difference to average track per point: {result_in_meters / len(average_track_x)} meters")  
    
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    plot_tracks_on_map(gpx_files, min_lat, max_lat, min_lon, max_lon, average_track_x, average_track_y)

