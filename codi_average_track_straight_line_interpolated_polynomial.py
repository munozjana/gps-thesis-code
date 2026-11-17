import gpxpy
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from statistics import median
import math
import scipy.interpolate
#from scipy.interpolate import CubicSpline
from datetime import datetime

def rdp(points, epsilon):
    def perpendicular_distance(point, line_start, line_end):
        point = np.array(point)
        line_start = np.array(line_start)
        line_end = np.array(line_end)
        return np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)

    def rdp_recursive(points, epsilon, start_index, end_index):
        # Find the point with the maximum distance
        max_distance = 0
        farthest_index = 0
        line_start = points[start_index]
        line_end = points[end_index]
        for i in range(start_index + 1, end_index):
            distance = perpendicular_distance(points[i], line_start, line_end)
            if distance > max_distance:
                max_distance = distance
                farthest_index = i
        
        # If the maximum distance is greater than epsilon, recursively simplify the segment
        if max_distance > epsilon:
            recursive_results_start = rdp_recursive(points, epsilon, start_index, farthest_index)
            recursive_results_end = rdp_recursive(points, epsilon, farthest_index, end_index)
            return recursive_results_start[:-1] + recursive_results_end
        
        # Otherwise, return the line segment defined by the start and end points
        else:
            return [line_start, line_end]

    if len(points) < 3:
        return points

    return rdp_recursive(points, epsilon, 0, len(points) - 1)


def get_points(trajectory, offsets):
    x_coords = [point.latitude for point in trajectory]
    y_coords = [point.longitude for point in trajectory]
    trajectory_length = len(x_coords)
    index_array = np.linspace(0, trajectory_length - 1, len(offsets)).astype(int)
    x_coords = [x_coords[i] for i in index_array]
    y_coords = [y_coords[i] for i in index_array]
    return x_coords, y_coords

def get_average_trajectory_uppp(filtered_trajectories, epsilon):
    
    if len(filtered_trajectories) < 5:
        n_best = 4
        trajectory_lengths = [len(trajectory) for trajectory in filtered_trajectories]
        distance_rank = np.argsort(trajectory_lengths)
        distance_outliers = distance_rank[n_best:]
        length_outliers = []
        filtered_indices = distance_rank[:n_best]
    else:
        distance_outliers = []
        length_outliers = []
        filtered_indices = range(len(filtered_trajectories))
    
    trajectory_list = [filtered_trajectories[i] for i in filtered_indices]
    trajectory_length = median([len(trajectory) for trajectory in trajectory_list])
    offsets = np.linspace(0, 1, int(trajectory_length) * 10)  # Convert trajectory_length to integer
    x_means = []
    y_means = []
    
  
    for t in trajectory_list:
        x_coords, y_coords = get_points(t, offsets)
        x_means.append(x_coords)
        y_means.append(y_coords)
        
    x_avg, y_avg = zip(*rdp(list(zip(np.mean(x_means, axis=0), np.mean(y_means, axis=0))), epsilon))
    
    return x_avg, y_avg

def get_average_trajectory(filtered_trajectories, epsilon):
    max_length = max(len(trajectory) for trajectory in filtered_trajectories)
    offsets = np.linspace(0, 1, max_length * 10)  # Use the maximum length for offsets
    x_means = []
    y_means = []
    
    for t in filtered_trajectories:
        x_coords, y_coords = get_points(t, offsets)
        x_means.append(x_coords)
        y_means.append(y_coords)
        

    x_avg, y_avg = zip(*rdp(list(zip(np.mean(x_means, axis=0), np.mean(y_means, axis=0))), epsilon))
    
    
    return x_avg, y_avg
    
def interpolate_points(track_points, desired_num_points):
    """
    Interpolate points for a track to achieve the desired number of points using polynomial interpolation.
    
    Args:
    - track_points: List of track points (GPXTrackPoint objects).
    - desired_num_points: The desired number of points for the track.
    
    Returns:
    - List of interpolated track points with the desired number of points.
    """
    num_points = len(track_points)
    if num_points >= desired_num_points:
        return track_points
    
    # Extract attributes from track points
    times = np.array([point.time.timestamp() for point in track_points])
    latitudes = np.array([point.latitude for point in track_points])
    longitudes = np.array([point.longitude for point in track_points])
    elevations = np.array([point.elevation for point in track_points])
    
    # Create new equally spaced time points
    new_times = np.linspace(times[0], times[-1], desired_num_points)
    
    # Create polynomial interpolators
    lat_interpolator = scipy.interpolate.PchipInterpolator(times, latitudes)
    lon_interpolator = scipy.interpolate.PchipInterpolator(times, longitudes)
    ele_interpolator = scipy.interpolate.PchipInterpolator(times, elevations)
    
    # Interpolate attributes at new time points
    new_latitudes = lat_interpolator(new_times)
    new_longitudes = lon_interpolator(new_times)
    new_elevations = ele_interpolator(new_times)
    new_times = [gpxpy.gpx.GPXTrackPoint(time=datetime.fromtimestamp(t)) for t in new_times]
    
    # Create new track points
    interpolated_points = [
        gpxpy.gpx.GPXTrackPoint(latitude=new_latitudes[i], longitude=new_longitudes[i], elevation=new_elevations[i], time=new_times[i].time)
        for i in range(desired_num_points)
    ]
    
    return interpolated_points

def make_tracks_equal_length(filtered_points):
    """
    Make all tracks within filtered_points have the same number of points.
    
    Args:
    - filtered_points: List of lists, each containing track points for a track.
    
    Returns:
    - List of lists, each containing track points with the same number of points.
    """
    max_num_points = max(len(track_points) for track_points in filtered_points)
    equal_length_tracks = []
    for track_points in filtered_points:
        equal_length_track = interpolate_points(track_points, max_num_points)
        equal_length_tracks.append(equal_length_track)
        
    return equal_length_tracks

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

    for track_points in tracks:
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


def create_average_track(filtered_points):
    # Initialize lists to store latitude and longitude values
    latitudes = []
    longitudes = []
    
    # Extract latitude and longitude values from filtered points
    for track_points in filtered_points:
        for point in track_points:
            latitudes.append(point.latitude)
            longitudes.append(point.longitude)
    
    # Compute average latitude and longitude
    average_latitude = sum(latitudes) / len(latitudes)
    average_longitude = sum(longitudes) / len(longitudes)
    
    # Create a new GPX track with averaged points
    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)
    
    # Add averaged points to the segment
    gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=average_latitude, longitude=average_longitude))
    
    # Save the new GPX file
    output_file = "average_track.gpx"
    with open(output_file, 'w') as f:
        f.write(gpx.to_xml())
    
    return output_file

def calculate_distance_to_average_track_averaging_every_point(average_track_points, equal_length_tracks):
    total_distance = 0
    num_tracks = len(equal_length_tracks)
    
    for track_points in equal_length_tracks:
        track_distance = 0
        for avg_point, track_point in zip(average_track_points, track_points):
            distance = math.sqrt((avg_point[0] - track_point.latitude) ** 2 + (avg_point[1] - track_point.longitude) ** 2)
            track_distance += distance
        total_distance += track_distance / len(track_points)  # Average distance for this track
    
    average_distance = total_distance / num_tracks
    
    return average_distance


def calculate_distance_to_average_track_nometers(average_track_points, equal_length_tracks):
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

def calculate_distance_to_average_track(average_track_points, equal_length_tracks):
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


def plot_tracks_on_map(gpx_files, min_lat, max_lat, min_lon, max_lon, average_track_x, average_track_y, equal_length_tracks):
    plt.figure(figsize=(10, 8))
    m = Basemap(projection='merc', llcrnrlat=min_lat, urcrnrlat=max_lat, llcrnrlon=min_lon, urcrnrlon=max_lon, resolution='h')
    
    m.drawmapboundary(color='white', linewidth=1.5)

    """
    for gpx_file in gpx_files:
        with open(gpx_file, 'r') as file:
            gpx = gpxpy.parse(file)
            for track in gpx.tracks:
                for segment in track.segments:
                    lons, lats = zip(*[(point.longitude, point.latitude) for point in segment.points
                                       if min_lat <= point.latitude <= max_lat and min_lon <= point.longitude <= max_lon])
                    x, y = m(lons, lats)
                    m.plot(x, y, '*:', markersize=0.5, linewidth=0.5)
    """
    
    for track_points in equal_length_tracks:
        lons, lats = zip(*[(point.longitude, point.latitude) for point in track_points
                           if min_lat <= point.latitude <= max_lat and min_lon <= point.longitude <= max_lon])
        x, y = m(lons, lats)
        m.plot(x, y, 'o-', markersize=1.6, linewidth=1)
    
     # Plot average track
    x_avg, y_avg = m(average_track_y, average_track_x)
    m.plot(x_avg, y_avg, 'o-', color='black', markersize=2, linewidth=1)
    
    
    plt.title('Interpolated tracks within Bounding Box with Average Track')
    plt.show()


def read_file_paths(file_path):
    with open(file_path, 'r') as file:
        file_paths = [line.strip() for line in file if line.strip()]
    return file_paths

if __name__ == "__main__":
    
    
    gpx_files = read_file_paths("gpx_files_1.txt")
    min_lat, max_lat, min_lon, max_lon = 41.769340, 41.770558, 2.467923, 2.471504
    
    #gpx_files = read_file_paths("gpx_files_2.txt")
    #min_lat, max_lat, min_lon, max_lon = 41.766670, 41.769058, 2.471346, 2.476506

    
    
    filtered_points = filter_points_in_section(gpx_files, min_lat, max_lat, min_lon, max_lon)
    
    equal_length_tracks = make_tracks_equal_length(filtered_points)

    distance = calculate_distance_pairs(equal_length_tracks)
    num_points = count_points(equal_length_tracks)
    print(f"Total distance between all tracks within the specified bounding box: {distance}")
    print(f"Total number of points in the bounding box: {num_points}")
    print(f"Total number of tracks: {len(equal_length_tracks)}")

    #epsilon_value = 0.00001 #rdp passa de 115 a 28
    #epsilon_value = 0.0000001 #rdp passa de 115 a 113
    epsilon_value = 0.00000001 #rdp passa de 115 a 115, tot i que encara no es fa be
    average_track_x, average_track_y = get_average_trajectory(equal_length_tracks, epsilon_value)
    print(f"Points on average track: {len(average_track_x)}")
    print(f"Length of average track: {compute_track_distance(list(zip(average_track_x, average_track_y))) } meters")

    
    result = calculate_distance_to_average_track(list(zip(average_track_x, average_track_y)), equal_length_tracks)
    print(f"Total average difference to average track: {result} meters")  
    print(f"Total average difference to average track per point: {result / len(average_track_x)} meters")  


    plot_tracks_on_map(gpx_files, min_lat, max_lat, min_lon, max_lon, average_track_x, average_track_y, equal_length_tracks)

