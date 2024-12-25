import requests
import numpy as np
import math
from PIL import Image
from io import BytesIO
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
import pymysql
import pickle
import time



port = "http://13.54.3.150"

size = 128

def get_db_connection():

    # Cố gắng kết nối tới cơ sở dữ liệu MySQL
    return pymysql.connect(
        host="junction.proxy.rlwy.net",
        user="root",
        port=18619,
        password="HkVMMSGHEnBLTbodCPyiJKMYVdFAezDK",
        database="railway"
    )
    
conn =  get_db_connection()      
cursor = conn.cursor()

def train(k):
    time.sleep(3)
    print("Training...")
    data = dataCollection()
    print("Done dataCollection")
    dataClean(data)
    print("Done dataClean")
    clusters = get_all_clusters()
    print("Done clusters")
    
    data = prepare_data_for_kmeans(clusters)
    print("Done prepare_data_for_kmeans")
    labels, centroids = apply_kmeans(data, n_clusters=k)
    print("Done training, watting save data...")
    
    # #show
    # pca = PCA(n_components=2)
    # reduced_data = pca.fit_transform(data)
    # # Áp dụng PCA lên các tâm cụm (centroids)
    # reduced_centroids = pca.transform(centroids)
    # # Trực quan hóa dữ liệu và tâm cụm
    # plt.figure(figsize=(10, 8))
    # #Vẽ dữ liệu đã phân cụm
    # scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=100, label='Data Points')
    # # Vẽ tâm cụm
    # plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    # plt.colorbar(scatter, label='Cluster Labels')
    # plt.title('Hình ảnh phân cụm với PCA', fontsize=16)
    # plt.xlabel('Principal Component 1', fontsize=12)
    # plt.ylabel('Principal Component 2', fontsize=12)
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    
    #save model
    image_ids = [cluster[0] for cluster in clusters]
    save_cluster_labels(labels, image_ids)
    save_centroids(centroids)
    print("Done")
    return  True

def search(imageRp):
    img_resized = imageRp.resize((size, size), Image.Resampling.LANCZOS)
    image_removeBG = img_resized
    # image_removeBG = remove(img_resized)
    image_array = np.array(image_removeBG)
    hsv_vector = extract_hsv(image_array)
    hog_vector = extract_hog(image_array)
    cluster= np.concatenate((hsv_vector, hog_vector))
    
    image_ids = get_sorted_image_ids(cluster)
    
    return image_ids 
def apply_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    kmeans.fit(data)
    labels = kmeans.labels_
    
    evaluate_kmeans(data, labels)

    return labels, kmeans.cluster_centers_



def dataClean(data):
    images_in_db = get_array_id_image()
    print(images_in_db)
    for image in data:
        # if(check_image(image['id']) == False):
        if( image['id'] not in images_in_db):
            response = requests.get("https://pushimage-production.up.railway.app/api/auth/image/" + image['link'])
            imageRp = Image.open(BytesIO(response.content))  
            img_resized = imageRp.resize((size, size), Image.Resampling.LANCZOS)
            image_removeBG = img_resized

            # image_removeBG = remove(img_resized)
            
            # Chuyển đổi ảnh sang mảng NumPy
            image_array = np.array(image_removeBG)
            hsv_vector = extract_hsv(image_array)
            hog_vector = extract_hog(image_array)
            cluster= np.concatenate((hsv_vector, hog_vector))
            save_image(image['id'], cluster ,0)
        print("image " + str(image['id']))
    return True    
    
    
def dataCollection():
    response = requests.get(port + "/api/auth/image/get/all").json()
    return response

def convert_bgr_to_hsv(R, G, B):
    R, G, B = R / 255, G / 255, B / 255
    max_rgb = max(R, G, B)
    min_rgb = min(R, G, B)

    V = max_rgb
    S = 0
    if V != 0:
        S = (V - min_rgb) / V
    H = 0
    if R == G and G == B:
        H = 0
    elif V == R:
        H = 60 * (G - B) / (V - min_rgb)
    elif V == G:
        H = 120 + 60 * (B - R) / (V - min_rgb)
    elif V == B:
        H = 240 + 60 * (R - G) / (V - min_rgb)
    if H < 0:
        H = H + 360
    return [H, S, V]
# Trích xuất đặc trưng hsv

def extract_hsv(image):
    # Mỗi mảng lưu một giá trị thuộc hệ màu HSV
    h_vector = np.zeros(6, dtype=int) 
    s_vector = np.zeros(8, dtype=int) 
    v_vector = np.zeros(10, dtype=int) 

    # Duyệt qua từng pixel trong ảnh
    width = size
    height = size
    for i in range(width):
        for j in range(height):
            B, G, R = image[j, i, 0], image[j, i, 1], image[j, i, 2]
            H, S, V = convert_bgr_to_hsv(R, G, B)

            # Chia bin ------------------------------
            h_index = min(5, math.floor(H / 60))
            s_index = min(5, math.floor(S / 0.125))
            v_index = min(5, math.floor(V / 0.1))

            h_vector[h_index] += 1
            s_vector[s_index] += 1
            v_vector[v_index] += 1
            # ---------------------------------------
    return np.concatenate((h_vector, s_vector, v_vector))

def convert_bgr_to_gray(image):
    width = size
    height = size
    gray_image = np.zeros((height, width), dtype=np.float32)
    for i in range(width):
        for j in range(height):
            B, G, R = image[j, i, 0], image[j, i, 1], image[j, i, 2]
            gray_image[j, i] = 0.299 * R + 0.587 * G + 0.114 * B
    return gray_image


# Trích đặc trưng hog
def extract_hog(image):
    gray_image = convert_bgr_to_gray(image)
    (hog_vector, hog_image) = hog(gray_image,
                                  orientations=9,
                                  pixels_per_cell=(8, 8),
                                  transform_sqrt=True,
                                  cells_per_block=(2, 2),
                                  block_norm="L2",
                                  visualize=True)
    return hog_vector

# Hàm tính toán khoảng cách Euclidean
def calculate_distance(centroid, new_image):
    result = 0
    for i in range(len(centroid)):
        result += (centroid[i] - new_image[i]) ** 2
    return math.sqrt(result)




    
def save_image(image_id,cluster, centroid_id):
    cluster_blob = pickle.dumps(cluster)

    sql = "INSERT INTO images (image_id,value, cluster_id) VALUES (%s, %s, %s)"
    values = (image_id, cluster_blob , centroid_id)
    

    cursor.execute(sql, values)
    conn.commit()
    
def check_image(image_id):
    query = "SELECT COUNT(*) FROM images WHERE image_id = %s"
    cursor.execute(query, (image_id,))
    result = cursor.fetchone()
    return result[0] > 0
def get_array_id_image():
    query = "SELECT image_id FROM images"
    cursor.execute(query)
    result = cursor.fetchall()
    image_ids = [row[0] for row in result]
    return image_ids
def get_all_clusters():
    query = "SELECT image_id, value, cluster_id FROM images"
    cursor.execute(query)
    result = cursor.fetchall()
    
    clusters = []
    for cluster in result:
        image_id = cluster[0]
        # Deserialize the value back from BLOB to original array format
        cluster_value = pickle.loads(cluster[1])
        clusters.append((image_id, cluster_value, cluster[2]))  # cluster[2] is cluster_id

    return clusters

def prepare_data_for_kmeans(clusters):
    data = []
    for cluster in clusters:

        # cluster_str = cluster[1] 
   
        # value = np.array(list(map(float, cluster_str.split(','))))
        data.append(cluster[1])

    return np.array(data)

def save_cluster_labels(labels, image_ids):
    query = "UPDATE images SET cluster_id = %s WHERE image_id = %s"
    

        # Cập nhật nhãn cho từng image_id
    for label, image_id in zip(labels, image_ids):
          
        label = int(label)  
        image_id = int(image_id)  
        cursor.execute(query, (label, image_id))
        
       
    conn.commit()
        
def save_centroids(centroids):
    delete_old_centroids()
    for i, centroid in enumerate(centroids):
        centroid_blob = pickle.dumps(centroid)

        query = "INSERT INTO centroids (cluster, value) VALUES (%s, %s)"
        
       
        cursor.execute(query, (i, centroid_blob))
        conn.commit()
def delete_old_centroids():
    query = "DELETE FROM centroids"
    cursor.execute(query)
    conn.commit()


def get_all_centroids():
    query = "SELECT id, value FROM centroids"

    conn =  get_db_connection()      
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()

    centroids = {}
    for centroid in results:
        centroid_id = centroid[0]
        # Deserialize the value from BLOB to original array format
        value = pickle.loads(centroid[1])
        centroids[centroid_id] = value

    return centroids

def get_clusters_by_centroid(centroid_id):
    query_centroid = "SELECT cluster FROM centroids WHERE id = %s"
    conn =  get_db_connection()      
    cursor = conn.cursor()
    cursor.execute(query_centroid, (centroid_id,))
    centroid_result = cursor.fetchone()
    
    cluster_id = centroid_result[0]
    
    query = "SELECT image_id, value FROM images WHERE cluster_id = %s"
    cursor.execute(query, (cluster_id,))
    results = cursor.fetchall()

    clusters = []
    for result in results:
        image_id = result[0]
        cluster_value = pickle.loads(result[1])
        # cluster_value = np.array(list(map(float, result[1].split(','))))
        
        clusters.append((image_id, cluster_value))

    return clusters

def find_nearest_centroid(new_cluster, centroids):
   
    distances = {cid: euclidean_distances([new_cluster], [value])[0][0] for cid, value in centroids.items()}
   
    sorted_centroids = sorted(distances, key=distances.get)
    return sorted_centroids

def sort_clusters_by_distance(new_cluster, clusters):
   
    clusters.sort(key=lambda x: euclidean_distances([new_cluster], [x[1]])[0][0])
   
    return [cluster[0] for cluster in clusters]

def get_sorted_image_ids(new_cluster):
   
    centroids = get_all_centroids()
    # print('ok1')
   
    nearest_centroids = find_nearest_centroid(new_cluster, centroids)
    
    sorted_image_ids = []
    # print('ok2')
    for i, centroid_id in enumerate(nearest_centroids):

        clusters = get_clusters_by_centroid(centroid_id)
        if i == 0: 
            sorted_image_ids.extend(sort_clusters_by_distance(new_cluster, clusters))
            # print('ok3')
        else: 
            sorted_image_ids.extend([cluster[0] for cluster in clusters])
            # print('ok4')

    return sorted_image_ids



# danh gia kmeans
def evaluate_kmeans(data, labels, true_labels=None):
    # Inertia
    inertia = KMeans(n_clusters=3).fit(data).inertia_
    print(f"Inertia: {inertia}")
    
    # Silhouette Score
    silhouette_avg = silhouette_score(data, labels)
    print(f"Silhouette Score: {silhouette_avg}")
    
    # Davies-Bouldin Index
    davies_bouldin = davies_bouldin_score(data, labels)
    print(f"Davies-Bouldin Index: {davies_bouldin}")
    
    # Homogeneity, Completeness, and V-Measure (if true_labels are available)
    if true_labels is not None:
        homogeneity = homogeneity_score(true_labels, labels)
        completeness = completeness_score(true_labels, labels)
        v_measure = v_measure_score(true_labels, labels)
        ari = adjusted_rand_score(true_labels, labels)
        print(f"Homogeneity: {homogeneity}")
        print(f"Completeness: {completeness}")
        print(f"V-Measure: {v_measure}")
        print(f"Adjusted Rand Index: {ari}")
    else:
        print("True labels not provided. Skipping Homogeneity, Completeness, V-Measure, and ARI.")