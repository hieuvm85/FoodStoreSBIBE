import requests
import numpy as np
import math
from PIL import Image
from io import BytesIO
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import mysql.connector



port = "https://foodstore-production-167c.up.railway.app"

size = 128

db = mysql.connector.connect(
    host="autorack.proxy.rlwy.net",
    user="root",
    port=42829,
    password="kBTDrcjMcanPjHCiipqrjePETXYLhUYn",
    database="railway"
)
cursor = db.cursor()

def train():
    data = dataCollection()
    dataClean(data)
    clusters = get_all_clusters()
    data = prepare_data_for_kmeans(clusters)
    labels, centroids = apply_kmeans(data, n_clusters=3)
     
    image_ids = [cluster[0] for cluster in clusters]
    save_cluster_labels(labels, image_ids)
    save_centroids(centroids)
    
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

def apply_kmeans(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    kmeans.fit(data)
    labels = kmeans.labels_

    return labels, kmeans.cluster_centers_



def dataClean(data):
    for image in data:
        if(check_image(image['id']) == False):
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
    h_vector = np.zeros(6, dtype=int) # Lưu giá trị Hue
    s_vector = np.zeros(8, dtype=int) # Lưu giá trị Saturation
    v_vector = np.zeros(10, dtype=int) # Lưu giá trị Value

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
    clusterR = [str(round(num, 4)) for num in cluster]
    cluster_str = ",".join(map(str, clusterR))
    # Câu lệnh SQL để chèn dữ liệu vào bảng image_features
    sql = "INSERT INTO cluster (image_id,cluster, centroid_id) VALUES (%s, %s, %s)"
    values = (image_id, cluster_str , centroid_id)

    cursor.execute(sql, values)
    db.commit()
    
def check_image(image_id):
    query = "SELECT COUNT(*) FROM cluster WHERE image_id = %s"
    cursor.execute(query, (image_id,))
    result = cursor.fetchone()
    return result[0] > 0

def get_all_clusters():
    query = "SELECT image_id, cluster FROM cluster"
    cursor.execute(query)
    result = cursor.fetchall()
    return result

def prepare_data_for_kmeans(clusters):
    data = []
    for cluster in clusters:
        # Lấy phần tử thứ 1 của mỗi tuple (chuỗi cluster)
        cluster_str = cluster[1] 
        # Chuyển chuỗi value thành list các số (sử dụng dấu phẩy để phân tách)
        value = np.array(list(map(float, cluster_str.split(','))))
        data.append(value)
    # Chuyển danh sách thành mảng numpy
    return np.array(data)

def save_cluster_labels(labels, image_ids):
    query = "UPDATE cluster SET centroid_id = %s WHERE image_id = %s"
    
    try:
        # Cập nhật nhãn cho từng image_id
        for label, image_id in zip(labels, image_ids):
            # Kiểm tra và ép kiểu dữ liệu trước khi gửi vào cơ sở dữ liệu
            label = int(label)  # Đảm bảo label là kiểu int
            image_id = int(image_id)  # Đảm bảo image_id là kiểu int
            cursor.execute(query, (label, image_id))
        
        # Commit thay đổi vào cơ sở dữ liệu
        db.commit()
        print(f"{cursor.rowcount} rows updated successfully.")
    
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        
def save_centroids(centroids):
    for i, centroid in enumerate(centroids):
        centroidRound = [str(round(num, 4)) for num in centroid]
        centroid_string = ",".join(map(str, centroidRound))  # Chuyển centroid thành chuỗi (các giá trị phân tách bằng dấu phẩy)
        
        # Lệnh SQL để lưu centroid
        query = "INSERT INTO centroids (cluster_id, value) VALUES (%s, %s)"
        
        # Thực thi câu lệnh SQL với cluster_id và giá trị centroid
        cursor.execute(query, (i, centroid_string))
        db.commit()
def delete_old_centroids():
    query = "DELETE FROM centroids"
    cursor.execute(query)
    db.commit()


def get_all_centroids():
    query = "SELECT id, value FROM centroids"
    cursor.execute(query)
    results = cursor.fetchall()

    centroids = {}
    for centroid in results:
        centroid_id = centroid[0]
        value = np.array(list(map(float, centroid[1].split(','))))
        centroids[centroid_id] = value

    return centroids

def get_clusters_by_centroid(centroid_id):
    query_centroid = "SELECT cluster_id FROM centroids WHERE id = %s"
    cursor.execute(query_centroid, (centroid_id,))
    centroid_result = cursor.fetchone()
    
    cluster_id = centroid_result[0]
    
    query = "SELECT image_id, cluster FROM cluster WHERE centroid_id = %s"
    cursor.execute(query, (cluster_id,))
    results = cursor.fetchall()

    clusters = []
    for result in results:
        image_id = result[0]
        cluster_value = np.array(list(map(float, result[1].split(','))))
        clusters.append((image_id, cluster_value))

    return clusters

def find_nearest_centroid(new_cluster, centroids):
    # Tính khoảng cách giữa cluster mới và tất cả các centroid
    distances = {cid: euclidean_distances([new_cluster], [value])[0][0] for cid, value in centroids.items()}
    # Sắp xếp các centroid theo khoảng cách gần nhất
    sorted_centroids = sorted(distances, key=distances.get)
    return sorted_centroids

def sort_clusters_by_distance(new_cluster, clusters):
    # Sắp xếp các clusters theo khoảng cách từ cluster mới
    clusters.sort(key=lambda x: euclidean_distances([new_cluster], [x[1]])[0][0])
    # Trả về danh sách image_id sau khi sắp xếp
    return [cluster[0] for cluster in clusters]

def get_sorted_image_ids(new_cluster):
    # Lấy tất cả các centroid
    centroids = get_all_centroids()

    # Tìm thứ tự các centroid theo khoảng cách gần nhất
    nearest_centroids = find_nearest_centroid(new_cluster, centroids)

    sorted_image_ids = []

    # Xử lý centroid gần nhất trước tiên
    for i, centroid_id in enumerate(nearest_centroids):
        clusters = get_clusters_by_centroid(centroid_id)

        if i == 0:  # Centroid gần nhất, cần sắp xếp
            sorted_image_ids.extend(sort_clusters_by_distance(new_cluster, clusters))
        else:  # Các centroid còn lại, không cần sắp xếp
            sorted_image_ids.extend([cluster[0] for cluster in clusters])

    return sorted_image_ids