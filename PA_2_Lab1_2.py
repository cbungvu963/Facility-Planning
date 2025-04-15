import pandas as pd
import numpy as np
from random import choice
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Nhập file Excel
file_path = "./dataforlab1.xlsx"
orders = pd.read_excel(r'F:\0. 0.0.0. HK242\Hoạch định mặt bằng\Thực hành\Code\Excel_files\SalesOrders.xlsx', sheet_name='Sheet1')
machines = pd.read_excel(r'F:\0. 0.0.0. HK242\Hoạch định mặt bằng\Thực hành\Code\Excel_files\Machines.xlsx', sheet_name='Sheet1')

print(machines.head())
print(orders.head())

# Đọc thêm dữ liệu từ Flow_Theo_Cap_May.xlsx và Closeness_SLP.xlsx
flow_df = pd.read_excel(r'F:\0. 0.0.0. HK242\Hoạch định mặt bằng\Thực hành\Code\Excel_files\Flow_Theo_Cap_May.xlsx', sheet_name='Sheet1')
closeness_df = pd.read_excel(r'F:\0. 0.0.0. HK242\Hoạch định mặt bằng\Thực hành\Code\Excel_files\Closeness_SLP.xlsx', sheet_name='Sheet1')

# Chuyển flow và closeness thành dạng từ điển
flow_dict = {(row.iloc[0], col): flow_df.at[i, col] 
             for i, row in flow_df.iterrows() 
             for col in flow_df.columns if col != flow_df.columns[0]}
closeness_dict = {(row.iloc[0], col): closeness_df.at[i, col] 
                  for i, row in closeness_df.iterrows() 
                  for col in closeness_df.columns if col != closeness_df.columns[0]}

# Tạo danh sách cặp máy liên kết có quantity hợp lệ 
from collections import defaultdict
quantity_map = defaultdict(int)  # Lưu tổng số hàng đi qua mỗi cặp máy

for i in range(orders.shape[0]):
    # Lấy Quantity của đơn hàng
    quantity = orders.loc[i, 'Quantity']
    
    # Tạo danh sách máy trong đơn hàng (loại bỏ NaN)
    x = [str(orders.loc[i, f'Task{j}']) for j in range(1, 6)]
    y = list(filter(lambda a: a != 'nan', x))
    
    # Ghép các cặp máy liên tiếp và chuẩn hóa
    for k in range(len(y) - 1):
        pair = (min(y[k], y[k+1]), max(y[k+1], y[k]))  # Chuẩn hóa (A, B)
        quantity_map[pair] += quantity  # Cộng dồn đúng quantity của đơn hàng

# In kết quả
print("\nTổng tần suất của các cặp máy sau khi nhân với Quantity:")
for key, value in quantity_map.items():
    print(f"{key}: {value}")
    
# Danh sách máy từ file Machines
mch = list(machines['Máy'])

# Tạo ma trận tần suất với giá trị mặc định là 0
frq_matrix0 = pd.DataFrame(0, index=mch, columns=mch, dtype=int)

# Gán tần suất vào ma trận
for (m1, m2), quantity in quantity_map.items():  # Lấy từ quantity_map đã tính đúng
    if m1 in mch and m2 in mch:  # Chỉ cập nhật nếu máy có trong danh sách
        frq_matrix0.loc[m1, m2] = quantity
        frq_matrix0.loc[m2, m1] = quantity  # Đối xứng

# Xuất ma trận tần suất vào file Excel
output_file = "lab1_2.xlsx"
frq_matrix0.to_excel(output_file, index=True)
print(f"Kết quả đã được lưu vào {output_file}")

# Tạo ma trận điểm rỗng có cùng kích thước với frq_matrix0
df3 = pd.DataFrame('', index=mch, columns=mch)  # Tạo ma trận rỗng với ký tự ''

# So sánh với các mức điểm và gán giá trị tương ứng
for dong in mch:
    for cot in mch:
        # Lấy giá trị tần suất từ frq_matrix0
        freq_value = frq_matrix0.loc[dong, cot]

        # Đảm bảo rằng freq_value là số và không phải NaN
        if pd.isna(freq_value):
            freq_value = 0

        # Gán giá trị theo mức tần suất
        if freq_value > 10000:
            df3.loc[dong, cot] = 'A'
        elif freq_value >= 5000:
            df3.loc[dong, cot] = 'E'
        elif freq_value >= 3000:
            df3.loc[dong, cot] = 'I'
        elif freq_value >= 1000:
            df3.loc[dong, cot] = 'O'
        elif freq_value > 0:
            df3.loc[dong, cot] = 'U'
        else:
            df3.loc[dong, cot] = 'X'  # Nếu freq_value = 0 thì gán 'X'

# In kết quả ra màn hình
print("\nMa trận điểm:")
print(df3)

# Hàm tính số lượng máy cần thiết
def calculate_machine_counts(sales_orders_df, machines_df, machine_list):
    C, E, R = 2000, 1, 1  # Giả định theo báo cáo
    total_quantity = sales_orders_df["Quantity"].sum()  # Tổng số lượng sản phẩm (P)
    
    machine_counts = {}
    for machine in machine_list:
        try:
            T = machines_df[machines_df["Máy"] == machine]["Thời gian gia công trung bình của máy cho các loại sản phẩm (giờ)"].values[0]
            T = float(T)  # Chuyển thành số thực
            M = (total_quantity * T) / (C * E * R)
            machine_counts[machine] = min(max(1, round(M)), 1)  # Giới hạn tối đa là 1
        except KeyError:
            print(f"Không tìm thấy cột 'Thời gian gia công trung bình của máy cho các loại sản phẩm (giờ)' trong machines_df")
            break
        except ValueError:
            print(f"Không thể chuyển đổi T của máy {machine} thành số: {T}")
            break
    
    # Điều chỉnh theo yêu cầu: MAL=3, MC=2, QC=1
    machine_counts["MAL"] = 3
    machine_counts["MC"] = 2
    machine_counts["QC"] = 1
    
    return machine_counts

# Hàm kiểm tra tổng diện tích
def check_total_area(machine_counts, machine_sizes):
    total_area = 0
    for machine, count in machine_counts.items():
        wid, length = machine_sizes[machine]
        total_area += wid * length * count
    
    print(f"Tổng diện tích cần thiết: {total_area} m²")
    if total_area > 800:
        raise ValueError("Cảnh báo: Tổng diện tích vượt quá 800 m², không thể bố trí hết tất cả máy!")
    return total_area

# Hàm sắp xếp máy bằng ALDEP
def arrange_machines_with_aldep(machine_list, closeness_dict):
    arranged_machines = []
    remaining_machines = machine_list.copy()
    
    # Bắt đầu với máy QC
    current_machine = "QC"
    arranged_machines.append(current_machine)
    remaining_machines.remove(current_machine)
    
    # Xếp các máy còn lại theo ALDEP
    while remaining_machines:
        relations = {"A": [], "E": [], "I": [], "O": [], "U": []}
        for machine in remaining_machines:
            rel = closeness_dict.get((current_machine, machine), "U")
            if rel in relations:
                relations[rel].append(machine)
        
        next_machine = None
        for level in ["A", "E", "I", "O", "U"]:
            if relations[level]:
                next_machine = choice(relations[level])
                break
        
        if not next_machine:
            next_machine = choice(remaining_machines)
        
        arranged_machines.append(next_machine)
        remaining_machines.remove(next_machine)
        current_machine = next_machine
    
    return arranged_machines

# Hàm tính điểm ưu tiên cho các cặp máy
def calculate_pair_scores(closeness_dict, flow_dict, machine_counts):
    closeness_weights = {'A': 5, 'E': 4, 'I': 3, 'O': 2, 'U': 1}
    machine_pairs = []
    
    for (m1, m2), closeness in closeness_dict.items():
        if m1 == m2 or closeness == '-' or closeness == 'U':
            continue
        flow = flow_dict.get((m1, m2), 0)
        score = closeness_weights[closeness] * flow * 1000
        if m1 in machine_counts and m2 in machine_counts:
            machine_pairs.append((m1, m2, score))
    
    machine_pairs.sort(key=lambda x: x[2], reverse=True)
    return machine_pairs

# Hàm tính tổng chi phí di chuyển
def calculate_total_cost(centroid_df, flow_df, machine_list):
    """
    Tính tổng chi phí di chuyển dựa trên khoảng cách Manhattan và lưu lượng giữa các máy.
    
    Parameters:
    - centroid_df: DataFrame chứa tọa độ trọng tâm của các máy (Machine, Centroid X, Centroid Y).
    - flow_df: DataFrame chứa lưu lượng giữa các máy (đọc từ Flow_Theo_Cap_May.xlsx).
    - machine_list: Danh sách các máy (dùng để ánh xạ chỉ số trong flow_df).
    
    Returns:
    - total_cost: Tổng chi phí di chuyển.
    - distances: Từ điển chứa khoảng cách giữa các cặp máy.
    """
    # Chuyển centroid_df thành dictionary để dễ truy cập
    machine_positions = {row['Machine']: (row['Centroid X'], row['Centroid Y']) 
                         for _, row in centroid_df.iterrows()}

    # Tính khoảng cách Manhattan giữa các máy
    distances = {}
    for machine1 in machine_positions:
        for machine2 in machine_positions:
            if machine1 != machine2:
                x1, y1 = machine_positions[machine1]
                x2, y2 = machine_positions[machine2]
                distances[(machine1, machine2)] = abs(x1 - x2) + abs(y1 - y2)

    # Chuyển flow_df thành ma trận numpy (bỏ cột đầu tiên là tên máy)
    flow_matrix = flow_df.set_index(flow_df.columns[0]).to_numpy()

    # Tính tổng chi phí
    total_cost = 0
    for i, m1 in enumerate(machine_list):
        for j, m2 in enumerate(machine_list):
            if i != j:
                # Lấy lưu lượng từ flow_matrix
                flow = flow_matrix[i, j] if i < flow_matrix.shape[0] and j < flow_matrix.shape[1] else 0
                
                # Tìm tất cả các phiên bản của m1 và m2 (ví dụ: MAL_1, MAL_2)
                m1_instances = [m for m in machine_positions.keys() if m.startswith(m1 + '_') or m == m1]
                m2_instances = [m for m in machine_positions.keys() if m.startswith(m2 + '_') or m == m2]

                # Tính chi phí cho tất cả các cặp giữa m1_instances và m2_instances
                for instance1 in m1_instances:
                    for instance2 in m2_instances:
                        if instance1 != instance2:
                            dist = distances.get((instance1, instance2), 0)  # Nếu không có khoảng cách, giả định 0
                            total_cost += flow * dist  # Chi phí = flow * dist

    print(f"Khoảng cách giữa các máy: {distances}")
    print(f"Tổng chi phí di chuyển: {total_cost}")
    return total_cost, distances

# Tính số lượng máy
machine_counts = calculate_machine_counts(orders, machines, mch)

# Kiểm tra tổng diện tích
machine_sizes = dict(zip(machines["Máy"], zip(machines["Wid"], machines["Length"])))
check_total_area(machine_counts, machine_sizes)

# Ma trận quan hệ
relations = {}
for m1 in df3.index:
    for m2 in df3.columns:
        if m1 != m2 and df3.loc[m1, m2] in ['A', 'E', 'I', 'O', 'U']:
            relations[(m1, m2)] = df3.loc[m1, m2]  # Lưu mức độ quan hệ

# Sắp xếp máy bằng ALDEP
arranged_machines = arrange_machines_with_aldep(mch, closeness_dict)

# Tính điểm ưu tiên cho các cặp máy
machine_pairs = calculate_pair_scores(closeness_dict, flow_dict, machine_counts)

# Tạo một danh sách để lưu tọa độ trọng tâm của các máy
machine_centroids = []

# Khởi tạo mặt bằng
width, length = 40, 20
fig, ax = plt.subplots(figsize=(40, 20))
ax.set_xlim(0, width)
ax.set_ylim(0, length)
ax.set_title("Bố trí mặt bằng tối ưu theo SLP", fontsize=16)
ax.set_xlabel("Width (m)")
ax.set_ylabel("Length (m)")

# Duyệt qua từng máy đã được đặt
x, y = 0, length  # Bắt đầu từ góc trên trái
min_distance = 1.0
column_machines = []
is_odd_column = True

# Duyệt qua từng máy theo thứ tự ưu tiên và số lượng máy cần đặt
for machine in arranged_machines:  # Sử dụng arranged_machines từ ALDEP
    machine_width, machine_length = machine_sizes[machine]
    num_machines = machine_counts.get(machine, 1) 

    for idx in range(num_machines):  # Lặp theo số lượng máy cần đặt
        # Kiểm tra nếu không đủ chỗ theo chiều dọc
        if is_odd_column:
            if y - machine_length < 0:
                x += max(width for _, width, _ in column_machines) + min_distance
                is_odd_column = False
                y = 0
                column_machines = []
        else:
            if y + machine_length > length:
                x += max(width for _, width, _ in column_machines) + min_distance
                is_odd_column = True
                y = length
                column_machines = []

        # Nếu hết không gian, dừng lại
        if x + machine_width > width:
            print("Không đủ chỗ để sắp xếp tất cả các máy trong mặt bằng.")
            break

        # Vẽ máy
        rect = patches.Rectangle((x, y - machine_length if is_odd_column else y),
                                 machine_width, machine_length,
                                 linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.6)
        ax.add_patch(rect)

        # Gắn nhãn (thêm số thứ tự nếu có nhiều máy)
        ax.text(x + machine_width / 2, (y - machine_length / 2 if is_odd_column else y + machine_length / 2),
                f"{machine} ({idx+1}/{num_machines})", ha='center', va='center', fontsize=8)

        # Tính tọa độ trọng tâm
        x_center = x + machine_width / 2
        y_center = y - machine_length / 2 if is_odd_column else y + machine_length / 2

        # Lưu tọa độ trọng tâm vào danh sách, phân biệt các máy cùng loại
        machine_centroids.append({
            'Machine': f"{machine}_{idx+1}" if num_machines > 1 else machine,
            'Centroid X': x_center,
            'Centroid Y': y_center
        })

        # Cập nhật vị trí y
        if is_odd_column:
            y -= machine_length + min_distance
        else:
            y += machine_length + min_distance

        # Lưu vào danh sách máy đã đặt
        column_machines.append((machine, machine_width, machine_length))
        
# Hiển thị mặt bằng
plt.show()

# Chuyển danh sách tọa độ trọng tâm thành DataFrame
centroid_df = pd.DataFrame(machine_centroids)

# Xuất tọa độ trọng tâm vào file Excel
centroid_output_file = "Machine_Centroids.xlsx"
centroid_df.to_excel(centroid_output_file, index=False)
print(f"Tọa độ trọng tâm đã được lưu vào {centroid_output_file}")

# Tính tổng chi phí di chuyển
total_cost, distances = calculate_total_cost(centroid_df, flow_df, mch)



