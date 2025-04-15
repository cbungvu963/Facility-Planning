import pandas as pd
import numpy as np
from random import choice

# Đọc dữ liệu từ các file Excel
machines_df = pd.read_excel(r'F:\0. 0.0.0. HK242\Hoạch định mặt bằng\Thực hành\Code\Excel_files\Machines.xlsx', sheet_name='Sheet1')  # Dữ liệu máy: tên, Wid, Length
sales_orders_df = pd.read_excel(r'F:\0. 0.0.0. HK242\Hoạch định mặt bằng\Thực hành\Code\Excel_files\SalesOrders.xlsx', sheet_name='Sheet1')  # Dữ liệu đơn hàng
flow_df = pd.read_excel(r'F:\0. 0.0.0. HK242\Hoạch định mặt bằng\Thực hành\Code\Excel_files\Flow_Theo_Cap_May.xlsx', sheet_name='Sheet1')  # Ma trận dòng chảy
closeness_df = pd.read_excel(r'F:\0. 0.0.0. HK242\Hoạch định mặt bằng\Thực hành\Code\Excel_files\Closeness_SLP.xlsx', sheet_name='Sheet1')  # Ma trận quan hệ AEIOUX
coords_df = pd.read_excel(r'F:\0. 0.0.0. HK242\Hoạch định mặt bằng\Thực hành\Code\Excel_files\Toa_do_ban_dau.xlsx', sheet_name='Sheet1')  # Tọa độ ban đầu và kích thước

# Lấy danh sách tên máy từ Machines.xlsx
machine_list = machines_df["Máy"].tolist()

# Kiểm tra dữ liệu
#print("Danh sách máy:", machine_list)
#print("Kích thước flow_df:", flow_df.shape)
#print("Kích thước closeness_df:", closeness_df.shape)
#print(flow_df)

#--------------------------------------------------
# Công thức: M = P * T / (C * E * R)
# Giả định: C = 2000 giờ, E = 1, R = 1 (theo báo cáo)
C, E, R = 2000, 1, 1

# Tính tổng số lượng sản phẩm (P) từ SalesOrders.xlsx
total_quantity = sales_orders_df["Quantity"].sum()

# Tính số máy cần thiết cho từng loại máy
machine_counts = {}
for machine in machine_list:
    # Giả định tên cột đúng là "Thời gian gia công trung bình của máy cho các loại sản phẩm (giờ)"
    # Nếu tên cột khác, thay đổi sau khi kiểm tra kết quả từ print ở trên
    try:
        T = machines_df[machines_df["Máy"] == machine]["Thời gian gia công trung bình của máy cho các loại sản phẩm (giờ)"].values[0]
        # Chuyển T thành kiểu số
        T = float(T)  # Chuyển chuỗi thành số thực
        M = (total_quantity * T) / (C * E * R)
        machine_counts[machine] = min(max(1, round(M)), 1)  # Giới hạn tối đa là 1
    except KeyError:
        print(f"Không tìm thấy cột 'Thời gian gia công trung bình của máy cho các loại sản phẩm (giờ)' trong machines_df")
        break
    except ValueError:
        print(f"Không thể chuyển đổi T của máy {machine} thành số: {T}")
        break


# Điều chỉnh theo báo cáo: MAL=3, MC=2, QC=2
machine_counts["MAL"] = 3
machine_counts["MC"] = 2
machine_counts["QC"] = 1                                #000000000000000000000000000000 # Giới hạn tối đa là 1 máy cho các máy khác

# Kiểm tra kết quả
#print("Tổng số lượng sản phẩm (P):", total_quantity)
#print("Số lượng máy cần thiết:", machine_counts)
"""
# Kiểm tra kết quả
Tổng số lượng sản phẩm (P): 130671
Số lượng máy cần thiết: {'MAL': 3, 'GCN': 1, 'SH': 1, 'LA': 1, 'GRS': 1, 'DC': 1, 'EWC': 1, 'MI': 1, 'Đ/CHẤT': 1, 'MC': 2, 'O_LA': 1, 'O_MC': 1, 'O_MI': 1, 'To': 1, 'O-MC': 1, 'T': 1, 'LA2': 1, 'EDM': 1, 'H CR': 1, 'WELD': 1, 'O-EWC': 1, 'GRI': 1, 'GRO': 1, 'SOB': 1, 'NI HOA': 1, 'SA W': 1, 'NI': 1, 'SA B': 1, 'NGUOI': 1, 'SAW': 1, 'SEH-CR': 1, 'BCAT': 1, 'QC': 2}
"""



#--------------------------------------------------
# 3. Xây dựng ma trận quan hệ AEIOUX

# Lấy ma trận quan hệ từ Closeness_SLP.xlsx
closeness_matrix = closeness_df.set_index("Flow").to_numpy()

# Tạo từ điển quan hệ giữa các cặp máy
relationship_dict = {}
for i, machine1 in enumerate(machine_list):
    for j, machine2 in enumerate(machine_list):
        if i != j:  # Bỏ qua đường chéo
            relationship_dict[(machine1, machine2)] = closeness_matrix[i, j]

# Kiểm tra một số quan hệ
#print("Quan hệ MAL-SH:", relationship_dict[("MAL", "SH")])  # Mong đợi: A   --> OK
#print("Quan hệ QC-GRS:", relationship_dict[("QC", "GRS")])  # Mong đợi: E   --> OK
#print(closeness_matrix)
'''
[['-' 'U' 'A' ... 'U' 'U' 'U'] 
 ['U' '-' 'E' ... 'U' 'U' 'U'] 
 ['U' 'U' '-' ... 'U' 'U' 'U'] 
 ...
 ['U' 'U' 'U' ... '-' 'U' 'U'] 
 ['U' 'U' 'U' ... 'U' '-' 'U'] 
 ['U' 'U' 'U' ... 'U' 'U' '-']]
'''


#--------------------------------------------------
# 4. Thực hiện giải thuật ALDEP
# Khởi tạo danh sách đã sắp xếp và danh sách chưa sắp xếp
arranged_machines = []
remaining_machines = machine_list.copy()

# Bước 1: Chọn máy đầu tiên (QC theo báo cáo)
current_machine = "QC"
arranged_machines.append(current_machine)
remaining_machines.remove(current_machine)

# Xếp các máy còn lại theo ALDEP
while remaining_machines:
    # Tìm các máy có quan hệ với current_machine
    relations = {"A": [], "E": [], "I": [], "O": [], "U": []}
    for machine in remaining_machines:
        rel = relationship_dict.get((current_machine, machine), "U")
        if rel in relations:
            relations[rel].append(machine)
    
    # Chọn máy theo thứ tự ưu tiên: A > E > I > O > U
    next_machine = None
    for level in ["A", "E", "I", "O", "U"]:
        if relations[level]:
            next_machine = choice(relations[level])  # Chọn ngẫu nhiên nếu có nhiều máy cùng mức
            break
    
    # Nếu không có quan hệ nào, chọn ngẫu nhiên từ các máy còn lại
    if not next_machine:
        next_machine = choice(remaining_machines)
    
    arranged_machines.append(next_machine)
    remaining_machines.remove(next_machine)
    current_machine = next_machine

# Kiểm tra kết quả
#print("Thứ tự máy theo ALDEP:", arranged_machines)
'''
Thứ tự máy theo ALDEP: ['QC', 'GRS', 'MC', 'LA2', 'MAL', 'SH', 'MI', 'To', 'BCAT', 'NGUOI', 'T', 'EDM', 'GRO', 'DC', 'EWC', 'O_MC', 'SA W', 'SA B', 'O_LA', 'WELD', 'O-EWC', 'SAW', 'Đ/CHẤT', 'SOB', 'NI', 'H CR', 'NI HOA', 'O-MC', 'SEH-CR', 'GCN', 'O_MI', 'GRI', 'LA']
'''


#--------------------------------------------------
# 5. Bố trí máy trên mặt bằng (40m x 20m)
# Khởi tạo lưới 40m x 20m (chia thành ô 1m x 1m)

grid = np.full((200, 400), None, dtype=object)

# Lấy kích thước máy từ Machines.xlsx
machine_sizes = {row["Máy"]: (row["Wid"], row["Length"]) for _, row in machines_df.iterrows()}

# Kiểm tra tổng diện tích trước khi bố trí
total_area = 0
for machine in machine_counts:
    wid, length = machine_sizes[machine]
    total_area += wid * length * machine_counts[machine]

print(f"Tổng diện tích cần thiết: {total_area} m²")

if total_area > 800:
    print("Cảnh báo: Tổng diện tích vượt quá 800 m², không thể bố trí hết tất cả máy!")
    raise ValueError("Diện tích vượt quá giới hạn!")

# Chuyển closeness và flow thành dạng từ điển
closeness_dict = {(row["Flow"], col): closeness_df.at[i, col] 
                  for i, row in closeness_df.iterrows() 
                  for col in closeness_df.columns if col != "Flow"}
flow_dict = {(row["Flow"], col): flow_df.at[i, col] 
             for i, row in flow_df.iterrows() 
             for col in flow_df.columns if col != "Flow"}

# Định nghĩa trọng số cho closeness
closeness_weights = {'A': 5, 'E': 4, 'I': 3, 'O': 2, 'U': 1}

# Tính điểm ưu tiên cho từng cặp máy (dựa trên closeness và flow)
machine_pairs = []
for (m1, m2), closeness in closeness_dict.items():
    if m1 == m2 or closeness == '-':
        continue
    if closeness == 'U':
        continue
    flow = flow_dict.get((m1, m2), 0)
    score = closeness_weights[closeness] * flow * 1000  # Nhân 1000 để ưu tiên lưu lượng cao
    if m1 in machine_counts and m2 in machine_counts:
        machine_pairs.append((m1, m2, score))

# Sắp xếp các cặp máy theo điểm ưu tiên giảm dần
machine_pairs.sort(key=lambda x: x[2], reverse=True)

# Bố trí máy
machine_positions = {}
occupied_regions = []
placed_machines = set()
remaining_machines = set()

# Tạo danh sách tất cả máy
for machine in arranged_machines:
    for i in range(machine_counts[machine]):
        machine_name = f"{machine}_{i}" if i > 0 else machine
        remaining_machines.add(machine_name)

# Đặt cặp máy có điểm ưu tiên cao nhất trước (SH và MAL)
m1, m2, _ = machine_pairs[0]  # SH và MAL
for i in range(machine_counts[m1]):
    m1_name = f"{m1}_{i}" if i > 0 else m1
    wid, length = machine_sizes[m1]
    wid_cells = int(wid * 10)
    length_cells = int(length * 10)

    # Đặt ở vị trí trung tâm
    x = 20 - wid / 2
    y = 10 - length / 2
    x_cells = int(x * 10)
    y_cells = int(y * 10)

    overlap = False
    for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
        x_lower_left = x
        y_lower_left = y
        x_upper_right = x + wid
        y_upper_right = y + length
        if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
            overlap = True
            break

    if not overlap:
        region = grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells]
        if region.size == wid_cells * length_cells and np.all(region == None):
            grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells] = m1_name
            machine_positions[m1_name] = (x + wid/2, y + length/2)
            occupied_regions.append((m1_name, x, y, x + wid, y + length))
            placed_machines.add(m1_name)
            remaining_machines.remove(m1_name)

for j in range(machine_counts[m2]):
    m2_name = f"{m2}_{j}" if j > 0 else m2
    wid, length = machine_sizes[m2]
    wid_cells = int(wid * 10)
    length_cells = int(length * 10)
    best_pos = None
    best_cost = float('inf')

    for y in np.arange(0, 20 - length + 0.1, 0.1):
        for x in np.arange(0, 40 - wid + 0.1, 0.1):
            x = round(x, 1)
            y = round(y, 1)
            x_lower_left = x
            y_lower_left = y
            x_upper_right = x + wid
            y_upper_right = y + length

            overlap = False
            for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                        y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                    overlap = True
                    break

            if not overlap:
                x_cells = int(x * 10)
                y_cells = int(y * 10)
                region = grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells]
                if region.size == wid_cells * length_cells and np.all(region == None):
                    cost = 0
                    x_center = x + wid / 2
                    y_center = y + length / 2
                    for placed_machine, (px, py) in machine_positions.items():
                        base_placed = placed_machine if '_' not in placed_machine else placed_machine.split('_')[0]
                        if base_placed == m1:
                            flow = flow_dict.get((m1, m2), 0)
                            closeness = closeness_dict.get((m1, m2), 'U')
                            dist = np.sqrt((x_center - px)**2 + (y_center - py)**2)
                            cost += (flow * 1000 + closeness_weights[closeness]) * dist

                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y)

    if best_pos:
        x, y = best_pos
        x_center = x + wid / 2
        y_center = y + length / 2
        x_cells = int(x * 10)
        y_cells = int(y * 10)
        grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells] = m2_name
        machine_positions[m2_name] = (x_center, y_center)
        occupied_regions.append((m2_name, x, y, x + wid, y + length))
        placed_machines.add(m2_name)
        remaining_machines.remove(m2_name)

# Đặt các máy còn lại
for m1, m2, _ in machine_pairs[1:]:  # Bỏ qua cặp đầu tiên (SH và MAL)
    for i in range(machine_counts[m1]):
        m1_name = f"{m1}_{i}" if i > 0 else m1
        if m1_name in placed_machines:
            continue
        wid, length = machine_sizes[m1]
        wid_cells = int(wid * 10)
        length_cells = int(length * 10)
        best_pos = None
        best_cost = float('inf')

        for y in np.arange(0, 20 - length + 0.1, 0.1):
            for x in np.arange(0, 40 - wid + 0.1, 0.1):
                x = round(x, 1)
                y = round(y, 1)
                x_lower_left = x
                y_lower_left = y
                x_upper_right = x + wid
                y_upper_right = y + length

                overlap = False
                for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                    if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                            y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                        overlap = True
                        break

                if not overlap:
                    x_cells = int(x * 10)
                    y_cells = int(y * 10)
                    region = grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells]
                    if region.size == wid_cells * length_cells and np.all(region == None):
                        cost = 0
                        x_center = x + wid / 2
                        y_center = y + length / 2
                        for placed_machine, (px, py) in machine_positions.items():
                            base_placed = placed_machine if '_' not in placed_machine else placed_machine.split('_')[0]
                            if base_placed == m2:
                                flow = flow_dict.get((m1, m2), 0)
                                closeness = closeness_dict.get((m1, m2), 'U')
                                dist = np.sqrt((x_center - px)**2 + (y_center - py)**2)
                                cost += (flow * 1000 + closeness_weights[closeness]) * dist

                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y)

        if best_pos:
            x, y = best_pos
            x_center = x + wid / 2
            y_center = y + length / 2
            x_cells = int(x * 10)
            y_cells = int(y * 10)
            grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells] = m1_name
            machine_positions[m1_name] = (x_center, y_center)
            occupied_regions.append((m1_name, x, y, x + wid, y + length))
            placed_machines.add(m1_name)
            remaining_machines.remove(m1_name)

    for j in range(machine_counts[m2]):
        m2_name = f"{m2}_{j}" if j > 0 else m2
        if m2_name in placed_machines:
            continue
        wid, length = machine_sizes[m2]
        wid_cells = int(wid * 10)
        length_cells = int(length * 10)
        best_pos = None
        best_cost = float('inf')

        for y in np.arange(0, 20 - length + 0.1, 0.1):
            for x in np.arange(0, 40 - wid + 0.1, 0.1):
                x = round(x, 1)
                y = round(y, 1)
                x_lower_left = x
                y_lower_left = y
                x_upper_right = x + wid
                y_upper_right = y + length

                overlap = False
                for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                    if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                            y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                        overlap = True
                        break

                if not overlap:
                    x_cells = int(x * 10)
                    y_cells = int(y * 10)
                    region = grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells]
                    if region.size == wid_cells * length_cells and np.all(region == None):
                        cost = 0
                        x_center = x + wid / 2
                        y_center = y + length / 2
                        for placed_machine, (px, py) in machine_positions.items():
                            base_placed = placed_machine if '_' not in placed_machine else placed_machine.split('_')[0]
                            if base_placed == m1:
                                flow = flow_dict.get((m1, m2), 0)
                                closeness = closeness_dict.get((m1, m2), 'U')
                                dist = np.sqrt((x_center - px)**2 + (y_center - py)**2)
                                cost += (flow * 1000 + closeness_weights[closeness]) * dist

                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y)

        if best_pos:
            x, y = best_pos
            x_center = x + wid / 2
            y_center = y + length / 2
            x_cells = int(x * 10)
            y_cells = int(y * 10)
            grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells] = m2_name
            machine_positions[m2_name] = (x_center, y_center)
            occupied_regions.append((m2_name, x, y, x + wid, y + length))
            placed_machines.add(m2_name)
            remaining_machines.remove(m2_name)

# Đặt các máy còn lại
for machine_name in remaining_machines.copy():
    base_machine = machine_name
    if machine_name in machine_sizes:
        wid, length = machine_sizes[machine_name]
    else:
        base_machine = machine_name.split('_')[0] if '_' in machine_name and machine_name.split('_')[1].isdigit() else machine_name
        if base_machine not in machine_sizes:
            print(f"Cảnh báo: Không tìm thấy kích thước cho máy '{base_machine}' (máy gốc: '{machine_name}')")
            continue
        wid, length = machine_sizes[base_machine]
    
    wid_cells = int(wid * 10)
    length_cells = int(length * 10)
    placed = False

    for y in np.arange(0, 20 - length + 0.1, 0.1):
        for x in np.arange(0, 40 - wid + 0.1, 0.1):
            x = round(x, 1)
            y = round(y, 1)
            x_lower_left = x
            y_lower_left = y
            x_upper_right = x + wid
            y_upper_right = y + length

            overlap = False
            for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                        y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                    overlap = True
                    break

            if not overlap:
                x_cells = int(x * 10)
                y_cells = int(y * 10)
                region = grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells]
                if region.size == wid_cells * length_cells and np.all(region == None):
                    grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells] = machine_name
                    machine_positions[machine_name] = (x + wid/2, y + length/2)
                    occupied_regions.append((machine_name, x_lower_left, y_lower_left, x_upper_right, y_upper_right))
                    placed_machines.add(machine_name)
                    remaining_machines.remove(machine_name)
                    placed = True
                    break
        if placed:
            break

# Kiểm tra kết quả
print("Tọa độ máy:", machine_positions)

"""
Tổng diện tích cần thiết: 292 m²
Dừng bố trí tại máy SEH-CR: Không đủ diện tích hoặc vượt quá số bước tối đa (1000)
Tọa độ máy: {'QC': (15.0, 6.0), 'QC_1': (25.0, 6.0), 'GRS': (20.0, 0.5), 'MC': (21.5, 0.5), 'MC_1': (22.5, 0.5), 'LA2': (22.5, 1.0), 'SA W': (23.5, 1.0), 'SA B': (24.0, 0.5), 'MI': (24.0, 0.5), 'To': (24.5, 1.0), 'GCN': (25.5, 0.5), 'SH': (26.5, 0.5), 'SOB': (27.5, 2.0), 'GRO': (28.5, 0.5), 'NGUOI': (29.5, 0.5), 'O_LA': (30.0, 0.5), 'EDM': (30.5, 1.5), 'O-MC': (31.5, 0.5), 'SAW': (32.5, 0.5), 'NI': (33.5, 1.0), 'O-EWC': (34.0, 0.5), 'T': (34.5, 0.5), 'H CR': (35.5, 1.0), 'BCAT': (36.5, 0.5), 'DC': (37.5, 0.5), 'NI HOA': (38.5, 1.0), 'Đ/CHẤT': (39.5, 2.0)}
"""


#--------------------------------------------------
# 6. Tính khoảng cách và chi phí tổng thể
# Tính khoảng cách Rectilinear
distances = {}
for m1 in machine_positions:
    for m2 in machine_positions:
        if m1 != m2:
            x1, y1 = machine_positions[m1]
            x2, y2 = machine_positions[m2]
            distances[(m1, m2)] = abs(x1 - x2) + abs(y1 - y2)

# Tính chi phí tổng thể
flow_matrix = flow_df.set_index("Flow").to_numpy()
total_cost = 0
for i, m1 in enumerate(machine_list):
    for j, m2 in enumerate(machine_list):
        if i != j:
            flow = flow_matrix[i, j]
            dist = distances.get((m1, m2), 0)  # Nếu không có khoảng cách, giả định 0
            total_cost += flow * dist  # Giả định chi phí đơn vị = 1

# Kiểm tra kết quả
#print("Khoảng cách giữa các máy:", distances)
print("Tổng chi phí:", total_cost)

"""
Khoảng cách giữa các máy: {('QC', 'QC_1'): 10.0, ('QC', 'T'): 11.0, ('QC', 'EDM'): 11.0, ('QC', 'GRO'): 13.0, ('QC', 'O_MI'): 14.0, ('QC', 'LA'): 14.0, ('QC', 'MC'): 17.0, ('QC', 'MC_1'): 18.0, ('QC', 'LA2'): 17.5, ('QC', 'GRI'): 19.0, ('QC', 'EWC'): 19.0, ('QC', 'O_LA'): 20.5, ('QC', 'O_MC'): 21.0, ('QC', 'O-MC'): 22.0, ('QC', 'MI'): 22.5, ('QC', 'To'): 22.5, ('QC', 'GRS'): 23.5, ('QC', 'NGUOI'): 24.0, ('QC', 'MAL'): 26.5, ('QC', 'MAL_1'): 27.5, ('QC', 'MAL_2'): 28.5, ('QC', 'SH'): 28.0, ('QC', 'Đ/CHẤT'): 27.5, ('QC', 'SA W'): 29.5, ('QC_1', 'QC'): 10.0, ('QC_1', 'T'): 10.0, ('QC_1', 'EDM'): 8.0, ('QC_1', 'GRO'): 8.0, ('QC_1', 'O_MI'): 7.0, ('QC_1', 'LA'): 5.0, ('QC_1', 'MC'): 7.0, ('QC_1', 'MC_1'): 8.0, ('QC_1', 'LA2'): 7.5, ('QC_1', 'GRI'): 9.0, ('QC_1', 'EWC'): 9.0, ('QC_1', 'O_LA'): 10.5, ('QC_1', 'O_MC'): 11.0, ('QC_1', 'O-MC'): 12.0, ('QC_1', 'MI'): 12.5, ('QC_1', 'To'): 12.5, ('QC_1', 'GRS'): 13.5, ('QC_1', 'NGUOI'): 14.0, ('QC_1', 'MAL'): 16.5, ('QC_1', 'MAL_1'): 17.5, ('QC_1', 'MAL_2'): 18.5, ('QC_1', 'SH'): 18.0, ('QC_1', 'Đ/CHẤT'): 17.5, ('QC_1', 'SA W'): 19.5, ('T', 'QC'): 11.0, ('T', 'QC_1'): 10.0, ('T', 'EDM'): 2.0, ('T', 'GRO'): 2.0, ('T', 'O_MI'): 3.0, ('T', 'LA'): 5.0, ('T', 'MC'): 6.0, ('T', 'MC_1'): 7.0, ('T', 'LA2'): 7.5, ('T', 'GRI'): 8.0, ('T', 'EWC'): 
10.0, ('T', 'O_LA'): 9.5, ('T', 'O_MC'): 10.0, ('T', 'O-MC'): 11.0, ('T', 'MI'): 11.5, ('T', 'To'): 12.5, ('T', 'GRS'): 12.5, ('T', 'NGUOI'): 13.0, ('T', 'MAL'): 16.5, ('T', 'MAL_1'): 17.5, ('T', 'MAL_2'): 18.5, ('T', 'SH'): 17.0, ('T', 'Đ/CHẤT'): 19.5, ('T', 'SA W'): 19.5, ('EDM', 'QC'): 11.0, ('EDM', 'QC_1'): 8.0, ('EDM', 'T'): 2.0, ('EDM', 'GRO'): 2.0, ('EDM', 'O_MI'): 3.0, ('EDM', 'LA'): 3.0, ('EDM', 'MC'): 6.0, ('EDM', 'MC_1'): 7.0, ('EDM', 'LA2'): 6.5, ('EDM', 'GRI'): 8.0, ('EDM', 'EWC'): 8.0, ('EDM', 'O_LA'): 9.5, ('EDM', 'O_MC'): 10.0, ('EDM', 'O-MC'): 11.0, ('EDM', 'MI'): 11.5, ('EDM', 'To'): 11.5, ('EDM', 'GRS'): 12.5, ('EDM', 'NGUOI'): 13.0, ('EDM', 'MAL'): 15.5, ('EDM', 'MAL_1'): 16.5, ('EDM', 'MAL_2'): 17.5, ('EDM', 'SH'): 17.0, ('EDM', 'Đ/CHẤT'): 17.5, ('EDM', 'SA W'): 18.5, ('GRO', 'QC'): 13.0, ('GRO', 'QC_1'): 8.0, ('GRO', 'T'): 2.0, ('GRO', 'EDM'): 2.0, ('GRO', 'O_MI'): 1.0, ('GRO', 'LA'): 3.0, ('GRO', 'MC'): 4.0, ('GRO', 'MC_1'): 5.0, ('GRO', 'LA2'): 5.5, ('GRO', 'GRI'): 6.0, ('GRO', 'EWC'): 8.0, ('GRO', 'O_LA'): 7.5, ('GRO', 'O_MC'): 8.0, ('GRO', 'O-MC'): 9.0, ('GRO', 'MI'): 9.5, ('GRO', 'To'): 10.5, ('GRO', 'GRS'): 10.5, ('GRO', 'NGUOI'): 11.0, ('GRO', 'MAL'): 14.5, ('GRO', 'MAL_1'): 15.5, ('GRO', 'MAL_2'): 16.5, ('GRO', 'SH'): 15.0, ('GRO', 'Đ/CHẤT'): 17.5, ('GRO', 'SA W'): 17.5, ('O_MI', 'QC'): 14.0, ('O_MI', 'QC_1'): 7.0, ('O_MI', 'T'): 3.0, ('O_MI', 'EDM'): 3.0, ('O_MI', 'GRO'): 1.0, ('O_MI', 'LA'): 2.0, ('O_MI', 'MC'): 3.0, ('O_MI', 'MC_1'): 4.0, ('O_MI', 'LA2'): 4.5, ('O_MI', 'GRI'): 5.0, ('O_MI', 'EWC'): 7.0, ('O_MI', 'O_LA'): 6.5, ('O_MI', 'O_MC'): 7.0, ('O_MI', 'O-MC'): 8.0, ('O_MI', 'MI'): 8.5, ('O_MI', 'To'): 9.5, ('O_MI', 'GRS'): 9.5, ('O_MI', 'NGUOI'): 10.0, ('O_MI', 'MAL'): 13.5, ('O_MI', 'MAL_1'): 14.5, ('O_MI', 'MAL_2'): 15.5, ('O_MI', 'SH'): 14.0, ('O_MI', 'Đ/CHẤT'): 16.5, ('O_MI', 'SA W'): 16.5, ('LA', 'QC'): 14.0, ('LA', 'QC_1'): 5.0, ('LA', 'T'): 5.0, ('LA', 'EDM'): 3.0, 
('LA', 'GRO'): 3.0, ('LA', 'O_MI'): 2.0, ('LA', 'MC'): 3.0, ('LA', 'MC_1'): 4.0, ('LA', 'LA2'): 3.5, ('LA', 'GRI'): 5.0, ('LA', 'EWC'): 5.0, ('LA', 'O_LA'): 6.5, ('LA', 'O_MC'): 7.0, ('LA', 'O-MC'): 8.0, ('LA', 'MI'): 8.5, ('LA', 'To'): 8.5, ('LA', 'GRS'): 9.5, ('LA', 'NGUOI'): 10.0, ('LA', 'MAL'): 12.5, ('LA', 'MAL_1'): 13.5, ('LA', 'MAL_2'): 14.5, ('LA', 'SH'): 14.0, ('LA', 'Đ/CHẤT'): 14.5, ('LA', 'SA W'): 15.5, ('MC', 'QC'): 
17.0, ('MC', 'QC_1'): 7.0, ('MC', 'T'): 6.0, ('MC', 'EDM'): 6.0, ('MC', 'GRO'): 4.0, ('MC', 'O_MI'): 3.0, ('MC', 'LA'): 3.0, ('MC', 'MC_1'): 1.0, ('MC', 'LA2'): 1.5, ('MC', 'GRI'): 2.0, ('MC', 'EWC'): 4.0, ('MC', 'O_LA'): 3.5, ('MC', 'O_MC'): 4.0, ('MC', 'O-MC'): 5.0, ('MC', 'MI'): 5.5, ('MC', 'To'): 6.5, ('MC', 'GRS'): 6.5, ('MC', 'NGUOI'): 7.0, ('MC', 'MAL'): 10.5, ('MC', 'MAL_1'): 11.5, ('MC', 'MAL_2'): 12.5, ('MC', 'SH'): 11.0, ('MC', 'Đ/CHẤT'): 13.5, ('MC', 'SA W'): 13.5, ('MC_1', 'QC'): 18.0, ('MC_1', 'QC_1'): 8.0, ('MC_1', 'T'): 7.0, ('MC_1', 'EDM'): 7.0, ('MC_1', 'GRO'): 5.0, ('MC_1', 'O_MI'): 4.0, ('MC_1', 'LA'): 4.0, ('MC_1', 'MC'): 1.0, ('MC_1', 'LA2'): 0.5, ('MC_1', 'GRI'): 1.0, ('MC_1', 'EWC'): 3.0, ('MC_1', 'O_LA'): 2.5, ('MC_1', 'O_MC'): 3.0, ('MC_1', 'O-MC'): 4.0, ('MC_1', 'MI'): 4.5, ('MC_1', 'To'): 5.5, ('MC_1', 'GRS'): 5.5, ('MC_1', 
'NGUOI'): 6.0, ('MC_1', 'MAL'): 9.5, ('MC_1', 'MAL_1'): 10.5, ('MC_1', 'MAL_2'): 11.5, ('MC_1', 'SH'): 10.0, ('MC_1', 'Đ/CHẤT'): 12.5, ('MC_1', 'SA W'): 12.5, ('LA2', 'QC'): 17.5, ('LA2', 'QC_1'): 7.5, ('LA2', 'T'): 
7.5, ('LA2', 'EDM'): 6.5, ('LA2', 'GRO'): 5.5, ('LA2', 'O_MI'): 4.5, ('LA2', 'LA'): 3.5, ('LA2', 'MC'): 1.5, ('LA2', 'MC_1'): 0.5, ('LA2', 'GRI'): 1.5, ('LA2', 'EWC'): 2.5, ('LA2', 'O_LA'): 3.0, ('LA2', 'O_MC'): 3.5, ('LA2', 'O-MC'): 4.5, ('LA2', 'MI'): 5.0, ('LA2', 'To'): 5.0, ('LA2', 'GRS'): 6.0, ('LA2', 'NGUOI'): 6.5, ('LA2', 'MAL'): 9.0, ('LA2', 'MAL_1'): 10.0, ('LA2', 'MAL_2'): 11.0, ('LA2', 'SH'): 10.5, ('LA2', 'Đ/CHẤT'): 
12.0, ('LA2', 'SA W'): 12.0, ('GRI', 'QC'): 19.0, ('GRI', 'QC_1'): 9.0, ('GRI', 'T'): 8.0, ('GRI', 'EDM'): 8.0, ('GRI', 'GRO'): 6.0, ('GRI', 'O_MI'): 5.0, ('GRI', 'LA'): 5.0, ('GRI', 'MC'): 2.0, ('GRI', 'MC_1'): 1.0, ('GRI', 'LA2'): 1.5, ('GRI', 'EWC'): 2.0, ('GRI', 'O_LA'): 1.5, ('GRI', 'O_MC'): 2.0, ('GRI', 'O-MC'): 3.0, ('GRI', 'MI'): 3.5, ('GRI', 'To'): 4.5, ('GRI', 'GRS'): 4.5, ('GRI', 'NGUOI'): 5.0, ('GRI', 'MAL'): 8.5, ('GRI', 'MAL_1'): 9.5, ('GRI', 'MAL_2'): 10.5, ('GRI', 'SH'): 9.0, ('GRI', 'Đ/CHẤT'): 11.5, ('GRI', 'SA W'): 11.5, ('EWC', 'QC'): 19.0, ('EWC', 'QC_1'): 9.0, ('EWC', 'T'): 10.0, ('EWC', 'EDM'): 8.0, ('EWC', 'GRO'): 8.0, ('EWC', 'O_MI'): 7.0, ('EWC', 'LA'): 5.0, ('EWC', 'MC'): 4.0, ('EWC', 'MC_1'): 3.0, ('EWC', 'LA2'): 2.5, ('EWC', 'GRI'): 2.0, ('EWC', 'O_LA'): 1.5, ('EWC', 'O_MC'): 2.0, ('EWC', 'O-MC'): 3.0, ('EWC', 'MI'): 3.5, ('EWC', 'To'): 3.5, ('EWC', 'GRS'): 4.5, ('EWC', 'NGUOI'): 5.0, ('EWC', 'MAL'): 7.5, ('EWC', 'MAL_1'): 8.5, ('EWC', 'MAL_2'): 9.5, ('EWC', 'SH'): 9.0, ('EWC', 'Đ/CHẤT'): 9.5, ('EWC', 'SA W'): 10.5, ('O_LA', 'QC'): 20.5, ('O_LA', 'QC_1'): 10.5, ('O_LA', 'T'): 9.5, ('O_LA', 'EDM'): 9.5, ('O_LA', 'GRO'): 7.5, ('O_LA', 'O_MI'): 6.5, ('O_LA', 'LA'): 6.5, ('O_LA', 'MC'): 3.5, ('O_LA', 'MC_1'): 2.5, ('O_LA', 'LA2'): 3.0, ('O_LA', 'GRI'): 1.5, ('O_LA', 'EWC'): 1.5, ('O_LA', 'O_MC'): 0.5, ('O_LA', 'O-MC'): 1.5, ('O_LA', 'MI'): 2.0, ('O_LA', 'To'): 3.0, ('O_LA', 'GRS'): 3.0, ('O_LA', 'NGUOI'): 3.5, ('O_LA', 'MAL'): 7.0, ('O_LA', 'MAL_1'): 8.0, ('O_LA', 'MAL_2'): 9.0, ('O_LA', 'SH'): 7.5, ('O_LA', 'Đ/CHẤT'): 10.0, ('O_LA', 'SA W'): 10.0, ('O_MC', 'QC'): 21.0, ('O_MC', 'QC_1'): 11.0, ('O_MC', 'T'): 10.0, ('O_MC', 'EDM'): 10.0, ('O_MC', 'GRO'): 8.0, ('O_MC', 'O_MI'): 7.0, ('O_MC', 'LA'): 7.0, ('O_MC', 'MC'): 4.0, ('O_MC', 'MC_1'): 3.0, ('O_MC', 'LA2'): 3.5, ('O_MC', 'GRI'): 2.0, ('O_MC', 'EWC'): 2.0, ('O_MC', 'O_LA'): 0.5, ('O_MC', 'O-MC'): 1.0, ('O_MC', 'MI'): 1.5, ('O_MC', 'To'): 2.5, ('O_MC', 'GRS'): 2.5, ('O_MC', 'NGUOI'): 3.0, ('O_MC', 'MAL'): 6.5, ('O_MC', 'MAL_1'): 7.5, ('O_MC', 'MAL_2'): 8.5, ('O_MC', 'SH'): 7.0, ('O_MC', 'Đ/CHẤT'): 9.5, ('O_MC', 'SA W'): 9.5, ('O-MC', 'QC'): 22.0, 
('O-MC', 'QC_1'): 12.0, ('O-MC', 'T'): 11.0, ('O-MC', 'EDM'): 11.0, ('O-MC', 'GRO'): 9.0, ('O-MC', 'O_MI'): 8.0, ('O-MC', 'LA'): 8.0, ('O-MC', 'MC'): 5.0, ('O-MC', 'MC_1'): 4.0, ('O-MC', 'LA2'): 4.5, ('O-MC', 'GRI'): 3.0, ('O-MC', 'EWC'): 3.0, ('O-MC', 'O_LA'): 1.5, ('O-MC', 'O_MC'): 1.0, ('O-MC', 'MI'): 0.5, ('O-MC', 'To'): 1.5, ('O-MC', 'GRS'): 1.5, ('O-MC', 'NGUOI'): 2.0, ('O-MC', 'MAL'): 5.5, ('O-MC', 'MAL_1'): 6.5, ('O-MC', 'MAL_2'): 7.5, ('O-MC', 'SH'): 6.0, ('O-MC', 'Đ/CHẤT'): 8.5, ('O-MC', 'SA W'): 8.5, ('MI', 'QC'): 22.5, ('MI', 'QC_1'): 12.5, ('MI', 'T'): 11.5, ('MI', 'EDM'): 11.5, ('MI', 'GRO'): 9.5, ('MI', 'O_MI'): 8.5, ('MI', 'LA'): 8.5, ('MI', 'MC'): 5.5, ('MI', 'MC_1'): 4.5, ('MI', 'LA2'): 5.0, ('MI', 'GRI'): 3.5, ('MI', 'EWC'): 3.5, ('MI', 'O_LA'): 2.0, ('MI', 'O_MC'): 1.5, ('MI', 'O-MC'): 0.5, ('MI', 'To'): 1.0, ('MI', 'GRS'): 1.0, ('MI', 'NGUOI'): 1.5, ('MI', 'MAL'): 5.0, ('MI', 'MAL_1'): 6.0, ('MI', 'MAL_2'): 7.0, ('MI', 'SH'): 5.5, ('MI', 'Đ/CHẤT'): 8.0, ('MI', 'SA W'): 8.0, ('To', 'QC'): 22.5, ('To', 'QC_1'): 12.5, ('To', 'T'): 12.5, ('To', 'EDM'): 11.5, ('To', 'GRO'): 10.5, ('To', 'O_MI'): 9.5, ('To', 'LA'): 8.5, ('To', 'MC'): 6.5, ('To', 'MC_1'): 5.5, ('To', 'LA2'): 5.0, ('To', 'GRI'): 4.5, ('To', 'EWC'): 3.5, ('To', 'O_LA'): 3.0, ('To', 'O_MC'): 2.5, ('To', 'O-MC'): 1.5, ('To', 'MI'): 1.0, ('To', 'GRS'): 1.0, ('To', 'NGUOI'): 1.5, ('To', 'MAL'): 4.0, ('To', 'MAL_1'): 5.0, ('To', 'MAL_2'): 6.0, ('To', 'SH'): 5.5, ('To', 'Đ/CHẤT'): 7.0, ('To', 'SA W'): 7.0, ('GRS', 
'QC'): 23.5, ('GRS', 'QC_1'): 13.5, ('GRS', 'T'): 12.5, ('GRS', 'EDM'): 12.5, ('GRS', 'GRO'): 10.5, ('GRS', 'O_MI'): 9.5, ('GRS', 'LA'): 9.5, ('GRS', 'MC'): 6.5, ('GRS', 'MC_1'): 5.5, ('GRS', 'LA2'): 6.0, ('GRS', 'GRI'): 4.5, ('GRS', 'EWC'): 4.5, ('GRS', 'O_LA'): 3.0, ('GRS', 'O_MC'): 2.5, ('GRS', 'O-MC'): 1.5, ('GRS', 'MI'): 1.0, ('GRS', 'To'): 1.0, ('GRS', 'NGUOI'): 0.5, ('GRS', 'MAL'): 4.0, ('GRS', 'MAL_1'): 5.0, ('GRS', 'MAL_2'): 6.0, ('GRS', 'SH'): 4.5, ('GRS', 'Đ/CHẤT'): 7.0, ('GRS', 'SA W'): 7.0, ('NGUOI', 'QC'): 24.0, ('NGUOI', 'QC_1'): 14.0, ('NGUOI', 'T'): 13.0, ('NGUOI', 'EDM'): 13.0, ('NGUOI', 'GRO'): 11.0, ('NGUOI', 'O_MI'): 10.0, ('NGUOI', 'LA'): 10.0, ('NGUOI', 'MC'): 7.0, ('NGUOI', 'MC_1'): 6.0, ('NGUOI', 'LA2'): 6.5, ('NGUOI', 'GRI'): 5.0, ('NGUOI', 'EWC'): 5.0, ('NGUOI', 'O_LA'): 3.5, ('NGUOI', 'O_MC'): 3.0, ('NGUOI', 'O-MC'): 2.0, ('NGUOI', 'MI'): 1.5, ('NGUOI', 'To'): 1.5, ('NGUOI', 'GRS'): 0.5, ('NGUOI', 'MAL'): 3.5, ('NGUOI', 'MAL_1'): 4.5, ('NGUOI', 'MAL_2'): 5.5, ('NGUOI', 'SH'): 4.0, ('NGUOI', 'Đ/CHẤT'): 6.5, ('NGUOI', 'SA W'): 6.5, ('MAL', 'QC'): 26.5, ('MAL', 'QC_1'): 16.5, ('MAL', 'T'): 16.5, ('MAL', 'EDM'): 15.5, ('MAL', 'GRO'): 14.5, ('MAL', 'O_MI'): 13.5, ('MAL', 'LA'): 12.5, ('MAL', 'MC'): 10.5, ('MAL', 'MC_1'): 9.5, ('MAL', 'LA2'): 9.0, ('MAL', 'GRI'): 8.5, ('MAL', 'EWC'): 7.5, ('MAL', 'O_LA'): 7.0, ('MAL', 'O_MC'): 6.5, ('MAL', 'O-MC'): 5.5, ('MAL', 'MI'): 5.0, ('MAL', 'To'): 4.0, ('MAL', 'GRS'): 4.0, ('MAL', 'NGUOI'): 3.5, ('MAL', 'MAL_1'): 1.0, ('MAL', 'MAL_2'): 2.0, ('MAL', 'SH'): 1.5, ('MAL', 'Đ/CHẤT'): 3.0, ('MAL', 'SA W'): 3.0, ('MAL_1', 'QC'): 27.5, ('MAL_1', 'QC_1'): 17.5, ('MAL_1', 'T'): 17.5, ('MAL_1', 'EDM'): 16.5, ('MAL_1', 'GRO'): 15.5, ('MAL_1', 'O_MI'): 14.5, ('MAL_1', 'LA'): 13.5, ('MAL_1', 'MC'): 11.5, ('MAL_1', 'MC_1'): 10.5, ('MAL_1', 'LA2'): 10.0, ('MAL_1', 'GRI'): 9.5, ('MAL_1', 'EWC'): 8.5, ('MAL_1', 'O_LA'): 8.0, ('MAL_1', 'O_MC'): 7.5, ('MAL_1', 'O-MC'): 6.5, ('MAL_1', 'MI'): 6.0, ('MAL_1', 'To'): 5.0, ('MAL_1', 'GRS'): 5.0, ('MAL_1', 'NGUOI'): 4.5, ('MAL_1', 'MAL'): 1.0, ('MAL_1', 'MAL_2'): 1.0, ('MAL_1', 'SH'): 0.5, ('MAL_1', 'Đ/CHẤT'): 2.0, ('MAL_1', 'SA W'): 2.0, ('MAL_2', 'QC'): 28.5, ('MAL_2', 'QC_1'): 18.5, ('MAL_2', 'T'): 18.5, ('MAL_2', 'EDM'): 17.5, ('MAL_2', 'GRO'): 16.5, ('MAL_2', 'O_MI'): 15.5, ('MAL_2', 'LA'): 14.5, ('MAL_2', 'MC'): 12.5, ('MAL_2', 'MC_1'): 11.5, 
('MAL_2', 'LA2'): 11.0, ('MAL_2', 'GRI'): 10.5, ('MAL_2', 'EWC'): 9.5, ('MAL_2', 'O_LA'): 9.0, ('MAL_2', 'O_MC'): 8.5, ('MAL_2', 'O-MC'): 7.5, ('MAL_2', 'MI'): 7.0, ('MAL_2', 'To'): 6.0, ('MAL_2', 'GRS'): 6.0, ('MAL_2', 'NGUOI'): 5.5, ('MAL_2', 'MAL'): 2.0, ('MAL_2', 'MAL_1'): 1.0, ('MAL_2', 'SH'): 1.5, ('MAL_2', 'Đ/CHẤT'): 1.0, ('MAL_2', 'SA W'): 1.0, ('SH', 'QC'): 28.0, ('SH', 'QC_1'): 18.0, ('SH', 'T'): 17.0, ('SH', 'EDM'): 17.0, ('SH', 'GRO'): 15.0, ('SH', 'O_MI'): 14.0, ('SH', 'LA'): 14.0, ('SH', 'MC'): 11.0, ('SH', 'MC_1'): 10.0, ('SH', 'LA2'): 10.5, ('SH', 'GRI'): 9.0, ('SH', 'EWC'): 9.0, ('SH', 'O_LA'): 7.5, ('SH', 'O_MC'): 7.0, ('SH', 'O-MC'): 6.0, ('SH', 'MI'): 5.5, ('SH', 'To'): 5.5, ('SH', 'GRS'): 4.5, ('SH', 'NGUOI'): 4.0, ('SH', 'MAL'): 1.5, ('SH', 'MAL_1'): 0.5, ('SH', 'MAL_2'): 1.5, ('SH', 'Đ/CHẤT'): 2.5, ('SH', 'SA W'): 2.5, ('Đ/CHẤT', 'QC'): 27.5, ('Đ/CHẤT', 'QC_1'): 17.5, ('Đ/CHẤT', 'T'): 19.5, ('Đ/CHẤT', 'EDM'): 17.5, ('Đ/CHẤT', 'GRO'): 17.5, ('Đ/CHẤT', 'O_MI'): 16.5, ('Đ/CHẤT', 'LA'): 14.5, ('Đ/CHẤT', 'MC'): 13.5, ('Đ/CHẤT', 'MC_1'): 12.5, ('Đ/CHẤT', 'LA2'): 12.0, ('Đ/CHẤT', 'GRI'): 11.5, ('Đ/CHẤT', 'EWC'): 9.5, ('Đ/CHẤT', 'O_LA'): 10.0, ('Đ/CHẤT', 'O_MC'): 9.5, ('Đ/CHẤT', 'O-MC'): 8.5, ('Đ/CHẤT', 'MI'): 8.0, ('Đ/CHẤT', 'To'): 7.0, ('Đ/CHẤT', 'GRS'): 7.0, ('Đ/CHẤT', 'NGUOI'): 6.5, ('Đ/CHẤT', 'MAL'): 3.0, ('Đ/CHẤT', 'MAL_1'): 2.0, ('Đ/CHẤT', 'MAL_2'): 1.0, ('Đ/CHẤT', 'SH'): 2.5, ('Đ/CHẤT', 'SA W'): 2.0, ('SA W', 'QC'): 29.5, ('SA W', 'QC_1'): 19.5, ('SA W', 'T'): 19.5, ('SA W', 'EDM'): 18.5, ('SA W', 'GRO'): 17.5, ('SA W', 'O_MI'): 16.5, ('SA W', 'LA'): 15.5, ('SA W', 'MC'): 13.5, ('SA W', 'MC_1'): 12.5, ('SA W', 'LA2'): 12.0, ('SA W', 'GRI'): 11.5, ('SA W', 'EWC'): 10.5, ('SA W', 'O_LA'): 10.0, ('SA W', 'O_MC'): 9.5, ('SA W', 'O-MC'): 8.5, ('SA W', 'MI'): 8.0, ('SA W', 'To'): 7.0, ('SA W', 'GRS'): 7.0, ('SA W', 'NGUOI'): 6.5, ('SA W', 'MAL'): 3.0, ('SA W', 'MAL_1'): 2.0, ('SA W', 'MAL_2'): 
1.0, ('SA W', 'SH'): 2.5, ('SA W', 'Đ/CHẤT'): 2.0}
Tổng chi phí: 4093913.5
"""

'''
Khoảng cách giữa các máy: {('QC', 'QC_1'): 10.0, ('QC', 'GRS'): 10.5, ('QC', 'MC'): 12.0, ('QC', 'MC_1'): 13.0, ('QC', 'EDM'): 12.0, ('QC', 'GRO'): 14.0, ('QC', 'NI'): 14.5, ('QC', 'NI HOA'): 15.5, ('QC', 'GRI'): 17.0, ('QC', 'MI'): 17.5, ('QC', 'To'): 17.5, ('QC', 'SA B'): 18.5, ('QC', 'O-EWC'): 18.5, ('QC', 'SOB'): 17.5, ('QC', 'LA2'): 19.5, ('QC', 'EWC'): 20.0, ('QC', 'O_MI'): 22.0, ('QC', 'LA'): 22.0, ('QC', 'SA W'): 23.5, ('QC', 'SAW'): 25.0, ('QC', 'O-MC'): 26.0, ('QC', 'T'): 27.0, ('QC', 'BCAT'): 28.0, ('QC', 'H CR'): 28.5, ('QC', 'MAL'): 29.5, ('QC_1', 'QC'): 10.0, ('QC_1', 'GRS'): 10.5, ('QC_1', 'MC'): 9.0, ('QC_1', 'MC_1'): 8.0, ('QC_1', 'EDM'): 7.0, ('QC_1', 'GRO'): 7.0, ('QC_1', 'NI'): 5.5, ('QC_1', 'NI HOA'): 5.5, ('QC_1', 'GRI'): 7.0, ('QC_1', 'MI'): 7.5, ('QC_1', 'To'): 7.5, ('QC_1', 'SA B'): 8.5, ('QC_1', 'O-EWC'): 8.5, ('QC_1', 'SOB'): 7.5, ('QC_1', 'LA2'): 9.5, ('QC_1', 'EWC'): 10.0, ('QC_1', 'O_MI'): 12.0, ('QC_1', 'LA'): 12.0, ('QC_1', 'SA W'): 13.5, ('QC_1', 'SAW'): 15.0, ('QC_1', 'O-MC'): 16.0, ('QC_1', 'T'): 17.0, ('QC_1', 'BCAT'): 18.0, ('QC_1', 'H CR'): 18.5, ('QC_1', 'MAL'): 19.5, ('GRS', 'QC'): 10.5, ('GRS', 'QC_1'): 10.5, ('GRS', 'MC'): 1.5, ('GRS', 'MC_1'): 2.5, ('GRS', 'EDM'): 3.5, ('GRS', 'GRO'): 3.5, ('GRS', 'NI'): 5.0, ('GRS', 'NI HOA'): 6.0, 
('GRS', 'GRI'): 6.5, ('GRS', 'MI'): 7.0, ('GRS', 'To'): 8.0, ('GRS', 'SA B'): 8.0, ('GRS', 'O-EWC'): 8.0, ('GRS', 'SOB'): 10.0, ('GRS', 'LA2'): 10.0, ('GRS', 'EWC'): 11.5, ('GRS', 'O_MI'): 11.5, ('GRS', 'LA'): 13.5, 
('GRS', 'SA W'): 14.0, ('GRS', 'SAW'): 14.5, ('GRS', 'O-MC'): 15.5, ('GRS', 'T'): 16.5, ('GRS', 'BCAT'): 17.5, ('GRS', 'H CR'): 19.0, ('GRS', 'MAL'): 20.0, ('MC', 'QC'): 12.0, ('MC', 'QC_1'): 9.0, ('MC', 'GRS'): 1.5, ('MC', 'MC_1'): 1.0, ('MC', 'EDM'): 2.0, ('MC', 'GRO'): 2.0, ('MC', 'NI'): 3.5, ('MC', 'NI HOA'): 4.5, ('MC', 'GRI'): 5.0, ('MC', 'MI'): 5.5, ('MC', 'To'): 6.5, ('MC', 'SA B'): 6.5, ('MC', 'O-EWC'): 6.5, ('MC', 'SOB'): 8.5, ('MC', 'LA2'): 8.5, ('MC', 'EWC'): 10.0, ('MC', 'O_MI'): 10.0, ('MC', 'LA'): 12.0, ('MC', 'SA W'): 12.5, ('MC', 'SAW'): 13.0, ('MC', 'O-MC'): 14.0, ('MC', 'T'): 15.0, ('MC', 'BCAT'): 16.0, ('MC', 'H CR'): 17.5, ('MC', 'MAL'): 18.5, ('MC_1', 'QC'): 13.0, ('MC_1', 'QC_1'): 8.0, ('MC_1', 'GRS'): 2.5, ('MC_1', 'MC'): 1.0, ('MC_1', 'EDM'): 1.0, ('MC_1', 'GRO'): 1.0, ('MC_1', 'NI'): 2.5, ('MC_1', 'NI HOA'): 3.5, ('MC_1', 'GRI'): 4.0, ('MC_1', 'MI'): 4.5, ('MC_1', 'To'): 5.5, ('MC_1', 'SA B'): 5.5, ('MC_1', 'O-EWC'): 5.5, ('MC_1', 'SOB'): 7.5, ('MC_1', 'LA2'): 7.5, ('MC_1', 'EWC'): 9.0, ('MC_1', 'O_MI'): 9.0, ('MC_1', 'LA'): 11.0, ('MC_1', 'SA W'): 11.5, ('MC_1', 'SAW'): 12.0, ('MC_1', 'O-MC'): 13.0, ('MC_1', 'T'): 14.0, ('MC_1', 'BCAT'): 15.0, ('MC_1', 'H CR'): 16.5, ('MC_1', 'MAL'): 17.5, ('EDM', 'QC'): 12.0, ('EDM', 'QC_1'): 7.0, ('EDM', 'GRS'): 3.5, ('EDM', 'MC'): 2.0, ('EDM', 'MC_1'): 1.0, ('EDM', 'GRO'): 2.0, ('EDM', 'NI'): 2.5, ('EDM', 'NI HOA'): 3.5, ('EDM', 'GRI'): 5.0, ('EDM', 'MI'): 5.5, ('EDM', 'To'): 5.5, ('EDM', 'SA B'): 6.5, ('EDM', 'O-EWC'): 6.5, ('EDM', 'SOB'): 6.5, ('EDM', 'LA2'): 7.5, ('EDM', 'EWC'): 8.0, ('EDM', 'O_MI'): 10.0, ('EDM', 'LA'): 10.0, ('EDM', 'SA W'): 11.5, ('EDM', 'SAW'): 13.0, ('EDM', 'O-MC'): 14.0, ('EDM', 'T'): 15.0, ('EDM', 'BCAT'): 16.0, ('EDM', 'H CR'): 16.5, ('EDM', 'MAL'): 17.5, ('GRO', 'QC'): 14.0, ('GRO', 'QC_1'): 7.0, ('GRO', 'GRS'): 3.5, ('GRO', 'MC'): 2.0, ('GRO', 'MC_1'): 1.0, ('GRO', 'EDM'): 2.0, ('GRO', 'NI'): 1.5, ('GRO', 'NI HOA'): 2.5, ('GRO', 'GRI'): 3.0, ('GRO', 'MI'): 3.5, ('GRO', 'To'): 4.5, ('GRO', 'SA B'): 4.5, ('GRO', 'O-EWC'): 4.5, ('GRO', 'SOB'): 6.5, ('GRO', 'LA2'): 6.5, ('GRO', 'EWC'): 8.0, ('GRO', 'O_MI'): 8.0, ('GRO', 'LA'): 10.0, ('GRO', 'SA W'): 10.5, ('GRO', 'SAW'): 11.0, ('GRO', 'O-MC'): 12.0, ('GRO', 'T'): 13.0, ('GRO', 'BCAT'): 14.0, ('GRO', 'H CR'): 15.5, ('GRO', 'MAL'): 16.5, ('NI', 'QC'): 14.5, ('NI', 'QC_1'): 5.5, ('NI', 'GRS'): 5.0, 
('NI', 'MC'): 3.5, ('NI', 'MC_1'): 2.5, ('NI', 'EDM'): 2.5, ('NI', 'GRO'): 1.5, ('NI', 'NI HOA'): 1.0, ('NI', 'GRI'): 2.5, ('NI', 'MI'): 3.0, ('NI', 'To'): 3.0, ('NI', 'SA B'): 4.0, ('NI', 'O-EWC'): 4.0, ('NI', 'SOB'): 5.0, ('NI', 'LA2'): 5.0, ('NI', 'EWC'): 6.5, ('NI', 'O_MI'): 7.5, ('NI', 'LA'): 8.5, ('NI', 'SA W'): 9.0, ('NI', 'SAW'): 10.5, ('NI', 'O-MC'): 11.5, ('NI', 'T'): 12.5, ('NI', 'BCAT'): 13.5, ('NI', 'H CR'): 14.0, ('NI', 'MAL'): 15.0, ('NI HOA', 'QC'): 15.5, ('NI HOA', 'QC_1'): 5.5, ('NI HOA', 'GRS'): 6.0, ('NI HOA', 'MC'): 4.5, ('NI HOA', 'MC_1'): 3.5, ('NI HOA', 'EDM'): 3.5, ('NI HOA', 'GRO'): 2.5, ('NI HOA', 'NI'): 1.0, ('NI HOA', 'GRI'): 1.5, ('NI HOA', 'MI'): 2.0, ('NI HOA', 'To'): 2.0, ('NI HOA', 'SA B'): 3.0, ('NI HOA', 'O-EWC'): 3.0, ('NI HOA', 'SOB'): 4.0, ('NI HOA', 'LA2'): 4.0, ('NI HOA', 'EWC'): 5.5, ('NI HOA', 'O_MI'): 6.5, ('NI HOA', 'LA'): 7.5, ('NI HOA', 'SA W'): 8.0, ('NI HOA', 'SAW'): 9.5, ('NI HOA', 'O-MC'): 10.5, ('NI HOA', 'T'): 11.5, ('NI HOA', 'BCAT'): 12.5, ('NI HOA', 'H CR'): 13.0, ('NI HOA', 'MAL'): 14.0, ('GRI', 'QC'): 17.0, ('GRI', 'QC_1'): 7.0, ('GRI', 'GRS'): 6.5, ('GRI', 'MC'): 5.0, ('GRI', 'MC_1'): 4.0, ('GRI', 'EDM'): 5.0, ('GRI', 'GRO'): 3.0, ('GRI', 'NI'): 2.5, ('GRI', 'NI HOA'): 1.5, ('GRI', 'MI'): 0.5, ('GRI', 'To'): 1.5, ('GRI', 'SA B'): 1.5, ('GRI', 'O-EWC'): 1.5, ('GRI', 'SOB'): 3.5, ('GRI', 'LA2'): 3.5, ('GRI', 'EWC'): 5.0, ('GRI', 'O_MI'): 5.0, ('GRI', 'LA'): 7.0, ('GRI', 'SA W'): 7.5, ('GRI', 'SAW'): 8.0, ('GRI', 'O-MC'): 9.0, ('GRI', 'T'): 10.0, ('GRI', 'BCAT'): 11.0, ('GRI', 'H CR'): 12.5, ('GRI', 'MAL'): 13.5, ('MI', 'QC'): 17.5, ('MI', 'QC_1'): 7.5, ('MI', 'GRS'): 7.0, ('MI', 'MC'): 5.5, ('MI', 'MC_1'): 4.5, ('MI', 'EDM'): 5.5, ('MI', 'GRO'): 3.5, ('MI', 'NI'): 3.0, ('MI', 'NI HOA'): 2.0, ('MI', 'GRI'): 0.5, ('MI', 'To'): 1.0, ('MI', 'SA B'): 1.0, ('MI', 'O-EWC'): 1.0, ('MI', 'SOB'): 3.0, ('MI', 'LA2'): 3.0, ('MI', 'EWC'): 4.5, ('MI', 'O_MI'): 4.5, ('MI', 'LA'): 6.5, ('MI', 'SA W'): 7.0, ('MI', 'SAW'): 7.5, ('MI', 'O-MC'): 8.5, ('MI', 'T'): 9.5, ('MI', 'BCAT'): 10.5, ('MI', 'H CR'): 12.0, ('MI', 'MAL'): 13.0, ('To', 'QC'): 17.5, ('To', 'QC_1'): 7.5, ('To', 'GRS'): 8.0, ('To', 'MC'): 6.5, ('To', 'MC_1'): 5.5, ('To', 'EDM'): 5.5, ('To', 'GRO'): 4.5, ('To', 'NI'): 3.0, ('To', 'NI HOA'): 2.0, ('To', 'GRI'): 1.5, ('To', 'MI'): 1.0, ('To', 'SA B'): 1.0, ('To', 'O-EWC'): 1.0, ('To', 'SOB'): 2.0, ('To', 'LA2'): 2.0, ('To', 'EWC'): 3.5, ('To', 'O_MI'): 4.5, ('To', 'LA'): 5.5, ('To', 'SA W'): 6.0, ('To', 'SAW'): 7.5, ('To', 'O-MC'): 8.5, ('To', 'T'): 9.5, ('To', 'BCAT'): 10.5, ('To', 'H CR'): 11.0, ('To', 'MAL'): 12.0, ('SA B', 'QC'): 18.5, ('SA B', 'QC_1'): 8.5, ('SA B', 'GRS'): 8.0, ('SA B', 'MC'): 6.5, ('SA B', 'MC_1'): 5.5, ('SA B', 'EDM'): 6.5, ('SA B', 'GRO'): 4.5, ('SA B', 'NI'): 4.0, ('SA B', 'NI HOA'): 3.0, ('SA B', 'GRI'): 1.5, ('SA B', 'MI'): 1.0, ('SA B', 'To'): 1.0, ('SA B', 'O-EWC'): 0.0, ('SA B', 'SOB'): 2.0, ('SA B', 'LA2'): 2.0, ('SA B', 'EWC'): 3.5, ('SA B', 'O_MI'): 3.5, ('SA B', 'LA'): 5.5, ('SA B', 'SA W'): 6.0, ('SA B', 'SAW'): 6.5, ('SA B', 'O-MC'): 7.5, ('SA B', 'T'): 8.5, ('SA B', 'BCAT'): 9.5, ('SA B', 'H CR'): 11.0, ('SA B', 'MAL'): 12.0, ('O-EWC', 'QC'): 18.5, ('O-EWC', 'QC_1'): 8.5, ('O-EWC', 'GRS'): 8.0, ('O-EWC', 'MC'): 6.5, ('O-EWC', 'MC_1'): 5.5, ('O-EWC', 'EDM'): 6.5, ('O-EWC', 'GRO'): 4.5, ('O-EWC', 'NI'): 4.0, ('O-EWC', 'NI HOA'): 3.0, ('O-EWC', 'GRI'): 1.5, ('O-EWC', 'MI'): 1.0, ('O-EWC', 'To'): 1.0, ('O-EWC', 'SA B'): 0.0, ('O-EWC', 'SOB'): 2.0, ('O-EWC', 'LA2'): 2.0, ('O-EWC', 'EWC'): 3.5, ('O-EWC', 'O_MI'): 3.5, ('O-EWC', 'LA'): 5.5, ('O-EWC', 'SA W'): 6.0, ('O-EWC', 'SAW'): 6.5, ('O-EWC', 'O-MC'): 7.5, ('O-EWC', 'T'): 8.5, ('O-EWC', 'BCAT'): 9.5, ('O-EWC', 'H CR'): 11.0, ('O-EWC', 'MAL'): 12.0, ('SOB', 'QC'): 17.5, ('SOB', 'QC_1'): 7.5, ('SOB', 'GRS'): 10.0, ('SOB', 'MC'): 8.5, ('SOB', 'MC_1'): 7.5, ('SOB', 'EDM'): 6.5, ('SOB', 'GRO'): 6.5, ('SOB', 'NI'): 5.0, ('SOB', 'NI HOA'): 4.0, ('SOB', 'GRI'): 3.5, ('SOB', 'MI'): 3.0, ('SOB', 'To'): 2.0, ('SOB', 'SA B'): 2.0, ('SOB', 'O-EWC'): 2.0, ('SOB', 'LA2'): 2.0, ('SOB', 'EWC'): 2.5, ('SOB', 'O_MI'): 4.5, ('SOB', 'LA'): 4.5, ('SOB', 'SA W'): 6.0, ('SOB', 'SAW'): 7.5, ('SOB', 'O-MC'): 8.5, ('SOB', 'T'): 9.5, ('SOB', 'BCAT'): 10.5, ('SOB', 'H CR'): 11.0, ('SOB', 'MAL'): 12.0, ('LA2', 'QC'): 19.5, ('LA2', 'QC_1'): 9.5, ('LA2', 'GRS'): 10.0, ('LA2', 'MC'): 8.5, ('LA2', 'MC_1'): 7.5, ('LA2', 'EDM'): 7.5, ('LA2', 'GRO'): 6.5, ('LA2', 'NI'): 5.0, ('LA2', 'NI HOA'): 4.0, ('LA2', 'GRI'): 3.5, ('LA2', 'MI'): 3.0, ('LA2', 'To'): 2.0, ('LA2', 'SA B'): 2.0, ('LA2', 'O-EWC'): 2.0, ('LA2', 'SOB'): 2.0, ('LA2', 'EWC'): 1.5, ('LA2', 'O_MI'): 2.5, ('LA2', 'LA'): 3.5, ('LA2', 'SA W'): 4.0, ('LA2', 'SAW'): 5.5, ('LA2', 'O-MC'): 6.5, ('LA2', 'T'): 7.5, ('LA2', 'BCAT'): 8.5, ('LA2', 'H CR'): 9.0, ('LA2', 'MAL'): 10.0, ('EWC', 'QC'): 20.0, ('EWC', 'QC_1'): 10.0, ('EWC', 'GRS'): 11.5, ('EWC', 'MC'): 10.0, ('EWC', 'MC_1'): 9.0, ('EWC', 'EDM'): 8.0, ('EWC', 'GRO'): 8.0, ('EWC', 'NI'): 6.5, ('EWC', 'NI HOA'): 5.5, ('EWC', 'GRI'): 5.0, ('EWC', 'MI'): 4.5, ('EWC', 'To'): 3.5, ('EWC', 'SA B'): 3.5, ('EWC', 'O-EWC'): 3.5, ('EWC', 'SOB'): 2.5, ('EWC', 'LA2'): 1.5, ('EWC', 'O_MI'): 2.0, ('EWC', 'LA'): 2.0, ('EWC', 'SA W'): 3.5, ('EWC', 'SAW'): 5.0, ('EWC', 'O-MC'): 6.0, ('EWC', 'T'): 7.0, ('EWC', 'BCAT'): 8.0, ('EWC', 'H CR'): 8.5, ('EWC', 'MAL'): 9.5, ('O_MI', 'QC'): 22.0, ('O_MI', 'QC_1'): 12.0, ('O_MI', 'GRS'): 11.5, ('O_MI', 'MC'): 10.0, ('O_MI', 'MC_1'): 9.0, ('O_MI', 'EDM'): 10.0, ('O_MI', 'GRO'): 8.0, ('O_MI', 'NI'): 7.5, ('O_MI', 'NI HOA'): 6.5, ('O_MI', 'GRI'): 5.0, ('O_MI', 'MI'): 4.5, ('O_MI', 'To'): 4.5, ('O_MI', 'SA B'): 3.5, ('O_MI', 'O-EWC'): 3.5, ('O_MI', 'SOB'): 4.5, ('O_MI', 'LA2'): 2.5, ('O_MI', 'EWC'): 2.0, ('O_MI', 'LA'): 2.0, ('O_MI', 'SA W'): 2.5, ('O_MI', 'SAW'): 3.0, ('O_MI', 'O-MC'): 4.0, ('O_MI', 'T'): 5.0, ('O_MI', 'BCAT'): 6.0, ('O_MI', 'H CR'): 7.5, ('O_MI', 'MAL'): 8.5, ('LA', 'QC'): 22.0, ('LA', 'QC_1'): 12.0, ('LA', 'GRS'): 13.5, ('LA', 'MC'): 12.0, ('LA', 'MC_1'): 11.0, ('LA', 'EDM'): 10.0, ('LA', 'GRO'): 10.0, ('LA', 'NI'): 8.5, ('LA', 'NI HOA'): 7.5, ('LA', 'GRI'): 7.0, ('LA', 'MI'): 6.5, ('LA', 'To'): 5.5, ('LA', 'SA B'): 5.5, ('LA', 'O-EWC'): 5.5, ('LA', 'SOB'): 4.5, ('LA', 'LA2'): 3.5, ('LA', 'EWC'): 2.0, 
('LA', 'O_MI'): 2.0, ('LA', 'SA W'): 1.5, ('LA', 'SAW'): 3.0, ('LA', 'O-MC'): 4.0, ('LA', 'T'): 5.0, ('LA', 'BCAT'): 6.0, ('LA', 'H CR'): 6.5, ('LA', 'MAL'): 7.5, ('SA W', 'QC'): 23.5, ('SA W', 'QC_1'): 13.5, ('SA W', 'GRS'): 14.0, ('SA W', 'MC'): 12.5, ('SA W', 'MC_1'): 11.5, ('SA W', 'EDM'): 11.5, ('SA W', 'GRO'): 10.5, ('SA W', 'NI'): 9.0, ('SA W', 'NI HOA'): 8.0, ('SA W', 'GRI'): 7.5, ('SA W', 'MI'): 7.0, ('SA W', 'To'): 6.0, ('SA W', 'SA B'): 6.0, ('SA W', 'O-EWC'): 6.0, ('SA W', 'SOB'): 6.0, ('SA W', 'LA2'): 4.0, ('SA W', 'EWC'): 3.5, ('SA W', 'O_MI'): 2.5, ('SA W', 'LA'): 1.5, ('SA W', 'SAW'): 1.5, ('SA W', 'O-MC'): 2.5, ('SA W', 'T'): 3.5, ('SA W', 'BCAT'): 4.5, ('SA W', 'H CR'): 5.0, ('SA W', 'MAL'): 6.0, ('SAW', 'QC'): 25.0, ('SAW', 'QC_1'): 15.0, ('SAW', 'GRS'): 14.5, ('SAW', 'MC'): 13.0, ('SAW', 'MC_1'): 12.0, ('SAW', 'EDM'): 13.0, ('SAW', 
'GRO'): 11.0, ('SAW', 'NI'): 10.5, ('SAW', 'NI HOA'): 9.5, ('SAW', 'GRI'): 8.0, ('SAW', 'MI'): 7.5, ('SAW', 'To'): 7.5, ('SAW', 'SA B'): 6.5, ('SAW', 'O-EWC'): 6.5, ('SAW', 'SOB'): 7.5, ('SAW', 'LA2'): 5.5, ('SAW', 'EWC'): 5.0, ('SAW', 'O_MI'): 3.0, ('SAW', 'LA'): 3.0, ('SAW', 'SA W'): 1.5, ('SAW', 'O-MC'): 1.0, ('SAW', 'T'): 2.0, ('SAW', 'BCAT'): 3.0, ('SAW', 'H CR'): 4.5, ('SAW', 'MAL'): 5.5, ('O-MC', 'QC'): 26.0, ('O-MC', 'QC_1'): 16.0, ('O-MC', 'GRS'): 15.5, ('O-MC', 'MC'): 14.0, ('O-MC', 'MC_1'): 13.0, ('O-MC', 'EDM'): 14.0, ('O-MC', 'GRO'): 12.0, ('O-MC', 'NI'): 11.5, ('O-MC', 'NI HOA'): 10.5, ('O-MC', 'GRI'): 9.0, ('O-MC', 'MI'): 8.5, ('O-MC', 'To'): 8.5, ('O-MC', 'SA B'): 7.5, ('O-MC', 'O-EWC'): 7.5, ('O-MC', 'SOB'): 8.5, ('O-MC', 'LA2'): 6.5, ('O-MC', 'EWC'): 6.0, ('O-MC', 'O_MI'): 4.0, ('O-MC', 'LA'): 4.0, ('O-MC', 'SA W'): 2.5, ('O-MC', 'SAW'): 1.0, ('O-MC', 'T'): 1.0, ('O-MC', 'BCAT'): 2.0, ('O-MC', 'H CR'): 3.5, ('O-MC', 'MAL'): 4.5, ('T', 'QC'): 27.0, ('T', 'QC_1'): 17.0, ('T', 'GRS'): 16.5, ('T', 'MC'): 15.0, ('T', 'MC_1'): 14.0, ('T', 'EDM'): 15.0, ('T', 'GRO'): 13.0, ('T', 'NI'): 12.5, ('T', 'NI HOA'): 11.5, ('T', 'GRI'): 10.0, ('T', 'MI'): 9.5, ('T', 'To'): 9.5, ('T', 'SA B'): 8.5, ('T', 'O-EWC'): 8.5, ('T', 'SOB'): 9.5, ('T', 'LA2'): 7.5, ('T', 'EWC'): 7.0, ('T', 'O_MI'): 5.0, ('T', 'LA'): 5.0, ('T', 'SA W'): 3.5, ('T', 'SAW'): 2.0, ('T', 'O-MC'): 1.0, ('T', 'BCAT'): 1.0, ('T', 'H CR'): 2.5, ('T', 'MAL'): 3.5, ('BCAT', 'QC'): 28.0, ('BCAT', 'QC_1'): 18.0, ('BCAT', 'GRS'): 17.5, ('BCAT', 'MC'): 16.0, ('BCAT', 'MC_1'): 15.0, ('BCAT', 'EDM'): 16.0, ('BCAT', 'GRO'): 14.0, ('BCAT', 'NI'): 13.5, ('BCAT', 'NI HOA'): 12.5, ('BCAT', 'GRI'): 11.0, ('BCAT', 'MI'): 10.5, ('BCAT', 'To'): 10.5, ('BCAT', 'SA B'): 9.5, ('BCAT', 'O-EWC'): 9.5, ('BCAT', 'SOB'): 10.5, ('BCAT', 'LA2'): 8.5, ('BCAT', 'EWC'): 8.0, ('BCAT', 'O_MI'): 6.0, ('BCAT', 'LA'): 6.0, ('BCAT', 'SA W'): 4.5, ('BCAT', 'SAW'): 3.0, ('BCAT', 'O-MC'): 2.0, ('BCAT', 'T'): 1.0, ('BCAT', 'H CR'): 1.5, ('BCAT', 'MAL'): 2.5, ('H CR', 'QC'): 28.5, ('H CR', 'QC_1'): 18.5, ('H CR', 'GRS'): 19.0, ('H CR', 'MC'): 17.5, ('H CR', 'MC_1'): 16.5, ('H CR', 'EDM'): 16.5, ('H CR', 'GRO'): 15.5, ('H CR', 'NI'): 14.0, ('H CR', 'NI HOA'): 13.0, ('H CR', 'GRI'): 12.5, ('H CR', 'MI'): 12.0, ('H CR', 'To'): 11.0, ('H CR', 'SA B'): 11.0, ('H CR', 'O-EWC'): 11.0, ('H CR', 'SOB'): 11.0, ('H CR', 'LA2'): 9.0, ('H CR', 'EWC'): 8.5, ('H CR', 'O_MI'): 7.5, ('H CR', 'LA'): 6.5, ('H CR', 'SA W'): 5.0, ('H CR', 'SAW'): 4.5, ('H CR', 'O-MC'): 3.5, ('H CR', 'T'): 2.5, ('H CR', 'BCAT'): 1.5, ('H CR', 'MAL'): 1.0, ('MAL', 'QC'): 29.5, ('MAL', 'QC_1'): 19.5, ('MAL', 'GRS'): 20.0, ('MAL', 'MC'): 18.5, ('MAL', 'MC_1'): 17.5, ('MAL', 'EDM'): 17.5, ('MAL', 'GRO'): 16.5, ('MAL', 'NI'): 15.0, ('MAL', 'NI HOA'): 14.0, ('MAL', 'GRI'): 13.5, ('MAL', 'MI'): 13.0, ('MAL', 'To'): 12.0, ('MAL', 'SA B'): 12.0, ('MAL', 'O-EWC'): 12.0, ('MAL', 'SOB'): 12.0, ('MAL', 'LA2'): 10.0, ('MAL', 'EWC'): 9.5, ('MAL', 'O_MI'): 8.5, ('MAL', 'LA'): 7.5, ('MAL', 'SA W'): 
6.0, ('MAL', 'SAW'): 5.5, ('MAL', 'O-MC'): 4.5, ('MAL', 'T'): 3.5, ('MAL', 'BCAT'): 2.5, ('MAL', 'H CR'): 1.0}
Tổng chi phí: 2490952.0
'''


#---------------------------------------------------------------------------
#Vẽ biểu đồ bố trí mặt bằng các máy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re

# Hàm vẽ mặt bằng và kiểm tra chồng lấn
def plot_layout(machine_positions, machine_sizes):
    # Tạo figure và axes với kích thước phù hợp
    fig, ax = plt.subplots(figsize=(12, 6))

    # Đặt giới hạn trục
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 20)
    ax.set_xlabel("Chiều rộng (m)")
    ax.set_ylabel("Chiều dài (m)")
    ax.set_title("Bố trí mặt bằng các máy (40m x 20m)")

    # Vẽ lưới nền
    ax.grid(True, linestyle='--', alpha=0.7)

    # Danh sách để lưu các vùng đã vẽ (để kiểm tra chồng lấn)
    occupied_regions = []

    # Vẽ từng máy dưới dạng hình chữ nhật
    for machine, (x_center, y_center) in machine_positions.items():
        # Xử lý tên máy để lấy kích thước
        base_name = re.sub(r'_\d+$', '', machine)
        
        if base_name not in machine_sizes:
            print(f"Cảnh báo: Không tìm thấy kích thước cho máy '{base_name}' (máy gốc: '{machine}')")
            continue
        
        # Lấy kích thước máy
        wid, length = machine_sizes[base_name]
        wid = float(wid)
        length = float(length)

        # Tính tọa độ góc dưới bên trái và trên bên phải
        x_lower_left = x_center - wid / 2
        y_lower_left = y_center - length / 2
        x_upper_right = x_center + wid / 2
        y_upper_right = y_center + length / 2

        # Kiểm tra chồng lấn với các vùng đã vẽ
        for (prev_machine, prev_x1, prev_y1, prev_x2, prev_y2) in occupied_regions:
            if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                    y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                print(f"Cảnh báo: Máy '{machine}' chồng lấn với '{prev_machine}'!")

        # Lưu vùng của máy hiện tại
        occupied_regions.append((machine, x_lower_left, y_lower_left, x_upper_right, y_upper_right))

        # Tạo hình chữ nhật
        rect = Rectangle(
            (x_lower_left, y_lower_left),
            wid,
            length,
            edgecolor='black',
            facecolor='lightblue',
            alpha=0.6
        )
        ax.add_patch(rect)

        # Thêm tên máy vào giữa hình chữ nhật
        ax.text(
            x_center, y_center,
            machine,
            ha='center',
            va='center',
            fontsize=8,
            weight='bold'
        )

    # Tùy chỉnh tỷ lệ trục
    ax.set_aspect('equal', adjustable='box')

    # Hiển thị biểu đồ
    plt.show()

# Gọi hàm vẽ mặt bằng
plot_layout(machine_positions, machine_sizes)

#=====================================
# Thay đoạn bố trí máy bằng thuật toán đơn giản hơn
machine_positions = {}
x, y = 0, 0
row_height = 0

for machine in arranged_machines:
    wid, length = machine_sizes[machine]
    wid = int(round(wid))
    length = int(round(length))
    count = machine_counts[machine]
    
    for i in range(count):
        # Kiểm tra xem có đủ chỗ trong hàng hiện tại không
        if x + wid > 40:
            x = 0
            y += row_height
            row_height = 0
        
        # Kiểm tra giới hạn chiều cao
        if y + length > 20:
            print(f"Không đủ diện tích để bố trí máy {machine}_{i}")
            break
        
        # Đặt máy
        grid[y:y+length, x:x+wid] = f"{machine}_{i}" if i > 0 else machine
        machine_positions[f"{machine}_{i}" if i > 0 else machine] = (x + wid/2, y + length/2)
        
        # Cập nhật vị trí
        x += wid
        row_height = max(row_height, length)

    if y + row_height > 20:
        print(f"Dừng bố trí tại máy {machine}: Không đủ diện tích")
        break

