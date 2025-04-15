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
machine_counts["QC"] = 1 



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


#--------------------------------------------------
# 5. Bố trí máy trên mặt bằng (40m x 20m)
# Khởi tạo lưới 40m x 20m (chia thành ô 1m x 1m)

machine_counts = {
    'MAL': 3, 'SH': 1, 'MC': 2, 'QC': 1, 'GRS': 1, 'MI': 1, 'LA': 1, 'SA W': 1,
    'GCN': 1, 'T': 1, 'EDM': 1, 'GRO': 1, 'EWC': 1, 'O-EWC': 1, 'WELD': 1, 'To': 1,
    'SAW': 1, 'LA2': 1, 'SOB': 1, 'O_LA': 1, 'GRI': 1, 'SEH-CR': 1, 'DC': 1, 'SA B': 1,
    'O_MC': 1, 'Đ/CHẤT': 1, 'O-MC': 1, 'NI': 1, 'H CR': 1, 'O_MI': 1, 'BCAT': 1, 'NGUOI': 1, 'NI HOA': 1
}
arranged_machines = list(machine_counts.keys())

# 5. Bố trí máy trên mặt bằng (40m x 20m)
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
closeness_dict = {(row.iloc[0], col): closeness_df.at[i, col] 
                  for i, row in closeness_df.iterrows() 
                  for col in closeness_df.columns if col != closeness_df.columns[0]}
flow_dict = {(row.iloc[0], col): flow_df.at[i, col] 
             for i, row in flow_df.iterrows() 
             for col in flow_df.columns if col != flow_df.columns[0]}

# Định nghĩa trọng số cho closeness
closeness_weights = {'A': 5, 'E': 4, 'I': 3, 'O': 2, 'U': 1}

# Tính điểm ưu tiên cho từng cặp máy
machine_pairs = []
for (m1, m2), closeness in closeness_dict.items():
    if m1 == m2 or closeness == '-':
        continue
    if closeness == 'U':
        continue
    flow = flow_dict.get((m1, m2), 0)
    score = closeness_weights[closeness] * flow * 1000
    if m1 in machine_counts and m2 in machine_counts:
        machine_pairs.append((m1, m2, score))

# Sắp xếp các cặp máy theo điểm ưu tiên giảm dần
machine_pairs.sort(key=lambda x: x[2], reverse=True)

# Bố trí máy
machine_positions = {}
occupied_regions = []
placed_machines = set()
remaining_machines = set()

# Thêm khoảng cách an toàn theo TCVN 4604:2012
buffer = 0.8  # Khoảng cách giữa các máy: 0,8 m
wall_buffer = 0.6  # Khoảng cách giữa máy và tường: 0,6 m
main_path_width = 2.4  # Chiều rộng lối đi chính: 2,4 m

# Tạo danh sách tất cả máy
for machine in arranged_machines:
    for i in range(machine_counts[machine]):
        machine_name = f"{machine}_{i}" if i > 0 else machine
        remaining_machines.add(machine_name)

# Đặt cặp máy có điểm ưu tiên cao nhất trước (MAL và SH)
m1, m2, _ = machine_pairs[0]  # MAL và SH
for i in range(machine_counts[m1]):
    m1_name = f"{m1}_{i}" if i > 0 else m1
    wid, length = machine_sizes[m1]
    wid_cells = int((wid + buffer) * 10)
    length_cells = int((length + buffer) * 10)

    # Điều chỉnh giới hạn không gian đặt máy dựa trên kích thước máy và khoảng cách tường
    min_x = wall_buffer + wid / 2
    max_x = 40 - wall_buffer - wid / 2
    min_y = wall_buffer + length / 2
    max_y = 20 - wall_buffer - length / 2

    # Đặt ở vị trí gần trung tâm nhưng lệch về bên trái để tối ưu với LA
    if m1_name == 'MAL':
        x = 15 - wid / 2
        y = 10 - length / 2
    else:
        # Đặt MAL_1 và MAL_2 gần MAL, nhưng thêm buffer để tránh chồng lấn
        prev_machine = f"{m1}_{i-1}" if i > 1 else m1
        prev_x, prev_y = machine_positions[prev_machine]
        prev_wid, prev_length = machine_sizes[m1]
        x = prev_x - (prev_wid / 2 + wid / 2 + buffer)
        y = 10 - length / 2

    # Đảm bảo máy không vượt quá giới hạn tường
    x = max(min_x, min(max_x, x))
    y = max(min_y, min(max_y, y))

    x_cells = int(x * 10)
    y_cells = int(y * 10)

    overlap = False
    for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
        x_lower_left = x - wid / 2
        y_lower_left = y - length / 2
        x_upper_right = x + wid / 2 + buffer
        y_upper_right = y + length / 2 + buffer
        if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
            overlap = True
            break

    if not overlap:
        region = grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells]
        if region.size == wid_cells * length_cells and np.all(region == None):
            grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells] = m1_name
            machine_positions[m1_name] = (x, y)
            occupied_regions.append((m1_name, x - wid / 2, y - length / 2, x + wid / 2 + buffer, y + length / 2 + buffer))
            placed_machines.add(m1_name)
            remaining_machines.remove(m1_name)

for j in range(machine_counts[m2]):
    m2_name = f"{m2}_{j}" if j > 0 else m2
    wid, length = machine_sizes[m2]
    wid_cells = int((wid + buffer) * 10)
    length_cells = int((length + buffer) * 10)

    # Điều chỉnh giới hạn không gian đặt máy
    min_x = wall_buffer + wid / 2
    max_x = 40 - wall_buffer - wid / 2
    min_y = wall_buffer + length / 2
    max_y = 20 - wall_buffer - length / 2

    best_pos = None
    best_cost = float('inf')

    for y in np.arange(min_y, max_y + 0.1, 0.1):
        for x in np.arange(min_x, max_x + 0.1, 0.1):
            x = round(x, 1)
            y = round(y, 1)
            x_lower_left = x - wid / 2
            y_lower_left = y - length / 2
            x_upper_right = x + wid / 2 + buffer
            y_upper_right = y + length / 2 + buffer

            # Kiểm tra lối đi chính
            path_ok = True
            for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                # Kiểm tra lối đi chính theo chiều ngang (giữa các hàng)
                if abs(y_lower_left - prev_y2) < main_path_width and abs(y_upper_right - prev_y1) < main_path_width:
                    if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2):
                        path_ok = False
                        break
                # Kiểm tra lối đi chính theo chiều dọc (giữa các cột)
                if abs(x_lower_left - prev_x2) < main_path_width and abs(x_upper_right - prev_x1) < main_path_width:
                    if not (y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                        path_ok = False
                        break

            overlap = False
            for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                        y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                    overlap = True
                    break

            if not overlap and path_ok:
                x_cells = int(x * 10)
                y_cells = int(y * 10)
                region = grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells]
                if region.size == wid_cells * length_cells and np.all(region == None):
                    cost = 0
                    for placed_machine, (px, py) in machine_positions.items():
                        base_placed = placed_machine if '_' not in placed_machine else placed_machine.split('_')[0]
                        if base_placed == m1:
                            flow = flow_dict.get((m1, m2), 0)
                            closeness = closeness_dict.get((m1, m2), 'U')
                            dist = np.sqrt((x - px)**2 + (y - py)**2)
                            cost += (flow * 1000 + closeness_weights[closeness]) * dist

                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y)

    if best_pos:
        x, y = best_pos
        x_cells = int(x * 10)
        y_cells = int(y * 10)
        grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells] = m2_name
        machine_positions[m2_name] = (x, y)
        occupied_regions.append((m2_name, x - wid / 2, y - length / 2, x + wid / 2 + buffer, y + length / 2 + buffer))
        placed_machines.add(m2_name)
        remaining_machines.remove(m2_name)

# Đặt cặp máy tiếp theo (MC và QC)
m1, m2, _ = machine_pairs[1]  # MC và QC
for i in range(machine_counts[m1]):
    m1_name = f"{m1}_{i}" if i > 0 else m1
    if m1_name in placed_machines:
        continue
    wid, length = machine_sizes[m1]
    wid_cells = int((wid + buffer) * 10)
    length_cells = int((length + buffer) * 10)

    # Điều chỉnh giới hạn không gian đặt máy
    min_x = wall_buffer + wid / 2
    max_x = 40 - wall_buffer - wid / 2
    min_y = wall_buffer + length / 2
    max_y = 20 - wall_buffer - length / 2

    best_pos = None
    best_cost = float('inf')

    for y in np.arange(min_y, max_y + 0.1, 0.1):
        for x in np.arange(min_x, max_x + 0.1, 0.1):
            x = round(x, 1)
            y = round(y, 1)
            x_lower_left = x - wid / 2
            y_lower_left = y - length / 2
            x_upper_right = x + wid / 2 + buffer
            y_upper_right = y + length / 2 + buffer

            # Kiểm tra lối đi chính
            path_ok = True
            for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                if abs(y_lower_left - prev_y2) < main_path_width and abs(y_upper_right - prev_y1) < main_path_width:
                    if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2):
                        path_ok = False
                        break
                if abs(x_lower_left - prev_x2) < main_path_width and abs(x_upper_right - prev_x1) < main_path_width:
                    if not (y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                        path_ok = False
                        break

            overlap = False
            for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                        y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                    overlap = True
                    break

            if not overlap and path_ok:
                x_cells = int(x * 10)
                y_cells = int(y * 10)
                region = grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells]
                if region.size == wid_cells * length_cells and np.all(region == None):
                    cost = 0
                    for placed_machine, (px, py) in machine_positions.items():
                        base_placed = placed_machine if '_' not in placed_machine else placed_machine.split('_')[0]
                        if base_placed == m2:
                            flow = flow_dict.get((m1, m2), 0)
                            closeness = closeness_dict.get((m1, m2), 'U')
                            dist = np.sqrt((x - px)**2 + (y - py)**2)
                            cost += (flow * 1000 + closeness_weights[closeness]) * dist
                        if base_placed == 'SH':
                            flow = flow_dict.get(('SH', m1), 0)
                            closeness = closeness_dict.get(('SH', m1), 'U')
                            dist = np.sqrt((x - px)**2 + (y - py)**2)
                            cost += (flow * 1000 + closeness_weights[closeness]) * dist

                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y)

    if best_pos:
        x, y = best_pos
        x_cells = int(x * 10)
        y_cells = int(y * 10)
        grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells] = m1_name
        machine_positions[m1_name] = (x, y)
        occupied_regions.append((m1_name, x - wid / 2, y - length / 2, x + wid / 2 + buffer, y + length / 2 + buffer))
        placed_machines.add(m1_name)
        remaining_machines.remove(m1_name)

for j in range(machine_counts[m2]):
    m2_name = f"{m2}_{j}" if j > 0 else m2
    if m2_name in placed_machines:
        continue
    wid, length = machine_sizes[m2]
    wid_cells = int((wid + buffer) * 10)
    length_cells = int((length + buffer) * 10)

    # Điều chỉnh giới hạn không gian đặt máy
    min_x = wall_buffer + wid / 2
    max_x = 40 - wall_buffer - wid / 2
    min_y = wall_buffer + length / 2
    max_y = 20 - wall_buffer - length / 2

    best_pos = None
    best_cost = float('inf')

    for y in np.arange(min_y, max_y + 0.1, 0.1):
        for x in np.arange(min_x, max_x + 0.1, 0.1):
            x = round(x, 1)
            y = round(y, 1)
            x_lower_left = x - wid / 2
            y_lower_left = y - length / 2
            x_upper_right = x + wid / 2 + buffer
            y_upper_right = y + length / 2 + buffer

            # Kiểm tra lối đi chính
            path_ok = True
            for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                if abs(y_lower_left - prev_y2) < main_path_width and abs(y_upper_right - prev_y1) < main_path_width:
                    if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2):
                        path_ok = False
                        break
                if abs(x_lower_left - prev_x2) < main_path_width and abs(x_upper_right - prev_x1) < main_path_width:
                    if not (y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                        path_ok = False
                        break

            overlap = False
            for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                        y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                    overlap = True
                    break

            if not overlap and path_ok:
                x_cells = int(x * 10)
                y_cells = int(y * 10)
                region = grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells]
                if region.size == wid_cells * length_cells and np.all(region == None):
                    cost = 0
                    for placed_machine, (px, py) in machine_positions.items():
                        base_placed = placed_machine if '_' not in placed_machine else placed_machine.split('_')[0]
                        if base_placed == m1:
                            flow = flow_dict.get((m1, m2), 0)
                            closeness = closeness_dict.get((m1, m2), 'U')
                            dist = np.sqrt((x - px)**2 + (y - py)**2)
                            cost += (flow * 1000 + closeness_weights[closeness]) * dist
                        if base_placed == 'GRS':
                            flow = flow_dict.get(('GRS', m2), 0)
                            closeness = closeness_dict.get(('GRS', m2), 'U')
                            dist = np.sqrt((x - px)**2 + (y - py)**2)
                            cost += (flow * 1000 + closeness_weights[closeness]) * dist

                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y)

    if best_pos:
        x, y = best_pos
        x_cells = int(x * 10)
        y_cells = int(y * 10)
        grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells] = m2_name
        machine_positions[m2_name] = (x, y)
        occupied_regions.append((m2_name, x - wid / 2, y - length / 2, x + wid / 2 + buffer, y + length / 2 + buffer))
        placed_machines.add(m2_name)
        remaining_machines.remove(m2_name)

# Đặt cặp GRS và QC
m1, m2, _ = machine_pairs[3]  # GRS và QC
for i in range(machine_counts[m1]):
    m1_name = f"{m1}_{i}" if i > 0 else m1
    if m1_name in placed_machines:
        continue
    wid, length = machine_sizes[m1]
    wid_cells = int((wid + buffer) * 10)
    length_cells = int((length + buffer) * 10)

    # Điều chỉnh giới hạn không gian đặt máy
    min_x = wall_buffer + wid / 2
    max_x = 40 - wall_buffer - wid / 2
    min_y = wall_buffer + length / 2
    max_y = 20 - wall_buffer - length / 2

    best_pos = None
    best_cost = float('inf')

    for y in np.arange(min_y, max_y + 0.1, 0.1):
        for x in np.arange(min_x, max_x + 0.1, 0.1):
            x = round(x, 1)
            y = round(y, 1)
            x_lower_left = x - wid / 2
            y_lower_left = y - length / 2
            x_upper_right = x + wid / 2 + buffer
            y_upper_right = y + length / 2 + buffer

            # Kiểm tra lối đi chính
            path_ok = True
            for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                if abs(y_lower_left - prev_y2) < main_path_width and abs(y_upper_right - prev_y1) < main_path_width:
                    if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2):
                        path_ok = False
                        break
                if abs(x_lower_left - prev_x2) < main_path_width and abs(x_upper_right - prev_x1) < main_path_width:
                    if not (y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                        path_ok = False
                        break

            overlap = False
            for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                        y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                    overlap = True
                    break

            if not overlap and path_ok:
                x_cells = int(x * 10)
                y_cells = int(y * 10)
                region = grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells]
                if region.size == wid_cells * length_cells and np.all(region == None):
                    cost = 0
                    for placed_machine, (px, py) in machine_positions.items():
                        base_placed = placed_machine if '_' not in placed_machine else placed_machine.split('_')[0]
                        if base_placed == m2:
                            flow = flow_dict.get((m1, m2), 0)
                            closeness = closeness_dict.get((m1, m2), 'U')
                            dist = np.sqrt((x - px)**2 + (y - py)**2)
                            cost += (flow * 1000 + closeness_weights[closeness]) * dist

                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y)

    if best_pos:
        x, y = best_pos
        x_cells = int(x * 10)
        y_cells = int(y * 10)
        grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells] = m1_name
        machine_positions[m1_name] = (x, y)
        occupied_regions.append((m1_name, x - wid / 2, y - length / 2, x + wid / 2 + buffer, y + length / 2 + buffer))
        placed_machines.add(m1_name)
        remaining_machines.remove(m1_name)

# Đặt các cặp máy tiếp theo theo machine_pairs
for m1, m2, _ in machine_pairs[4:]:
    for i in range(machine_counts[m1]):
        m1_name = f"{m1}_{i}" if i > 0 else m1
        if m1_name in placed_machines:
            continue
        wid, length = machine_sizes[m1]
        wid_cells = int((wid + buffer) * 10)
        length_cells = int((length + buffer) * 10)

        # Điều chỉnh giới hạn không gian đặt máy
        min_x = wall_buffer + wid / 2
        max_x = 40 - wall_buffer - wid / 2
        min_y = wall_buffer + length / 2
        max_y = 20 - wall_buffer - length / 2

        best_pos = None
        best_cost = float('inf')

        for y in np.arange(min_y, max_y + 0.1, 0.1):
            for x in np.arange(min_x, max_x + 0.1, 0.1):
                x = round(x, 1)
                y = round(y, 1)
                x_lower_left = x - wid / 2
                y_lower_left = y - length / 2
                x_upper_right = x + wid / 2 + buffer
                y_upper_right = y + length / 2 + buffer

                # Kiểm tra lối đi chính
                path_ok = True
                for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                    if abs(y_lower_left - prev_y2) < main_path_width and abs(y_upper_right - prev_y1) < main_path_width:
                        if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2):
                            path_ok = False
                            break
                    if abs(x_lower_left - prev_x2) < main_path_width and abs(x_upper_right - prev_x1) < main_path_width:
                        if not (y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                            path_ok = False
                            break

                overlap = False
                for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                    if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                            y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                        overlap = True
                        break

                if not overlap and path_ok:
                    x_cells = int(x * 10)
                    y_cells = int(y * 10)
                    region = grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells]
                    if region.size == wid_cells * length_cells and np.all(region == None):
                        cost = 0
                        for placed_machine, (px, py) in machine_positions.items():
                            base_placed = placed_machine if '_' not in placed_machine else placed_machine.split('_')[0]
                            if base_placed == m2:
                                flow = flow_dict.get((m1, m2), 0)
                                closeness = closeness_dict.get((m1, m2), 'U')
                                dist = np.sqrt((x - px)**2 + (y - py)**2)
                                cost += (flow * 1000 + closeness_weights[closeness]) * dist

                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y)

        if best_pos:
            x, y = best_pos
            x_cells = int(x * 10)
            y_cells = int(y * 10)
            grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells] = m1_name
            machine_positions[m1_name] = (x, y)
            occupied_regions.append((m1_name, x - wid / 2, y - length / 2, x + wid / 2 + buffer, y + length / 2 + buffer))
            placed_machines.add(m1_name)
            remaining_machines.remove(m1_name)

    for j in range(machine_counts[m2]):
        m2_name = f"{m2}_{j}" if j > 0 else m2
        if m2_name in placed_machines:
            continue
        wid, length = machine_sizes[m2]
        wid_cells = int((wid + buffer) * 10)
        length_cells = int((length + buffer) * 10)

        # Điều chỉnh giới hạn không gian đặt máy
        min_x = wall_buffer + wid / 2
        max_x = 40 - wall_buffer - wid / 2
        min_y = wall_buffer + length / 2
        max_y = 20 - wall_buffer - length / 2

        best_pos = None
        best_cost = float('inf')

        for y in np.arange(min_y, max_y + 0.1, 0.1):
            for x in np.arange(min_x, max_x + 0.1, 0.1):
                x = round(x, 1)
                y = round(y, 1)
                x_lower_left = x - wid / 2
                y_lower_left = y - length / 2
                x_upper_right = x + wid / 2 + buffer
                y_upper_right = y + length / 2 + buffer

                # Kiểm tra lối đi chính
                path_ok = True
                for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                    if abs(y_lower_left - prev_y2) < main_path_width and abs(y_upper_right - prev_y1) < main_path_width:
                        if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2):
                            path_ok = False
                            break
                    if abs(x_lower_left - prev_x2) < main_path_width and abs(x_upper_right - prev_x1) < main_path_width:
                        if not (y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                            path_ok = False
                            break

                overlap = False
                for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                    if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                            y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                        overlap = True
                        break

                if not overlap and path_ok:
                    x_cells = int(x * 10)
                    y_cells = int(y * 10)
                    region = grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells]
                    if region.size == wid_cells * length_cells and np.all(region == None):
                        cost = 0
                        for placed_machine, (px, py) in machine_positions.items():
                            base_placed = placed_machine if '_' not in placed_machine else placed_machine.split('_')[0]
                            if base_placed == m1:
                                flow = flow_dict.get((m1, m2), 0)
                                closeness = closeness_dict.get((m1, m2), 'U')
                                dist = np.sqrt((x - px)**2 + (y - py)**2)
                                cost += (flow * 1000 + closeness_weights[closeness]) * dist

                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y)

        if best_pos:
            x, y = best_pos
            x_cells = int(x * 10)
            y_cells = int(y * 10)
            grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells] = m2_name
            machine_positions[m2_name] = (x, y)
            occupied_regions.append((m2_name, x - wid / 2, y - length / 2, x + wid / 2 + buffer, y + length / 2 + buffer))
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
    
    wid_cells = int((wid + buffer) * 10)
    length_cells = int((length + buffer) * 10)

    # Điều chỉnh giới hạn không gian đặt máy
    min_x = wall_buffer + wid / 2
    max_x = 40 - wall_buffer - wid / 2
    min_y = wall_buffer + length / 2
    max_y = 20 - wall_buffer - length / 2

    placed = False
    for y in np.arange(min_y, max_y + 0.1, 0.1):
        for x in np.arange(min_x, max_x + 0.1, 0.1):
            x = round(x, 1)
            y = round(y, 1)
            x_lower_left = x - wid / 2
            y_lower_left = y - length / 2
            x_upper_right = x + wid / 2 + buffer
            y_upper_right = y + length / 2 + buffer

            # Kiểm tra lối đi chính
            path_ok = True
            for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                if abs(y_lower_left - prev_y2) < main_path_width and abs(y_upper_right - prev_y1) < main_path_width:
                    if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2):
                        path_ok = False
                        break
                if abs(x_lower_left - prev_x2) < main_path_width and abs(x_upper_right - prev_x1) < main_path_width:
                    if not (y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                        path_ok = False
                        break

            overlap = False
            for _, prev_x1, prev_y1, prev_x2, prev_y2 in occupied_regions:
                if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                        y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                    overlap = True
                    break

            if not overlap and path_ok:
                x_cells = int(x * 10)
                y_cells = int(y * 10)
                region = grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells]
                if region.size == wid_cells * length_cells and np.all(region == None):
                    grid[y_cells:y_cells+length_cells, x_cells:x_cells+wid_cells] = machine_name
                    machine_positions[machine_name] = (x, y)
                    occupied_regions.append((machine_name, x_lower_left, y_lower_left, x_upper_right, y_upper_right))
                    placed_machines.add(machine_name)
                    remaining_machines.remove(machine_name)
                    placed = True
                    break
        if placed:
            break


# Kiểm tra kết quả
print("Tọa độ máy:", machine_positions)


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


#---------------------------------------------------------------------------
#7 .Vẽ biểu đồ bố trí mặt bằng các máy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re

# Hàm vẽ mặt bằng
def plot_layout(machine_positions, machine_sizes):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 20)
    ax.set_xlabel("Chiều rộng (m)")
    ax.set_ylabel("Chiều dài (m)")
    ax.set_title("Bố trí mặt bằng các máy (40m x 20m)")
    ax.grid(True, linestyle='--', alpha=0.7)
    occupied_regions = []

    for machine, (x_center, y_center) in machine_positions.items():
        base_name = re.sub(r'_\d+$', '', machine)
        if base_name not in machine_sizes:
            print(f"Cảnh báo: Không tìm thấy kích thước cho máy '{base_name}' (máy gốc: '{machine}')")
            continue
        wid, length = machine_sizes[base_name]
        wid = float(wid)
        length = float(length)
        x_lower_left = x_center - wid / 2
        y_lower_left = y_center - length / 2
        x_upper_right = x_center + wid / 2
        y_upper_right = y_center + length / 2

        for (prev_machine, prev_x1, prev_y1, prev_x2, prev_y2) in occupied_regions:
            if not (x_upper_right <= prev_x1 or x_lower_left >= prev_x2 or
                    y_upper_right <= prev_y1 or y_lower_left >= prev_y2):
                print(f"Cảnh báo: Máy '{machine}' chồng lấn với '{prev_machine}'!")

        occupied_regions.append((machine, x_lower_left, y_lower_left, x_upper_right, y_upper_right))
        rect = Rectangle(
            (x_lower_left, y_lower_left),
            wid,
            length,
            edgecolor='black',
            facecolor='lightblue',
            alpha=0.6
        )
        ax.add_patch(rect)
        ax.text(
            x_center, y_center,
            machine,
            ha='center',
            va='center',
            fontsize=8,
            weight='bold'
        )

    ax.set_aspect('equal', adjustable='box')
    plt.show()

# Gọi hàm vẽ mặt bằng
plot_layout(machine_positions, machine_sizes)
