import numpy as np

def find_zero_zone(chunk, start_index, search_length, search_window_size=11):
    zone = chunk[start_index:start_index + search_length]

    zero_threshold = 1.0e-4
    # Search for consecutive zeros
    for idx in range(len(zone), -1 + search_window_size, -1):
        index_to_start = idx-search_window_size
        abs_zone = np.abs(zone[index_to_start:idx])
        if np.all(abs_zone < zero_threshold):
            index_midpoint = index_to_start + int(search_window_size // 2)
            return (start_index + index_midpoint), None
    
    # Fall back method
    return find_zero_crossing(chunk, start_index, search_length)

def find_zero_crossing(chunk, start_index, search_length):
    zone = chunk[start_index:start_index + search_length]
    sign_changes = np.where(np.diff(np.sign(zone)) != 0)[0]
    
    if len(sign_changes) == 0:
        print("No zero-crossings found in this zone. This should not be happening!")
    else:
        zc_index = start_index + sign_changes[0] + 1
        prev_value = chunk[zc_index - 1]
        curr_value = chunk[zc_index]
        crossing_direction = np.sign(curr_value) - np.sign(prev_value)
        return zc_index, crossing_direction

def find_matching_index(chunk, center_index, max_offset, crossing_direction):
    if crossing_direction == None:
        return center_index
    
    # Fall back for zero_crossing
    data_length = len(chunk)
    for offset in range(max_offset + 1):
        idx_forward = center_index + offset
        idx_backward = center_index - offset

        # Forward direction
        if idx_forward < data_length - 1:
            prev_sign = np.sign(chunk[idx_forward])
            curr_sign = np.sign(chunk[idx_forward + 1])
            direction = curr_sign - prev_sign
            if direction == crossing_direction:
                return idx_forward + 1

        # Backward direction
        if idx_backward > 0:
            prev_sign = np.sign(chunk[idx_backward - 1])
            curr_sign = np.sign(chunk[idx_backward])
            direction = curr_sign - prev_sign
            if direction == crossing_direction:
                return idx_backward
    return None