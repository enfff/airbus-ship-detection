from datetime import datetime
import re

file_path = "models/model_noaug_id0/log.txt"  # Replace with the path to your file

lines = []
with open(file_path, 'r') as file:
    lines_buf = file.readlines()
    pattern = re.compile(r"E: \d+ B: \d+")
    lines = [line for line in lines_buf if pattern.search(line)]

def calculate_elapsed_time(lines=lines):
    start_time = datetime.strptime(lines[0].split(' ')[0] + ' ' + lines[0].split(' ')[1], '%m-%d %H:%M:%S')
    end_time = datetime.strptime(lines[-1].split(' ')[0] + ' ' + lines[-1].split(' ')[1], '%m-%d %H:%M:%S')
    return end_time - start_time

print(f"(TRAIN) Total elapsed time: {calculate_elapsed_time()} (hours)")
print(f"(TRAIN) Mean batch training time: {float(calculate_elapsed_time().total_seconds()/len(lines)).__round__(4)} (secs)")