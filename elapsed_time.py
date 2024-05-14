from datetime import datetime

# example entry: 05-11 21:54:30 INFO E: 0 B: 231

def calculate_elapsed_time(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        print(lines[0][:15])

        start_time = datetime.strptime(lines[0].split(' ')[0] + ' ' + lines[0].split(' ')[1], '%m-%d %H:%M:%S')
        end_time = datetime.strptime(lines[-1].split(' ')[0] + ' ' + lines[-1].split(' ')[1], '%m-%d %H:%M:%S')


        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        
        return elapsed_time

file_path = "models/model_epochs1_noaug_id0/log_05-11_20.41.30.txt"  # Replace with the path to your file
elapsed_time = calculate_elapsed_time(file_path)
print("Total elapsed time: ", elapsed_time)