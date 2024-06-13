# import serial
# import csv

# # Serial port configuration
# serial_port = 'COM3'  # Change this to your serial port
# baud_rate = 115200  # Change this to match your device's baud rate

# # Open serial port
# ser = serial.Serial(serial_port, baud_rate)

# # Open CSV file in write mode
# csv_file = open('ser_data.csv', 'w', newline='')
# csv_writer = csv.writer(csv_file)

# try:
#     while True:
#         # Read data from serial port
#         serial_data = ser.readline().decode().strip()
        
#         # Split data if needed
#         data_list = serial_data.split(',')  # Change delimiter as per your data
        
#         # Write data to CSV file
#         csv_writer.writerow(data_list)
        
#         # Print data to console
#         print(data_list)
# except KeyboardInterrupt:
#     # Close serial port and CSV file upon Ctrl+C
#     ser.close()
#     csv_file.close()



import serial
import csv
import pandas as pd

# Serial port configuration
serial_port = 'COM3'  # Change this to your serial port
baud_rate = 115200  # Change this to match your device's baud rate

# Open serial port
ser = serial.Serial(serial_port, baud_rate)

# Open CSV file in write mode
csv_file = open('serial_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# Write header row with attribute names
csv_writer.writerow(['Voltage (V)', 'Current (A)', 'Power (W)', 'Energy (kWh)'])

try:
    while True:
        # Read data from serial port
        serial_data = ser.readline().decode().strip()
        
        # Split data if needed
        data_list = serial_data.split(',')  # Change delimiter as per your data
        
        # Write data to CSV file
        csv_writer.writerow(data_list)
        
        # Print data to console
        print(data_list)
except KeyboardInterrupt:
    # Close serial port and CSV file upon Ctrl+C
    ser.close()
    csv_file.close()

# Read data from the CSV file into a pandas DataFrame
df = pd.read_csv("serial_data.csv")

# Remove leading zeros and decimal point from the 'Energy (kWh)' column and convert to integer
# Remove leading zeros and decimal point from the 'Energy (kWh)' column and convert to integer
df['Energy (kWh)'] = df['Energy (kWh)'].apply(lambda x: int(str(x).replace('.', '').lstrip('0')) if str(x).replace('.', '').lstrip('0').isdigit() else x)
# Save the modified DataFrame back to a CSV file
df.to_csv('modified_serial_data.csv', index=False)
