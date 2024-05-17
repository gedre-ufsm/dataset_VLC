#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import re
import os

def extract_parameters_and_combine(script_path):
    """
    Extracts essential parameters from the GNU Radio generated Python script,
    combines device address and stream channel information for both source and sink,
    and removes redundant entries.
    
    Args:
    script_path (str): The file path to the GNU Radio generated Python script.
    
    Returns:
    dict: A dictionary of extracted and combined parameters.
    """
    # Define regular expressions for extracting parameters
    regexes = {
        'samp_rate': r'self\.samp_rate = samp_rate = ([\d\.e\+\-]+)',
        'samples': r'self\.samples = samples = ([\d\.e\+\-]+)',
        'gain': r'self\.gain = gain = ([\d\.e\+\-]+)',
        'frequency': r'self\.frequency = frequency = ([\d\.e\+\-]+)',
        'device_address_source': r'uhd\.usrp_source\(\s*","\.join\(\("(addr0=[^"]+|serial=[^"]+)"',
        'stream_channel_source': r'uhd\.usrp_source\(.+?channels=\[([0-9]+)\]',
        'device_address_sink': r'uhd\.usrp_sink\(\s*","\.join\(\("(addr0=[^"]+|serial=[^"]+)"',
        'stream_channel_sink': r'uhd\.usrp_sink\(.+?channels=\[([0-9]+)\]',
        'file_sink_source': r'self\.blocks_file_sink_0_1_0 = blocks\.file_sink\(.*?, \'(.*?)\', False\)',
        'file_sink_received': r'self\.blocks_file_sink_0_0 = blocks\.file_sink\(.*?, \'(.*?)\', False\)',
        'file_sink_sent': r'self\.blocks_file_sink_0 = blocks\.file_sink\(.*?, \'(.*?)\', False\)',
    }
    parameters = {}
    
    with open(script_path, 'r') as file:
        content = file.read()
        
        # Extract parameters using defined regular expressions
        for key, regex in regexes.items():
            match = re.search(regex, content, re.MULTILINE | re.DOTALL)
            if match:
                parameters[key] = match.group(1)
                
    # Combine device address and channel for source and sink for clearer representation
    if 'device_address_source' in parameters and 'stream_channel_source' in parameters:
        parameters['source_adress'] = f" {parameters['device_address_source']} (Channel: {parameters['stream_channel_source']})"
    
    if 'device_address_sink' in parameters and 'stream_channel_sink' in parameters:
        parameters['sink_adress'] = f" {parameters['device_address_sink']} (Channel: {parameters['stream_channel_sink']})"
    
    # Remove original entries to avoid duplication
    for key in ['device_address_source', 'stream_channel_source', 'device_address_sink', 'stream_channel_sink']:
        parameters.pop(key, None)
    
    return parameters


def print_and_save_with_combined_params(script_path, message, file_path='output.txt'):
    """
    Extracts parameters from the GNU Radio generated script, combines relevant information,
    and saves it to a specified log file with an initial message.
    
    Args:
    script_path (str): Path to the GNU Radio generated Python script.
    message (str): Initial message to write to the log file.
    file_path (str): Path to the log file where information will be saved.
    """
    parameters = extract_parameters_and_combine(script_path)

    with open(file_path, 'a') as f:
        # Write the initial analysis message
        f.write(message + '')

        # Add extracted parameters to the log
        f.write("*\n--- Simulation Parameters ---\n")
        for key, value in sorted(parameters.items()):
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("--- End of Parameters ---\n\n Signal analysis: ")

# Example of usage
script_path = "channel_dataset.py"
initial_message = ""
print_and_save_with_combined_params(script_path, initial_message)

# This function serves dual purposes: it prints messages to the console and saves them to a specified file.
def print_and_save(message, file_path='output.txt'):
    print(message)  # Display the message on the console.
    with open(file_path, 'a') as f:  # Open the file in append mode or create it if it doesn't exist.
        f.write(message + '\n')  # Write the message to the file and move to a new line.

# Specify the file paths for the transmitted (sent) and received I/Q data.
enviado_path = 'sent'  # Assumes these are raw binary files without an extension.
recebido_path = 'received'

# Read I/Q data from binary files as 32-bit floats.
with open(enviado_path, 'rb') as f:
    x_float32 = np.fromfile(f, dtype=np.float32)  # Transmitted signal.

with open(recebido_path, 'rb') as f:
    y_float32 = np.fromfile(f, dtype=np.float32)  # Received signal.

print(f"length of y: {len(y_float32)}")
print(f"length of x: {len(x_float32)}")

# Ensure 'y_float32' is trimmed to match the length of 'x_float32'.

if len(y_float32) > len(x_float32):
	jump=round(0.3*len(x_float32))
	#jump=0
	y_float32 = y_float32[jump:len(x_float32)]
	x_float32 = x_float32[jump:len(x_float32)]

else:
	jump = round(0.3*len(y_float32))
	#jump=0
	x_float32 = x_float32[jump:len(y_float32)]
	y_float32 = y_float32[jump:len(y_float32)]


# Ensure 'y_float32' is trimmed to match the length of 'x_float32'.
#jump=round(0.5*len(x_float32))
#y_float32_trimmed = y_float32[jump:len(x_float32)]
#x_float32=x_float32[:len(y_float32)]
#x_float32=x_float32[jump:len(x_float32)]

# Ensure both x and y lengths are even for proper complex number formation
#if len(x_float32) % 2 != 0:
#   x_float32 = x_float32[:-1]
#if len(y_float32) % 2 != 0:
#    y_float32 = y_float32[:-1]


print(f"length of y: {len(y_float32)}")
print(f"length of x: {len(x_float32)}")

# Convert pairs of floats to complex samples for both transmitted and received signals.
IQ_x_complex = x_float32[::2] + 1j * x_float32[1::2]  # Transmitted I/Q samples.
IQ_y_complex = y_float32[::2] + 1j * y_float32[1::2]  # Received I/Q samples

def synchronize_and_normalize(transmitted_signal, received_signal, print_delay=True):
    """
    Synchronizes and normalizes the received signal with respect to the transmitted signal.
    It corrects the time delay and adjusts for phase and amplitude differences.
    
    Parameters:
    - transmitted_signal: The transmitted I/Q samples.
    - received_signal: The received I/Q samples.
    - print_delay: Flag to print detected and corrected delays.
    
    Returns:
    - The normalized received signal.
    """
    # Perform FFT on both signals for frequency domain analysis.
    fft_transmitted = np.fft.fft(transmitted_signal)
    fft_received = np.fft.fft(received_signal)

    # Calculate cross-correlation using IFFT to find the delay.
    cross_correlation = np.fft.ifft(fft_received * np.conj(fft_transmitted))
    delay = np.argmax(np.abs(cross_correlation)) # Find the index of maximum correlation which represents the delay.
	
    # Print the detected delay before any correction.
    if print_delay:
        print_and_save(f"Detected delay before correction: {delay}")

    # Correct the delay if the detected delay exceeds half the length of the signal, indicating a wrap-around.
    if delay > len(transmitted_signal) // 2:
        corrected_delay = delay - len(transmitted_signal)
    else:
        corrected_delay = delay
        
    # Align the received signal by rolling it back by the detected delay.
    received_signal_aligned = np.roll(received_signal, -corrected_delay)

    # Recalculate cross-correlation after alignment to verify correction. Ideally, the new delay should be zero.
    fft_received_aligned = np.fft.fft(received_signal_aligned)
    new_cross_correlation = np.fft.ifft(fft_received_aligned * np.conj(fft_transmitted))
    new_delay = np.argmax(np.abs(new_cross_correlation))
    
    # Ensures printing the recalculated delay accurately reflects the adjustment.
    if print_delay:
        print_and_save(f"Delay after correction: {new_delay}")

    # Adjust for phase differences between the transmitted and aligned received signals.
    phase_difference = np.angle(np.fft.fft(received_signal_aligned) / fft_transmitted)
    average_phase_difference = np.angle(np.mean(np.exp(1j * phase_difference)))
   
    # Apply the phase correction to the aligned received signal.
    received_signal_aligned_corrected = received_signal_aligned * np.exp(-1j * average_phase_difference)
    
    # Normalize the amplitude of the corrected received signal to match the transmitted signal.
    normalization_factor = np.sqrt(np.mean(np.abs(transmitted_signal)**2) / np.mean(np.abs(received_signal_aligned_corrected)**2))
    #normalization_factor=1
    IQ_y_complex_normalized = received_signal_aligned_corrected * normalization_factor


    return IQ_y_complex_normalized
    
# Process signals and print results.
IQ_y_complex_normalized = synchronize_and_normalize(IQ_x_complex, IQ_y_complex)   

print("length IQ_y_complex:",len(IQ_y_complex))
print("length IQ_x_complex:",len(IQ_x_complex))

   
# Function to estimate noise power.
def estimate_noise_power(transmitted_signal, received_signal):
    """
    Estimates the noise power based on the difference between the transmitted and received signals.
    
    Parameters:
    - transmitted_signal: The transmitted I/Q samples.
    - received_signal: The normalized received I/Q samples.
    
    Returns:
    - The estimated noise power.
    """
    noise_signal = received_signal - transmitted_signal  # Noise signal.
    noise_power = np.mean(np.abs(noise_signal) ** 2)  # Noise power calculation.
    return noise_power

# Function to calculate SNR.
def calculate_snr(transmitted_signal, received_signal):
    """
    Calculates the Signal-to-Noise Ratio (SNR) in dB.
    
    Parameters:
    - transmitted_signal: The transmitted I/Q samples.
    - received_signal: The normalized received I/Q samples.
    
    Returns:
    - SNR in dB.
    """
    P_signal = np.mean(np.abs(transmitted_signal)**2)  # Power of the transmitted signal.
    noise_signal = received_signal - transmitted_signal
    P_noise = np.mean(np.abs(noise_signal)**2)  # Power of the noise.
    SNR_dB = 10 * np.log10(P_signal / P_noise)  # SNR calculation.
    return SNR_dB
    
def calculate_evm(transmitted_signal, received_signal):
    # Calcular o vetor de erro
    error_vector = received_signal - transmitted_signal
    
    # Calcular a magnitude do vetor de erro
    error_magnitude = np.abs(error_vector)
    
    # Calcular a potência média dos símbolos transmitidos
    average_power = np.mean(np.abs(transmitted_signal)**2)
    
    # Calcular EVM
    EVM = np.sqrt(np.mean(error_magnitude**2) / average_power)
    
    # Converter EVM para porcentagem
    EVM_percentage = EVM * 100
    
    # Convertendo EVM para dB
    EVM_dB = 20 * np.log10(EVM)

   
    return EVM_percentage, EVM_dB

# Calculate SVM
evm_percentage, evm_dB = calculate_evm(IQ_x_complex, IQ_y_complex_normalized)
print_and_save(f"EVM: {evm_percentage:.2f}%")
print_and_save(f"EVM (dB): {evm_dB:.2f} dB")

# Calculate noise power and SNR.
noise_power = estimate_noise_power(IQ_x_complex, IQ_y_complex_normalized)
print_and_save(f"Noise power: {noise_power}")

SNR_dB = calculate_snr(IQ_x_complex, IQ_y_complex_normalized)
print_and_save(f"SNR: {SNR_dB} dB")
 
 
# Calcular a FFT
fft_resultado = np.fft.fft(IQ_y_complex_normalized)
fft_resultado2 = np.fft.fft(IQ_x_complex)

parameters = extract_parameters_and_combine(script_path)
samp_rate = float(parameters.get('samp_rate', 1))

# Calcular frequências correspondentes
freqs = np.fft.fftfreq(len(IQ_y_complex_normalized), 1 / samp_rate) 
 

# Printing the first 10 values of IQ_x_complex
print_and_save(f"\nShape of IQ_x_complex: {IQ_x_complex.shape} ")
print_and_save(f"First 10 values of IQ_x_complex: {IQ_x_complex[:10]} ")

# Printing the first 10 values of IQ_y_complex_normalized
print_and_save(f"\nShape of IQ_y_complex (normalized): {IQ_y_complex_normalized.shape} ")
print_and_save(f"First 10 values of IQ_y_complex (normalized): {IQ_y_complex_normalized[:10]} ")
print_and_save("\nEND\n")
 
# Printing the first 10 values of IQ_y_complex
print_and_save(f"\nShape of IQ_y_complex: {IQ_y_complex.shape} ")
print_and_save(f"First 10 values of IQ_y_complex: {IQ_y_complex[10:20]} ")
print_and_save("\nEND\n")
 
##PLOTS 

# Plotting FFTs
plt.figure(figsize=(14, 6))

# Recebido
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(freqs, np.abs(fft_resultado))
plt.title("FFT, signal Received")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")

#Enviado
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(freqs, np.abs(fft_resultado2))
plt.title("FFT, signal sent")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig('FFT.png')


# Plotting the constellation diagram.
plt.figure(figsize=(18, 6))  # Set the figure size.

# Plot for the normalized received signal constellation.
plt.subplot(1, 3, 3)  # 1 row, 3 columns, position 3
plt.scatter(IQ_y_complex_normalized.real, IQ_y_complex_normalized.imag, marker='.', s=0.1)
plt.title('Normalized Received Signal Constellation')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.axis('equal')  # Ensures that the axes are of the same scale.

# Plot for the transmitted signal constellation.
plt.subplot(1, 3, 1)  # 1 row, 3 columns, position 1
plt.scatter(IQ_x_complex.real, IQ_x_complex.imag, marker='.', s=0.1)
plt.title('Transmitted Signal Constellation')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.axis('equal')

# Plot for the original received signal constellation.
plt.subplot(1, 3, 2)  # 1 row, 3 columns, position 2
plt.scatter(IQ_y_complex.real, IQ_y_complex.imag, marker='.', s=0.1)
plt.title('Original Received Signal Constellation')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.axis('equal')

plt.savefig('phase_adjustment.png')
plt.tight_layout()

# Plotting Real and Imaginary parts.
plt.figure(figsize=(14, 6))

# Real parts.
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(IQ_x_complex.real, label='Sent Real')
plt.plot(IQ_y_complex_normalized.real, label='Received Real')
plt.title('Real Parts')
plt.xlabel('Index')
plt.ylabel('Real Value')
plt.legend()

# Imaginary parts.
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(IQ_x_complex.imag, label='Sent Imaginary')
plt.plot(IQ_y_complex_normalized.imag, label='Received Imaginary')
plt.title('Imaginary Parts')
plt.xlabel('Index')
plt.ylabel('Imaginary Value')
plt.legend()
plt.tight_layout()
plt.savefig('real_and_imaginary_time.png')

# Plotting Magnitudes and Phases.
plt.figure(figsize=(14, 6))

# Magnitudes.
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(np.abs(IQ_x_complex), label='Sent Magnitude')
plt.plot(np.abs(IQ_y_complex_normalized), label='Received Magnitude')
plt.title('Magnitudes')
plt.xlabel('Index')
plt.ylabel('Magnitude')
plt.legend()

# Phases.
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(np.angle(IQ_x_complex), label='Sent Phase')
plt.plot(np.angle(IQ_y_complex_normalized), label='Received Phase')
plt.title('Phases')
plt.xlabel('Index')
plt.ylabel('Phase (radians)')
plt.legend()
plt.tight_layout()
plt.savefig('magnitudes_and_phases_time.png')

# Identifying the midpoint and selecting 100 middle samples.
mid_point = len(IQ_x_complex) // 2
start = mid_point - 50
end = mid_point + 50
IQ_x_complex_mid_samples = IQ_x_complex[start:end]
IQ_y_complex_normalized_mid_samples = IQ_y_complex_normalized[start:end]

# Plotting the middle samples of real and imaginary parts.
plt.figure(figsize=(14, 6))

# Middle real samples.
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(IQ_x_complex_mid_samples.real, label='Sent Real')
plt.plot(IQ_y_complex_normalized_mid_samples.real, label='Received Real')
plt.title('Real Parts - Middle Samples')
plt.xlabel('Index')
plt.ylabel('Real Value')
plt.legend()

# Middle imaginary samples.
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(IQ_x_complex_mid_samples.imag, label='Sent Imaginary')
plt.plot(IQ_y_complex_normalized_mid_samples.imag, label='Received Imaginary')
plt.title('Imaginary Parts - Middle Samples')
plt.xlabel('Index')
plt.ylabel('Imaginary Value')
plt.legend()
plt.tight_layout()
plt.savefig('real_and_imaginary_time_MID.png')

# Plotting Magnitudes and Phases of the middle samples
plt.figure(figsize=(14, 6))

# Plotting magnitudes of middle samples
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(np.abs(IQ_x_complex_mid_samples), label='Sent Magnitude Mid Samples')
plt.plot(np.abs(IQ_y_complex_normalized_mid_samples), label='Received Magnitude Mid Samples')
plt.title('Middle Magnitudes')
plt.xlabel('Sample Index')
plt.ylabel('Magnitude')
plt.legend()

# Plotting phases of middle samples
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(np.angle(IQ_x_complex_mid_samples), label='Sent Phase Mid Samples')
plt.plot(np.angle(IQ_y_complex_normalized_mid_samples), label='Received Phase Mid Samples')
plt.title('Middle Phases')
plt.xlabel('Sample Index')
plt.ylabel('Phase (radians)')
plt.legend()
plt.tight_layout()
plt.savefig('magnitudes_and_phases_mid_samples.png')
#
#plt.show()


# Create a new directory for the files
directory = "IQ_data"
if not os.path.exists(directory):
    os.makedirs(directory)

# Function to save data in both .npy and .mat formats
def save_data(data, filename):
    np.save(os.path.join(directory, filename + '.npy'), data)
   # savemat(os.path.join(directory, filename + '.mat'), {filename: data})

# 1. Combine I and Q into complex-valued arrays
save_data(IQ_x_complex, 'sent_data_complex')
save_data(IQ_y_complex_normalized, 'received_data_complex')

# Separate I and Q components for the normalized signals
I_x = IQ_x_complex.real
Q_x = IQ_x_complex.imag
IQ_x_tuple = np.column_stack((I_x, Q_x))

I_y = IQ_y_complex.real
Q_y = IQ_y_complex.imag
IQ_y_tuple = np.column_stack((I_y, Q_y))

# 2. Save as a tuple
save_data(IQ_x_tuple, 'sent_data_tuple')
save_data(IQ_y_tuple, 'received_data_tuple')


# The 'input' function call at the end keeps the script from closing immediately after execution, 
# allowing the user to see the final plots in interactive environments. 
# This can be particularly useful when running the script from a command line interface.
input("Press Enter to exit...")  # Keeps the script running until the user presses Enter, useful for viewing plots.

