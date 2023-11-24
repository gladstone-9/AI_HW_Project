# import requests
# import os

# # Get more data

# base_url = "https://datasets-server.huggingface.co/assets/speech_commands/--/v0.01/train/"

# output_dir = "audio"

# word = "yes"

# start = 1
# end = 3

# for file_num in range(start,end):
#     file_num_str = str(file_num).lstrip('0')

#     file_url = f"{base_url}{file_num}/audio/audio.wav"

#     output_file_name = f"{word}.{file_num_str}.wav"
#     output_path = os.path.join(output_dir, output_file_name)

#     response = requests.get(file_url)

#     if response.status_code == 200:
#         with open(output_path, 'wb') as file:
#             file.write(response.content)
#         print(f"Downloaded: {output_file_name}")
#     else:
#         print(f"Failed to download: {output_file_name} (URL: {file_url})")

# print("Download completed")

# import requests

# url = "https://datasets-server.huggingface.co/splits"
# params = {"dataset": "speech_commands"}

# response = requests.get(url, params=params)

# if response.status_code == 200:
#     # The request was successful
#     data = response.json()
#     print(data)
# else:
#     # There was an error with the request
#     print(f"Error: {response.status_code}")
#     print(response.text)