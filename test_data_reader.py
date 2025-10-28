from data.data_reader import DataReader


if __name__ == "__main__":
    reader = DataReader()
    sessions = reader.list_sessions()
    print("Available sessions:", sessions)

    if sessions:
        data = reader.get_data(sessions[0])
        for antenna_num, data_array in data.items():
            print(f"Antenna {antenna_num}: Data shape: {data_array.shape}")
