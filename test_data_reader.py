from data.data_reader import DataReader
from data.obs_md import ObservationMetadata
import os


if __name__ == "__main__":
    reader = DataReader()
    sessions = reader.list_sessions()
    print("Available sessions:", sessions)

    if sessions:
        data = reader.get_data(sessions[0])
        for antenna_num, data_array in data.items():
            print(f"Antenna {antenna_num}: Data shape: {data_array.shape}")
            
        obs = reader.get_metadata(sessions[0])

    # Basic info
    print(obs.session_folder)           # "session_20251028_225045"
    print(obs.device_count)             # 9
    print(obs.status)                   # "completed"

    # Configuration
    print(f"Frequency: {obs.center_frequency:,.0f} Hz")  # "Frequency: 1,420,000,000 Hz"
    print(f"Sample rate: {obs.sampling_rate:,.0f} Hz")   # "Sample rate: 10,000,000 Hz"

    # Device management
    devices = obs.list_devices()                    # ['1', '2', '3', ...]
    process_1 = obs.get_process('1')               # Full process dict for device '1'
    completed = obs.get_processes_by_status('observation_complete')

    # Timing
    start_time = obs.get_process_start_time('1')   # datetime object
    duration = obs.get_process_duration('1')       # 5.261285066604614
    all_durations = obs.get_all_process_durations() # {'1': 5.26, '2': 5.31, ...}

    # Analysis
    print(obs.are_files_synchronized())            # True
    timing_stats = obs.get_timing_statistics()     # Mean, median, std dev, etc.
    completion = obs.get_completion_summary()      # Completion rates and status breakdown

    # Summary
    print(obs.summary())                           # Human-readable summary
    print(obs)                                     # ObservationMetadata(session='session_20251028_225045', devices=9, status='completed')
