import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import scipy

# Constants
WORKING_START = 8  # 8:00
WORKING_END = 17   # 17:00
SIMULATION_DAYS = 7 # One week

def generate_patient_arrivals():
    """Generate one week of patient arrivals during working hours"""
    # Create base timestamp for start of simulation
    start_date = datetime.now().replace(hour=WORKING_START, minute=0, second=0, microsecond=0)
    
    # Generate 7 days of patients
    data = {
        'patient_type': [],
        'arrival_time': [],
        'scan_duration': [],
        'scheduled_time': [],
        'waiting_time': []
    }
    
    # For each day
    for day in range(SIMULATION_DAYS):
        current_day = start_date + timedelta(days=day)
        
        # Generate type 1 patients for this day (mean arrival rate: 1 per 29.04 minutes)
        n_type1 = np.random.poisson(lam=(WORKING_END-WORKING_START)*60/29.04)  # Number of type 1 patients per day
        arrival_minutes_type1 = np.random.uniform(0, (WORKING_END-WORKING_START)*60, n_type1)  # Arival at random points in workday
        
        # Put patients of type 1 in dataframe
        for minutes in arrival_minutes_type1:
            arrival_time = current_day + timedelta(minutes=minutes)
            data['patient_type'].append(1)
            data['arrival_time'].append(arrival_time)
            data['scan_duration'].append(np.random.normal(25.96, 5.866455))
            data['scheduled_time'].append(None)
            data['waiting_time'].append(None)
        
        # Calculate Weibull shape parameter (k) using mean (51) and std (18)
        cv = 18/51  # coefficient of variation = std/mean
        k = 3.0  # shape parameter for Weibull (cv = 0.353 leads to k ≈ 3.0)

        # Generate number of type 2 patients using Weibull distribution
        n_type2 = int(np.random.weibull(k) * (((WORKING_END-WORKING_START)*60)/51/scipy.special.gamma(1 + 1/k)))  # Number of type 2 patients per day
        # Generate their arrival times uniformly throughout the day
        arrival_minutes_type2 = np.random.uniform(0, (WORKING_END-WORKING_START)*60, n_type2)  # Arrival at random points in workday
        
        # Put patients of type 2 in dataframe
        for minutes in arrival_minutes_type2:
            arrival_time = current_day + timedelta(minutes=minutes)
            data['patient_type'].append(2)
            data['arrival_time'].append(arrival_time)
            data['scan_duration'].append(np.random.beta(6.077, 14.41) * 135.3) # Beta distribution with alpha = 6.07, beta 14.41, scale = mean/(alpha/(alpha+beta)) = 40.14/0.2966 ≈ 135.3
            data['scheduled_time'].append(None)
            data['waiting_time'].append(None)
    
    # Create DataFrame and sort by arrival time
    df = pd.DataFrame(data)
    return df.sort_values('arrival_time').reset_index(drop=True)

def schedule_dedicated_machines(patients_df):
   """Schedule patients with dedicated machines"""
   # Filter out type 1 patients and create copy for machine 1
   machine1_df = patients_df[patients_df['patient_type'] == 1].copy()
   # Filter out type 2 patients and create copy for machine 2  
   machine2_df = patients_df[patients_df['patient_type'] == 2].copy()
   
   # Fixed time slot duration for type 1 patients
   slot_duration_type1 = 30
   # Fixed time slot duration for type 2 patients  
   slot_duration_type2 = 45
   
   def schedule_machine(df, slot_duration):
       # Loop through each patient in the dataframe
       for idx in df.index:
           # Get next day after patient arrival as scheduling date
           current_date = df.loc[idx, 'arrival_time'].date() + timedelta(days=1)
           
           while True:
               # Set time to start of working day
               current_time = datetime.combine(current_date, datetime.min.time().replace(hour=WORKING_START))
               # Convert to pandas timestamp for easier datetime operations
               current_time = pd.Timestamp(current_time)
               
               # Keep checking slots until end of working day
               while current_time.hour < WORKING_END:
                   # Calculate when current time slot would end
                   slot_end = current_time + pd.Timedelta(minutes=slot_duration)
                   
                   # Get mask of already scheduled appointments
                   scheduled_mask = df['scheduled_time'].notna()
                   if scheduled_mask.any():
                       # Check if current slot overlaps with any scheduled appointments
                       overlap = df[scheduled_mask][
                           (df[scheduled_mask]['scheduled_time'] < slot_end) & 
                           ((df[scheduled_mask]['scheduled_time'] + pd.Timedelta(minutes=slot_duration)) > current_time)
                       ]
                       # If no overlap found, schedule patient in this slot
                       if len(overlap) == 0:
                           df.loc[idx, 'scheduled_time'] = current_time
                           # Calculate waiting time in hours
                           df.loc[idx, 'waiting_time'] = (current_time - df.loc[idx, 'arrival_time']).total_seconds() / 3600
                           break
                   else:
                       # If no appointments scheduled yet, use first slot
                       df.loc[idx, 'scheduled_time'] = current_time
                       # Calculate waiting time in hours
                       df.loc[idx, 'waiting_time'] = (current_time - df.loc[idx, 'arrival_time']).total_seconds() / 3600
                       break
                       
                   # Move to next possible slot
                   current_time += pd.Timedelta(minutes=slot_duration)
               
               # If patient was scheduled, break out of date loop
               if pd.notna(df.loc[idx, 'scheduled_time']):
                   break
                   
               # If no slot found today, try next day
               current_date += timedelta(days=1)
       
       return df
   
   # Schedule type 1 patients on machine 1
   machine1_df = schedule_machine(machine1_df, slot_duration_type1)
   # Schedule type 2 patients on machine 2
   machine2_df = schedule_machine(machine2_df, slot_duration_type2)
   
   # Return both scheduled dataframes
   return machine1_df, machine2_df

def schedule_shared_machines(patients_df):
   """Schedule patients with shared machines (any patient can use either machine)"""
   # Create empty dataframes for each machine
   machine1_df = pd.DataFrame(columns=patients_df.columns)
   machine2_df = pd.DataFrame(columns=patients_df.columns)
   
   # Sort patients by arrival time and make a copy
   patients_df = patients_df.sort_values('arrival_time').copy()
   # Set fixed slot durations for each patient type
   slot_duration_type1 = 30
   slot_duration_type2 = 45
   
   # Loop through each patient
   for idx in patients_df.index:
       # Get current patient info
       patient = patients_df.loc[idx]
       # Set slot duration based on patient type
       slot_duration = slot_duration_type1 if patient['patient_type'] == 1 else slot_duration_type2
       # Start scheduling from next day
       current_date = patient['arrival_time'].date() + timedelta(days=1)
       
       # Keep trying until we find a slot
       while True:
           # Start at beginning of working day
           current_time = pd.Timestamp(datetime.combine(current_date, datetime.min.time().replace(hour=WORKING_START)))
           
           # Try each time slot during the day
           while current_time.hour < WORKING_END:
               # Calculate when this slot would end
               slot_end = current_time + pd.Timedelta(minutes=slot_duration)
               
               # Try scheduling on machine 1
               # If machine 1 is empty OR there's no overlap with existing appointments
               if len(machine1_df) == 0 or not any(
                   (machine1_df['scheduled_time'] < slot_end) & 
                   ((machine1_df['scheduled_time'] + pd.Timedelta(minutes=slot_duration)) > current_time)
               ):
                   # Schedule the patient
                   patients_df.loc[idx, 'scheduled_time'] = current_time
                   # Calculate how long they waited
                   patients_df.loc[idx, 'waiting_time'] = (current_time - patient['arrival_time']).total_seconds() / 3600
                   # Add patient to machine 1's schedule
                   machine1_df = pd.concat([machine1_df, patients_df.loc[[idx]]], ignore_index=True)
                   break
               
               # If machine 1 didn't work, try machine 2
               # Same logic as machine 1
               if len(machine2_df) == 0 or not any(
                   (machine2_df['scheduled_time'] < slot_end) & 
                   ((machine2_df['scheduled_time'] + pd.Timedelta(minutes=slot_duration)) > current_time)
               ):
                   patients_df.loc[idx, 'scheduled_time'] = current_time
                   patients_df.loc[idx, 'waiting_time'] = (current_time - patient['arrival_time']).total_seconds() / 3600
                   machine2_df = pd.concat([machine2_df, patients_df.loc[[idx]]], ignore_index=True)
                   break
               
               # If neither machine worked, try next time slot
               current_time += pd.Timedelta(minutes=slot_duration)
           
           # If we found a slot, move to next patient
           if pd.notna(patients_df.loc[idx, 'scheduled_time']):
               break
               
           # If no slots today, try tomorrow
           current_date += timedelta(days=1)
   
   # Return schedules for both machines
   return machine1_df, machine2_df

def calculate_kpis(machine1_df, machine2_df):
    """Calculate Key Performance Indicators"""
    # Combine all patients and calculate average waiting time
    all_patients = pd.concat([machine1_df, machine2_df], ignore_index=True)
    avg_waiting_time = all_patients['waiting_time'].mean()
    
    total_days = 0
    total_idle_time = 0
    total_overtime = 0
    
    for df in [machine1_df, machine2_df]:
        if len(df) == 0:
            continue
            
        # Ensure scheduled_time is datetime and sort
        df = df.copy()
        df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
        df = df.sort_values('scheduled_time')
        
        # Get unique dates
        unique_dates = df['scheduled_time'].dt.date.unique()
        total_days += len(unique_dates)
        
        # Analyze each day
        for date in unique_dates:
            day_appointments = df[df['scheduled_time'].dt.date == date].copy()
            
            # Initialize the actual start time of each appointment
            actual_times = []
            current_time = None
            
            # Calculate actual start and end times considering delays
            for idx, row in day_appointments.iterrows():
                scheduled_time = row['scheduled_time']
                actual_duration = row['scan_duration']
                
                if current_time is None:
                    # First appointment starts at scheduled time
                    actual_start = scheduled_time
                else:
                    # Next appointment starts after previous one finishes
                    actual_start = max(scheduled_time, current_time)
                
                actual_end = actual_start + pd.Timedelta(minutes=actual_duration)
                current_time = actual_end
                actual_times.append((actual_start, actual_end))
            
            # Calculate overtime if last appointment ends after 17:00
            if actual_times:
                last_end_time = actual_times[-1][1]
                day_end = pd.Timestamp(date).replace(hour=WORKING_END, minute=0)
                
                if last_end_time > day_end:
                    overtime_minutes = (last_end_time - day_end).total_seconds() / 60
                    total_overtime += overtime_minutes
            
            # Calculate idle time between appointments
            for i in range(len(actual_times) - 1):
                current_end = actual_times[i][1]
                next_start = actual_times[i + 1][0]
                
                if next_start > current_end:
                    idle_minutes = (next_start - current_end).total_seconds() / 60
                    total_idle_time += idle_minutes
    
    # Convert to daily averages (in hours)
    avg_daily_idle_time = (total_idle_time / total_days) / 60 if total_days > 0 else 0
    avg_daily_overtime = (total_overtime / total_days) / 60 if total_days > 0 else 0
    
    return {
        'average_waiting_time': avg_waiting_time,
        'idle_time': avg_daily_idle_time,
        'overtime': avg_daily_overtime
    }

def run_simulation():
    """ Run the simulation """
    # Generate patient arrivals
    print("Generating patient arrivals...")
    patients_df = generate_patient_arrivals()
    print(f"Generated {len(patients_df)} patients")
    
    # Run both scheduling strategies
    print("\nRunning dedicated machines strategy...")
    dedicated_m1, dedicated_m2 = schedule_dedicated_machines(patients_df.copy())

    print("Running shared machines strategy...")
    shared_m1, shared_m2 = schedule_shared_machines(patients_df.copy())

    # Calculate KPIs for both strategies
    dedicated_kpis = calculate_kpis(dedicated_m1, dedicated_m2)
    shared_kpis = calculate_kpis(shared_m1, shared_m2)
    
    print("\nResults for Dedicated Machines:")
    print(f"Average Waiting Time: {dedicated_kpis['average_waiting_time']:.2f} hours")
    print(f"Average daily Idle Time: {dedicated_kpis['idle_time']:.2f} hours")
    print(f"Average daily Overtime: {dedicated_kpis['overtime']:.2f} hours")
    
    print("\nResults for Shared Machines:")
    print(f"Average Waiting Time: {shared_kpis['average_waiting_time']:.2f} hours")
    print(f"Average daily Idle Time: {shared_kpis['idle_time']:.2f} hours")
    print(f"Average daily Overtime: {shared_kpis['overtime']:.2f} hours")
    
    return {
        'dedicated': {'machine1': dedicated_m1, 'machine2': dedicated_m2},
        'shared': {'machine1': shared_m1, 'machine2': shared_m2}
    }

def run_multiple_simulations(n_runs):
    """ Run the simulation multiple times and average the results """
    # Initialize accumulators for KPIs
    total_dedicated = {
        'average_waiting_time': 0,
        'idle_time': 0,
        'overtime': 0
    }
    total_shared = {
        'average_waiting_time': 0,
        'idle_time': 0,
        'overtime': 0
    }
    
    print(f"Starting {n_runs} simulation runs...")
    
    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")
        
        # Generate new patients for this run
        patients_df = generate_patient_arrivals()
        
        # Run both scheduling strategies
        dedicated_m1, dedicated_m2 = schedule_dedicated_machines(patients_df.copy())
        shared_m1, shared_m2 = schedule_shared_machines(patients_df.copy())
        
        # Calculate KPIs for this run
        dedicated_kpis = calculate_kpis(dedicated_m1, dedicated_m2)
        shared_kpis = calculate_kpis(shared_m1, shared_m2)
        
        # Accumulate results
        for metric in total_dedicated.keys():
            total_dedicated[metric] += dedicated_kpis[metric]
            total_shared[metric] += shared_kpis[metric]
    
    # Calculate averages
    avg_dedicated = {
        metric: value / n_runs 
        for metric, value in total_dedicated.items()
    }
    avg_shared = {
        metric: value / n_runs 
        for metric, value in total_shared.items()
    }
    
    # Print averaged results
    print("\nAveraged Results over", n_runs, "runs:")
    print("\nDedicated Machines Strategy:")
    print(f"Average Waiting Time: {avg_dedicated['average_waiting_time']:.2f} hours")
    print(f"Average daily Idle Time: {avg_dedicated['idle_time']:.2f} hours")
    print(f"Average daily Overtime: {avg_dedicated['overtime']:.2f} hours")
    
    print("\nShared Machines Strategy:")
    print(f"Average Waiting Time: {avg_shared['average_waiting_time']:.2f} hours")
    print(f"Average daily Idle Time: {avg_shared['idle_time']:.2f} hours")
    print(f"Average daily Overtime: {avg_shared['overtime']:.2f} hours")
    
    return {
        'dedicated': avg_dedicated,
        'shared': avg_shared
    }

if __name__ == "__main__":
    #results = run_multiple_simulations(50)
    results = run_simulation()