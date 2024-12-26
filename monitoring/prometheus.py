from prometheus_api_client import PrometheusConnect
import time, sqlite3, json, pandas as pd
from os.path import join

from utils.mqtt import connect_node, log_time

command = "scaph_process_power_consumption_microwatts"
processes = {"python3poison.py": "poison", 
             "python3-mclustering.cluster": "cluster", 
             "python3train.py": "train",
             "python3mqtt_train.py": "controller"}

active = 0

def monitor(test_folder):
    test_path = join("database", "results", test_folder)
    prometh = PrometheusConnect("http://localhost:9090")
    conn = sqlite3.connect(join(test_path, "metrics.db"))
    cursor = conn.cursor()

    cursor.execute("CREATE TABLE IF NOT EXISTS metrics (timestamp INTEGER, value TEXT)")
    df = pd.DataFrame(columns=['timestamp', 'total'] + list(processes.values()))

    while active > 0: #this should be chamged por a mqtt messahe to stop the process
        time.sleep(5)
        total_power, process_power = 0, {}
        for data in prometh.custom_query(command + "{}"):
            metric = data['metric']
            if metric.get('__name__') != command:
                continue
            ## Obtain timestamp and metric value
            timestamp = int(data['value'][0])
            value = float(data['value'][1])/1e6
            ## Add value to total power
            total_power += value
            ## Search for container processes
            if metric.get('cmdline') in processes.keys():
                cmdline = processes[metric['cmdline']]
                if cmdline not in process_power:
                    process_power[cmdline] = 0
                process_power[cmdline] += value
        query_metrics = {"total": total_power, **process_power}
        cursor.execute("INSERT INTO metrics (timestamp, value) VALUES (?, ?)", (timestamp, json.dumps(query_metrics)))
        conn.commit()
        df = pd.concat([df, pd.DataFrame([query_metrics])], ignore_index=True)
    df.to_csv(join(test_path, 'metrics.csv'), index=False, mode="a")
    conn.close()

def on_message(client, userdata, msg):
    global active
    if msg.topic == "/start_monitor":
        print(f"{log_time()} Node: monitoring | Executing", flush=True)
        params = json.loads(msg.payload)
        active += 1
        if active == 1:
            monitor(params["folder"])
    elif msg.topic == "/stop_monitor":
        active -= 1
        print(f"{log_time()} Node: monitoring | Finishing", flush=True)


if __name__ == "__main__":
    connect_node("monitor", ["start_monitor", "stop_monitor"], on_message)