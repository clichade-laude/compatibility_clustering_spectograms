from prometheus_api_client import PrometheusConnect
import time, sqlite3, json, pandas as pd

from utils.mqtt import connect_node

metrics_path = "database/metrics/"
command = "scaph_process_power_consumption_microwatts"
processes = {"python3poison.py": "poison", 
             "python3-mclustering.cluster": "cluster", 
             "python3train.py": "train",
             "python3mqtt_train.py": "controller"}

active = 0

def monitor():
    prometh = PrometheusConnect()
    conn = sqlite3.connect(metrics_path + "metrics.db")
    cursor = conn.cursor()

    cursor.execute("CREATE TABLE IF NOT EXISTS metrics (timestamp INTEGER, value TEXT)")
    df = pd.DataFrame(columns=['timestamp', 'total'] + list(processes.values()))

    while active > 0: #this should be chamged por a mqtt messahe to stop the process
        time.sleep(5)
        total_power, process_power = 0, {}
        for data in prometh.custom_query(command + "{}"):
            metric = data['metric']
            value = float(data['value'][1])/1e6
            if metric.get('__name__') == command:
                total_power += value
            elif metric.get('cmdline') in processes.keys():
                timestamp = int(data['value'][0])
                cmdline = processes[metric['cmdline']]
                if cmdline not in process_power:
                    process_power[cmdline] = 0
                process_power[cmdline] += value
        query_metrics = {"total": total_power, **process_power}
        cursor.execute("INSERT INTO metrics (timestamp, value) VALUES (?, ?)", timestamp, json.dumps(query_metrics))
        conn.commit()
        df = df.append(query_metrics, ignore_index=True)
    df.to_csv(metrics_path + 'metrics.csv', index=False)
    conn.close()

def on_message(client, userdata, msg):
    if msg.topic == "start_monitor":
        print("Node: monitoring | Executing", flush=True)
        active += 1
        if active == 1:
            monitor()
    elif msg.topic == "stop_monitor":
        active -= 1
        print("Node: monitoring | Finishing", flush=True)


if __name__ == "__main__":
    connect_node(["start_monitor", "stop_monitor"], on_message)
