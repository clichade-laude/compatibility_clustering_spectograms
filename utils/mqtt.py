import paho.mqtt.client as mqtt
import json, os

from threading import Thread

def publish_mqtt(client: mqtt.Client, topic, **kwargs):
    client_id = client._client_id.decode("utf-8")
    print(f"Node: {client_id} | Publishing on /{topic}", flush=True)
    client.publish(f"/{topic}", json.dumps(kwargs))

def on_connect(client: mqtt.Client, userdata, flags, rc, properties):
    if rc == 0:
        print("Connected successfully", flush=True)
        topic_lst = [userdata['topic']] if isinstance(userdata['topic'], str) else userdata['topic']
        for topic in topic_lst:
            client.subscribe(f"/{topic}")
    else:
        print("Connection failed with code", rc)

def on_message(client, userdata, msg):
    process_message = userdata['process_message']
    Thread(target=process_message, args=(client, userdata["topic"], msg)).start()

def connect_mqtt(node, topic, process_message = None):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=node, userdata={'topic': topic, 'process_message': process_message})
    client.on_connect = on_connect
    client.on_message = on_message
    
    broker = os.getenv('MQTT_BROKER', 'localhost')  # 'localhost' es el valor por defecto si la variable no está definida
    client.connect(broker)
    return client

def connect_node(node:str, topic:str, process_message):
    client = connect_mqtt(node, topic, process_message)
    print() ## Necessary
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("Disconnecting")
        client.disconnect()