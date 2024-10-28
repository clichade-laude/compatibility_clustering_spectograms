import paho.mqtt.client as mqtt
import json, os

def publish_mqtt(client, topic, **kwargs):
    ret = client.publish(f"/{topic}", json.dumps(kwargs))
    ret.wait_for_publish()

def on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        print("Connected successfully", flush=True)
        client.subscribe(f"/{userdata}")
    else:
        print("Connection failed with code", rc)

def connect_mqtt(topic: str, on_message):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, topic)
    client.on_connect = on_connect
    client.on_message = on_message
    client.user_data_set(topic)

    broker = os.getenv('MQTT_BROKER', 'localhost')  # 'localhost' es el valor por defecto si la variable no está definida
    client.connect(broker, 1883, 60)
    return client

def connect_node(topic: str, on_message):
    client = connect_mqtt(topic, on_message)
    client.loop_forever()