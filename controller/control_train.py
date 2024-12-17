from utils.mqtt import connect_node, publish_mqtt
import json, time

class Arguments(object):
    def __init__(self, initial_data):
        for key, value in initial_data.items():
            setattr(self, key, value)
        self.poison = True if self.poison_info != None else False


def exec_poison(client):
    publish_mqtt(client, "poison", dataset=args.dataset, poison=args.poison_info, folder=args.folder)

def exec_cluster(client):
    publish_mqtt(client, "cluster", dataset=args.dataset, model=args.model, batch=args.batch, folder=args.folder)

def exec_train(client):
    publish_mqtt(client, "train", dataset=args.dataset, model=args.model, epochs=args.epochs, batch=args.batch, cluster=args.cluster, folder=args.folder)

def on_message(client, userdata, msg):
    params = json.loads(msg.payload)
    node = params.pop("node")
    print(f"Node: controller | Received msg from {node}", flush=True)

    ## Arguments configuration
    if node == "start":
        global args
        args = Arguments(params)
        print(f"Saving results on {args.folder}", flush=True)
    elif node == "poison":
        args.dataset = params.pop("dataset")

    ## Start and stop monitoring
    if node == "start":
        publish_mqtt(client, "start_monitor")
        time.sleep(10) ## Wait for monitor to start
    elif node == "train":
        publish_mqtt(client, "stop_monitor")

    ## Actions execution
    if node == "train":
        print("Node: controller | Workflow finished", flush=True)
    elif node == "start" and args.poison:
        exec_poison(client)
    elif node == "poison" and args.cluster:
        exec_cluster(client)
    else:
        exec_train(client)

if __name__ == "__main__":
    connect_node("control", on_message)



