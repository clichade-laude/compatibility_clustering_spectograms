from utils.mqtt import connect_node, publish_mqtt, log_time
import json, time

class Arguments(object):
    def __init__(self, initial_data):
        for key, value in initial_data.items():
            setattr(self, key, value)
        self.orig_dataset = self.dataset
        self.cluster = False if not args.poison else self.cluster

def exec_poison(client):
    publish_mqtt(client, "poison", dataset=args.dataset, poison=args.poison, folder=args.folder)

def exec_cluster(client):
    publish_mqtt(client, "cluster", dataset=args.dataset, model=args.model, batch=args.batch, folder=args.folder)

def exec_train(client):
    publish_mqtt(client, "train", dataset=args.dataset, model=args.model, epochs=args.epochs, batch=args.batch, cluster=args.cluster, folder=args.folder)

def exec_test(client):
    publish_mqtt(client, "test", dataset=args.dataset, orig_dataset=args.orig_dataset, batch=args.batch, folder=args.folder, poison=args.poison)

def on_message(client, userdata, msg):
    params = json.loads(msg.payload)
    node = params.pop("node")
    print(f"{log_time()} Node: control | Received msg from {node}", flush=True)

    ## Arguments configuration
    if node == "start":
        global args
        args = Arguments(params)
        print(f"{log_time()} Saving results on {args.folder}", flush=True)
    elif node == "poison":
        args.dataset = params.pop("dataset")

    ## Start and stop monitoring
    if node == "start":
        publish_mqtt(client, "start_monitor", folder=args.folder)
        time.sleep(10) ## Wait for monitor to start
    elif node == "test":
        publish_mqtt(client, "stop_monitor")

    ## Actions execution
    if node == "test":
        print(f"{log_time()} Node: controller | Workflow finished", flush=True)
    elif node == "train":
        exec_test(client)
    elif node == "start" and args.poison:
        exec_poison(client)
    elif node == "poison" and args.cluster:
        exec_cluster(client)
    else:
        exec_train(client)

if __name__ == "__main__":
    connect_node("control", "control", on_message)



