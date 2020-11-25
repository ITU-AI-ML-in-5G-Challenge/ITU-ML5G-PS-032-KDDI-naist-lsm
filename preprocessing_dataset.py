import pandas as pd
import numpy as np
import glob
import json
from tqdm import tqdm
import os
import datetime
import dateutil.parser
import argparse
import itertools

parser = argparse.ArgumentParser(description="Preprocessing dataset")
parser.add_argument('--dataset', default="output-directory_all",
                    help='dataset path')
parser.add_argument('--label', default="label-for-learning.json",
                    help='dataset path')
parser.add_argument('--resdir', default="results",
                    help='Result path')
args = parser.parse_args()


def create_device_list():
    # Node, interface, networkを紐付けるデータセット
    with open("{:}/network-device-bgpnw2/20200629105300.json".format(args.dataset)) as f:
        data = json.load(f)
    li = []
    for device in data['devices']:
        for interface in device['modules']['openconfig-interfaces']['interfaces']['interface']:
            if 'description' in interface['config']:
                ret = {
                    "device": device['name'],
                    "interface": interface['name'],
                    'network': interface['config']['description']
                }
                li.append(ret)
    df = pd.DataFrame.from_dict(li)
    li = []
    with open("{:}/virtual-infrastructure-bgpnw2/20200629105300.json".format(args.dataset)) as f:
        data = json.load(f)
        for port in data['ports']:
            ret = {
                "device": port['device'],
                "network": port['network'],
                "ip_address": port['ip_address'][0]['ip_address']
            }
            li.append(ret)
    df2 = pd.DataFrame.from_dict(li)
    df3 = pd.merge(df, df2, on=['device', 'network'], how='inner')
    df3.to_csv("{:}/device_list.tsv".format(data_path), sep="\t")
    device_list = pd.read_csv("{:}/device_list.tsv".format(data_path), sep="\t", header=0,
                              names=('device', 'interface', 'network', 'ip_address'))


network_device_path = '{:}/network-device-bgpnw2'.format(args.dataset)
physical_infrastructure_path = '{:}/physical-infrastructure-bgpnw2'.format(args.dataset)
virtual_infrastructure_path = '{:}/virtual-infrastructure-bgpnw2'.format(args.dataset)
label_filename = args.label
data_path = args.resdir

if not os.path.exists(data_path):
    os.makedirs(data_path)

if not os.path.exists("{:}/device_list.tsv".format(data_path)):
    create_device_list()
device_list = pd.read_csv("{:}/device_list.tsv".format(data_path), sep="\t", header=0,
                          names=('device', 'interface', 'network', 'ip_address'))
li_timestamps = pd.read_csv("{:}/timestamp.txt".format(data_path), header=0, sep="\t").timestamp.values


def save_packets():
    # 各Interfaceのincoming/outgoing packets rate
    device_list = pd.read_csv("{:}/device_list.tsv".format(data_path), sep="\t", header=0,
                              names=('device', 'interface', 'network', 'ip_address'))
    network = device_list.network.values
    files = glob.glob("{:}/*".format(virtual_infrastructure_path))
    metrics = ['network-incoming-packets-rate', 'network-outgoing-packets-rate',
               "network-incoming-packets-drop", "network-outgoing-packets-drop",
               "network-incoming-packets-error", "network-outgoing-packets-error"
               ]
    for metric in metrics:
        rets = []
        for filename in tqdm(files):
            timestamp = int(filename.split('/')[-1].split('.')[0])
            with open(filename) as f:
                data = json.load(f)
            ret = {}
            for port in data['ports']:
                if port['network'] in network and metric in port['metrics']:
                    ret["{:}+{:}".format(port['device'], port['network'])] = port['metrics'][metric]
                else:
                    ret["{:}+{:}".format(port['device'], port['network'])] = 0
            ret['timestamp'] = timestamp
            rets.append(ret)
        df = pd.DataFrame.from_dict(rets)
        df_timestamp = pd.read_csv("{:}/timestamp.txt".format(data_path), header=0).timestamp.values
        timestamps = df.timestamp.values
        labels = list(set(df.columns) - {"timestamp"})
        for d in df_timestamp:
            if not d in timestamps:
                ret = {}
                ret['timestamp'] = int(d)
                for l in labels:
                    ret[l] = 0
                rets.append(ret)
        df = pd.DataFrame.from_dict(rets)
        df.to_csv("{:}/{:}.tsv".format(data_path, metric), sep="\t")


def round_time(dt=None, date_delta=datetime.timedelta(minutes=1), to='average'):
    """
    Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    dateDelta : timedelta object, we round to a multiple of this, default 1 minute.
    from:  http://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object-python
    """
    round_to = date_delta.total_seconds()
    if dt is None:
        dt = datetime.now()
    seconds = (dt - dt.min).seconds

    if seconds % round_to == 0:
        rounding = (seconds + round_to / 2) // round_to * round_to
    else:
        if to == 'up':
            # // is a floor division, not a comment on following line (like in javascript):
            rounding = (seconds + round_to) // round_to * round_to
        elif to == 'down':
            rounding = seconds // round_to * round_to
        else:
            rounding = (seconds + round_to / 2) // round_to * round_to

    return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)


def save_failure():
    with open(label_filename) as f:
        data = json.load(f)
    ret = {}
    events = {
        'ixnetwork-bgp-hijacking-start': 1,
        'ixnetwork-bgp-injection-start': 2,
        'node-down': 3,
        'interface-down': 4,
        'tap-loss-start': 5,
        'tap-delay-start': 6,
        'normal': 0
    }
    rets = []
    for recipe in tqdm(data['recipes']):
        if recipe["time"] != 300 or recipe["status"] == 'failed':
            continue
        if 'start' in recipe['type'] or 'down' in recipe['type']:
            jst_time = dateutil.parser.parse(recipe["started_at"])
            utc_time = jst_time + datetime.timedelta(hours=9)
            utc_time = round_time(dt=utc_time, to='up')
            started_at = int(utc_time.strftime("%Y%m%d%H%M%S"))
            jst_time = dateutil.parser.parse(recipe["stopped_at"])
            utc_time = jst_time + datetime.timedelta(hours=9)
            utc_time = round_time(dt=utc_time, to='up')
            stopped_at = int(utc_time.strftime("%Y%m%d%H%M%S"))
            utc_time += datetime.timedelta(minutes=1)
            unstable_stopped_at = int(utc_time.strftime("%Y%m%d%H%M%S"))
            if 'bgp-hijacking-start' in recipe['type']:
                ret["started"] = stopped_at
                ret['status'] = events['ixnetwork-bgp-hijacking-start']
                ret["interface"] = ""
                ret["node"] = ""
                ret["sp"] = recipe["sp"]
                ret["original_sp"] = ""
            if 'bgp-injection-start' in recipe['type']:
                ret["started"] = stopped_at
                ret['status'] = events['ixnetwork-bgp-injection-start']
                ret["interface"] = ""
                ret["node"] = ""
                ret["sp"] = recipe["sp"]
                ret["original_sp"] = recipe["original_sp"]
            if 'node-down' in recipe['type']:
                ret["started"] = stopped_at
                ret['status'] = events['node-down']
                ret["node"] = recipe['node']
                ret["interface"] = ""
                ret["sp"] = ""
                ret["original_sp"] = ""
            if 'interface-down' in recipe['type']:
                ret["started"] = stopped_at
                ret['status'] = events['interface-down']
                ret["interface"] = recipe["ifname"]
                ret["node"] = recipe['node']
                ret["sp"] = ""
                ret["original_sp"] = ""
            if 'tap-loss-start' in recipe['type']:
                ret["started"] = stopped_at
                ret['status'] = events['tap-loss-start']
                ret["interface"] = device_list[(device_list.device == recipe['node']) & (
                    device_list.network == recipe["network"])].interface.values[0]
                ret["node"] = recipe['node']
                ret["sp"] = ""
                ret["original_sp"] = ""
            if 'tap-delay-start' in recipe['type']:
                ret["started"] = stopped_at
                ret['status'] = events['tap-delay-start']
                ret["interface"] = device_list[(device_list.device == recipe['node']) & (
                    device_list.network == recipe["network"])].interface.values[0]
                ret["node"] = recipe['node']
                ret["sp"] = ""
                ret["original_sp"] = ""
            ret["started"] = stopped_at
            ret["unstabled-started"] = unstable_stopped_at
            continue
        else:
            jst_time = dateutil.parser.parse(recipe["started_at"])
            utc_time = jst_time + datetime.timedelta(hours=9)
            utc_time = round_time(dt=utc_time, to='down')
            started_at = int(utc_time.strftime("%Y%m%d%H%M%S"))
            jst_time = dateutil.parser.parse(recipe["stopped_at"])
            utc_time = jst_time + datetime.timedelta(hours=9)
            utc_time = round_time(dt=utc_time, to='down')
            stopped_at = int(utc_time.strftime("%Y%m%d%H%M%S"))
            utc_time += datetime.timedelta(minutes=1)
            unstable_stopped_at = int(utc_time.strftime("%Y%m%d%H%M%S"))
            ret["ended"] = stopped_at
            ret["unstabled-ended"] = unstable_stopped_at
        rets.append(ret)
        ret = {}
    df = pd.DataFrame.from_dict(rets)
    unstabled = list(set(df["unstabled-started"].values)) + list(set(df["unstabled-ended"].values)) + \
        list(set(df["ended"].values)) + list(set(df["started"].values))
    timestamps = pd.read_csv("{:}/timestamp.txt".format(data_path), header=0).timestamp.values
    rets = []
    for timestamp in tqdm(timestamps):
        if timestamp in unstabled:
            continue
        ret = {}
        val = df[(df["started"] < timestamp) & (df["ended"] > timestamp)].status.values
        if len(val) > 0:
            ret["status"] = val[0]
        else:
            ret['status'] = events['normal']  # normal
        ret["timestamp"] = timestamp
        rets.append(ret)
    df1 = pd.DataFrame.from_dict(rets)
    df1 = df1.sort_values("timestamp").reset_index(drop=True)
    df1.to_csv("{:}/label.tsv".format(data_path), sep="\t")
    df1.to_csv("{:}/stable/label.tsv".format(data_path), sep="\t")
    # including unstable and stable
    rets = []
    timestamps = pd.read_csv("{:}/timestamp.txt".format(data_path), header=0).timestamp.values
    for timestamp in tqdm(timestamps):
        ret = {}
        val = df[(df["started"] < timestamp) & (df["ended"] > timestamp)].status.values
        if len(val) > 0:
            ret["status"] = val[0]
        else:
            ret['status'] = events['normal']  # normal
        ret["timestamp"] = timestamp
        rets.append(ret)
    df1 = pd.DataFrame.from_dict(rets)
    df1 = df1.sort_values("timestamp").reset_index(drop=True)
    path = '{:}/unstable'.format(data_path)
    if not os.path.exists(path):
        os.makedirs(path)
    df1.to_csv("{:}/unstable/label.tsv".format(data_path), sep="\t")

    rets = []
    for timestamp in tqdm(timestamps):
        if timestamp in unstabled:
            continue
        ret = {}
        val = df[(df["started"] < timestamp) & (df["ended"] > timestamp)].status.values
        if len(val) > 0:
            node = df[(df["started"] < timestamp) & (df["ended"] > timestamp)].node.values
            interface = df[(df["started"] < timestamp) & (df["ended"] > timestamp)].interface.values
            original_sp = df[(df["started"] < timestamp) & (df["ended"] > timestamp)].original_sp.values
            sp = df[(df["started"] < timestamp) & (df["ended"] > timestamp)].sp.values
            ret["status"] = val[0]
            ret["node"] = node[0]
            ret["interface"] = interface[0]
            ret["original_sp"] = original_sp[0]
            ret["sp"] = sp[0]
        else:
            ret['status'] = events['normal']  # normal
            ret["node"] = ""
            ret["interface"] = ""
            ret["original_sp"] = ""
            ret["sp"] = ""
        ret["timestamp"] = timestamp
        rets.append(ret)
    df1 = pd.DataFrame.from_dict(rets)
    df1 = df1.sort_values("timestamp").reset_index(drop=True)
    path = '{:}/stable'.format(data_path)
    if not os.path.exists(path):
        os.makedirs(path)
    df1.to_csv("{:}/stable/failure.tsv".format(data_path), sep="\t")
    df1.to_csv("{:}/failure.tsv".format(data_path), sep="\t")

    rets = []
    for timestamp in tqdm(timestamps):
        ret = {}
        val = df[(df["started"] < timestamp) & (df["ended"] > timestamp)].status.values
        if len(val) > 0:
            node = df[(df["started"] < timestamp) & (df["ended"] > timestamp)].node.values
            interface = df[(df["started"] < timestamp) & (df["ended"] > timestamp)].interface.values
            original_sp = df[(df["started"] < timestamp) & (df["ended"] > timestamp)].original_sp.values
            sp = df[(df["started"] < timestamp) & (df["ended"] > timestamp)].sp.values
            ret["status"] = val[0]
            ret["node"] = node[0]
            ret["interface"] = interface[0]
            ret["original_sp"] = original_sp[0]
            ret["sp"] = sp[0]
        else:
            ret['status'] = events['normal']  # normal
            ret["node"] = ""
            ret["interface"] = ""
            ret["original_sp"] = ""
            ret["sp"] = ""
        ret["timestamp"] = timestamp
        rets.append(ret)
    df1 = pd.DataFrame.from_dict(rets)
    df1 = df1.sort_values("timestamp").reset_index(drop=True)
    path = '{:}/unstable'.format(data_path)
    if not os.path.exists(path):
        os.makedirs(path)
    df1.to_csv("{:}/unstable/failure.tsv".format(data_path), sep="\t")


def save_cpu_util():
    # 各ノードのcpu 利用率
    files = glob.glob(virtual_infrastructure_path + "/*")
    metrics = ['cpu_util', 'cpu-delta']
    for metric in metrics:
        rets = []
        for filename in tqdm(files):
            with open(filename) as f:
                data = json.load(f)
            timestamp = filename.split('/')[-1].split('.')[0]
            ret = {}
            for d in data['devices']:
                ret[d["name"]] = d['metrics'][metric]
            ret["timestamp"] = int(timestamp)
            rets.append(ret)
        df = pd.DataFrame.from_dict(rets)
        df_timestamp = pd.read_csv("{:}/timestamp.txt".format(data_path), header=0).timestamp.values
        timestamps = df.timestamp.values
        labels = list(set(df.columns) - {"timestamp"})
        for d in df_timestamp:
            if not d in timestamps:
                ret = {}
                ret['timestamp'] = int(d)
                for l in labels:
                    ret[l] = 0
                rets.append(ret)
        df = pd.DataFrame.from_dict(rets)
        df.to_csv("{:}/{:}.tsv".format(data_path, metric), sep="\t")


def save_pps():
    # CAUTION: TOO HEAVEY
    # 各Interfaceのtx (transit), rx (receive) の packet per sec (pps) と kbps
    # delay の検出に利用できそう
    files = glob.glob(network_device_path + "/*")
    metrics = ["tx-kbps", "rx-kbps", "tx-pps", "rx-pps", "in-octets", "in-unicast-pkts", "in-broadcast-pkts", "in-multicast-pkts",
               "in-discards", "out-octets", "out-unicast-pkts", "out-broadcast-pkts", "out-multicast-pkts", "out-discards"]
    rets = {}
    for metric in metrics:
        rets[metric] = []
    for filename in tqdm(files):
        with open(filename) as f:
            data = json.load(f)
        ret = {}
        for metric in metrics:
            ret[metric] = {}
        timestamp = int(filename.split('/')[-1].split('.')[0])
        if not timestamp in li_timestamps:
            continue
        for device in data["devices"]:
            if "Cisco-IOS-XE-interfaces-oper" in device["modules"].keys():
                for interface in device["modules"]["Cisco-IOS-XE-interfaces-oper"]["interfaces"]["interface"]:
                    statistics = interface['statistics']
                    networks = device_list[(device_list.device == device["name"]) & (
                        device_list.interface == interface["name"])]["network"].values
                    if len(networks) == 0:
                        continue
                    label = "{:}+{:}".format(device["name"], networks[0])
                    for metric in metrics:
                        ret[metric][label] = statistics[metric]
        for metric in metrics:
            ret[metric]["timestamp"] = timestamp
            rets[metric].append(ret[metric])
    for metric in metrics:
        df = pd.DataFrame.from_dict(rets[metric])
        df_timestamp = pd.read_csv("{:}/timestamp.txt".format(data_path), header=0).timestamp.values
        timestamps = df.timestamp.values
        labels = list(set(df.columns) - {"timestamp"})
        for d in df_timestamp:
            if not d in timestamps:
                ret = {}
                ret['timestamp'] = int(d)
                for l in labels:
                    ret[l] = 0
                rets[metric].append(ret)
        df = pd.DataFrame.from_dict(rets[metric])
        df.to_csv("{:}/{:}.tsv".format(data_path, metric), sep="\t")


def save_interface():
    _dict = {
        "up": 1,
        "UP": 1,
        "down": 0,
        "DOWN": 0
    }
    files = glob.glob(network_device_path + "/*")
    metrics = ["admin-status"]
    rets = {}
    for metric in metrics:
        rets[metric] = []
    for filename in tqdm(files):
        with open(filename) as f:
            data = json.load(f)
        ret = {}
        for metric in metrics:
            ret[metric] = {}
        timestamp = int(filename.split('/')[-1].split('.')[0])
        if not timestamp in li_timestamps:
            continue
        for device in data["devices"]:
            if "openconfig-interfaces" in device["modules"].keys():
                for interface in device["modules"]["openconfig-interfaces"]["interfaces"]["interface"]:
                    #                 statistics = interface['state']['oper-status']
                    networks = device_list[(device_list.device == device["name"]) & (
                        device_list.interface == interface["name"])]["network"].values
                    if len(networks) == 0:
                        continue
                    label = "{:}+{:}".format(device["name"], networks[0])
    #                 print(label, interface['state']['oper-status'], interface['state']['admin-status'])
                    for metric in metrics:
                        ret[metric][label] = _dict[interface['state'][metric]]
        for metric in metrics:
            ret[metric]["timestamp"] = timestamp
            rets[metric].append(ret[metric])
    for metric in metrics:
        df = pd.DataFrame.from_dict(rets[metric])
        df_timestamp = pd.read_csv("{:}/timestamp.txt".format(data_path), header=0).timestamp.values
        timestamps = df.timestamp.values
        labels = list(set(df.columns) - {"timestamp"})
        for d in df_timestamp:
            if not d in timestamps:
                ret = {}
                ret['timestamp'] = int(d)
                for l in labels:
                    ret[l] = 0
                rets[metric].append(ret)
        df = pd.DataFrame.from_dict(rets[metric])
        df.to_csv("{:}/{:}.tsv".format(data_path, metric), sep="\t")


def save_queue():
    files = glob.glob(network_device_path + "/*")
    metrics = ["output-pkts", "output-bytes", "queue-size-pkts", "queue-size-bytes", "drop-pkts", "drop-bytes"]
    rets = {}
    for metric in metrics:
        rets[metric] = []
    for filename in tqdm(files):
        with open(filename) as f:
            data = json.load(f)
        ret = {}
        for metric in metrics:
            ret[metric] = {}
        timestamp = int(filename.split('/')[-1].split('.')[0])
        if not timestamp in li_timestamps:
            continue
        for device in data["devices"]:
            if "Cisco-IOS-XE-interfaces-oper" in device["modules"].keys():
                for interface in device["modules"]["Cisco-IOS-XE-interfaces-oper"]["interfaces"]["interface"]:
                    if 'diffserv-info' in interface:
                        statistics = interface['diffserv-info'][0]['diffserv-target-classifier-stats'][0]['queuing-stats']
                        networks = device_list[(device_list.device == device["name"]) & (
                            device_list.interface == interface["name"])]["network"].values
                        if len(networks) == 0:
                            continue
                        label = "{:}+{:}".format(device["name"], networks[0])
                        for metric in metrics:
                            ret[metric][label] = statistics[metric]
        for metric in metrics:
            ret[metric]["timestamp"] = timestamp
            rets[metric].append(ret[metric])
    for metric in metrics:
        df = pd.DataFrame.from_dict(rets[metric])
        df_timestamp = pd.read_csv("{:}/timestamp.txt".format(data_path), header=0).timestamp.values
        timestamps = df.timestamp.values
        labels = list(set(df.columns) - {"timestamp"})
        for d in df_timestamp:
            if not d in timestamps:
                ret = {}
                ret['timestamp'] = int(d)
                for l in labels:
                    ret[l] = 0
                rets[metric].append(ret)
        df = pd.DataFrame.from_dict(rets[metric])
        df.to_csv("{:}/{:}.tsv".format(data_path, metric), sep="\t")


def save_bgp_state_data():
    rets = []
    files = glob.glob(network_device_path + "/*")
    for filename in tqdm(files):
        with open(filename) as f:
            data = json.load(f)
        timestamp = int(filename.split('/')[-1].split('.')[0])
        if not timestamp in li_timestamps:
            continue
        for device in data["devices"]:
            if "Cisco-IOS-XE-bgp-oper" in device["modules"].keys():
                if 'neighbors' in device["modules"]["Cisco-IOS-XE-bgp-oper"]["bgp-state-data"]:
                    for neighbor in device["modules"]["Cisco-IOS-XE-bgp-oper"]["bgp-state-data"]["neighbors"]['neighbor']:
                        state = neighbor['bgp-neighbor-counters']
                        activity = neighbor["prefix-activity"]
                        ret = {
                            'timestamp': timestamp,
                            'device': device['name'],
                            'neighbor-id': neighbor["neighbor-id"],
                            'as': neighbor['as'],
                            'installed-prefixes': neighbor['installed-prefixes'],
                            'bgp-neighbor-counters-sent-opens': state['sent']['opens'],
                            'bgp-neighbor-counters-sent-updates': state['sent']['updates'],
                            'bgp-neighbor-counters-sent-notifications': state['sent']['notifications'],
                            'bgp-neighbor-counters-sent-route-refreshes': state['sent']['route-refreshes'],
                            'bgp-neighbor-counters-received-opens': state['received']['opens'],
                            'bgp-neighbor-counters-received-updates': state['received']['updates'],
                            'bgp-neighbor-counters-received-notifications': state['received']['notifications'],
                            'bgp-neighbor-counters-received-route-refreshes': state['received']['route-refreshes'],
                            'bgp-neighbor-counters-inq-depth': state['inq-depth'],
                            'bgp-neighbor-counters-outq-depth': state['outq-depth'],
                            'prefix-activity-sent-current-prefixes': activity['sent']['current-prefixes'],
                            'prefix-activity-sent-total-prefixes': activity['sent']['total-prefixes'],
                            'prefix-activity-sent-implicit-withdraw': activity['sent']['implicit-withdraw'],
                            'prefix-activity-sent-explicit-withdraw': activity['sent']['explicit-withdraw'],
                            'prefix-activity-sent-bestpaths': activity['sent']['bestpaths'],
                            'prefix-activity-sent-multipaths': activity['sent']['multipaths'],
                            'prefix-activity-received-current-prefixes': activity['received']['current-prefixes'],
                            'prefix-activity-received-total-prefixes': activity['received']['total-prefixes'],
                            'prefix-activity-received-implicit-withdraw': activity['received']['implicit-withdraw'],
                            'prefix-activity-received-explicit-withdraw': activity['received']['explicit-withdraw'],
                            'prefix-activity-received-bestpaths': activity['received']['bestpaths'],
                            'prefix-activity-received-multipaths': activity['received']['multipaths'],
                        }
                        rets.append(ret)
    df = pd.DataFrame.from_dict(rets)
    df.to_csv("{:}/{:}.tsv".format(data_path, "bgp-state-data"), sep="\t")


def save_bgp_state_data_details():
    df = pd.read_csv("{:}/bgp-state-data.tsv".format(data_path), sep='\t', index_col=0)
    df = df.sort_values("timestamp")
    df_timestamps = list(df.groupby(["timestamp"]).mean().reset_index().timestamp.values)
    timestamps = pd.read_csv("{:}/timestamp.txt".format(data_path), sep='\t').timestamp.values
    neighbors = df.groupby(["neighbor-id", 'as', 'device']).mean().reset_index()[["neighbor-id", 'as', 'device']].values
    metrics = [
        'bgp-neighbor-counters-received-updates',
        'bgp-neighbor-counters-sent-updates',
        'prefix-activity-received-current-prefixes',
        'prefix-activity-sent-current-prefixes',
        'installed-prefixes',
        'prefix-activity-received-bestpaths',
        'prefix-activity-sent-bestpaths',
        'bgp-neighbor-counters-sent-opens',
        'bgp-neighbor-counters-sent-updates',
        'bgp-neighbor-counters-sent-notifications',
        'bgp-neighbor-counters-sent-route-refreshes',
        'bgp-neighbor-counters-received-opens',
        'bgp-neighbor-counters-received-updates',
        'bgp-neighbor-counters-received-notifications',
        'bgp-neighbor-counters-received-route-refreshes',
        'bgp-neighbor-counters-inq-depth',
        'bgp-neighbor-counters-outq-depth',
    ]
    rets = {}
    for metric in metrics:
        rets[metric] = []
    for timestamp in tqdm(timestamps):
        ret = {}
        for metric in metrics:
            ret[metric] = {}
        if timestamp in df_timestamps:
            for neighbor in neighbors:
                neighbor_id = neighbor[0]
                AS = neighbor[1]
                device = neighbor[2]
                label = "{:}-{:}-{:}".format(device, neighbor_id, AS)
                for metric in metrics:
                    val = df[(df.timestamp == timestamp) & (df["as"] == AS) & (
                        df["neighbor-id"] == neighbor_id) & (df["device"] == device)][metric].values
                    if len(val) > 0:
                        ret[metric][label] = val[0]
                    else:
                        ret[metric][label] = 0
        else:
            for neighbor in neighbors:
                neighbor_id = neighbor[0]
                AS = neighbor[1]
                device = neighbor[2]
                label = "{:}-{:}-{:}".format(device, neighbor_id, AS)
                for metric in metrics:
                    ret[metric][label] = 0
        for metric in metrics:
            ret[metric]["timestamp"] = timestamp
            rets[metric].append(ret[metric])
    for metric in metrics:
        df = pd.DataFrame.from_dict(rets[metric])
        df.to_csv("{:}/{:}.tsv".format(data_path, metric), sep="\t")


def save_as_path():
    files = glob.glob(network_device_path + "/*")
    filename1 = "{:}/as-path.txt".format(data_path)
    with open(filename1, "w") as f1:
        f1.write("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n".format("timestamp", "device", "prefix",
                                                                                  "nexthop", "as-path", "origin", 'metric', "local-pref", "weight", "as", "as-path-size"))
        for filename in tqdm(files):
            with open(filename) as f:
                data = json.load(f)
            timestamp = int(filename.split('/')[-1].split('.')[0])
            if not timestamp in li_timestamps:
                continue
            for d in data['devices']:
                if 'Cisco-IOS-XE-bgp-oper' in d['modules'].keys():
                    if 'bgp-route-vrfs' in d['modules']["Cisco-IOS-XE-bgp-oper"]['bgp-state-data']:
                        entries = d['modules']["Cisco-IOS-XE-bgp-oper"]['bgp-state-data']['bgp-route-vrfs']['bgp-route-vrf'][
                            0]['bgp-route-afs']['bgp-route-af'][0]['bgp-route-filters']["bgp-route-entries"]["bgp-route-entry"]
                        for entry in entries:
                            f1.write("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n".format(timestamp, d["name"], entry['prefix'], entry['bgp-path-entries']['bgp-path-entry'][0]['nexthop'], entry['bgp-path-entries']['bgp-path-entry'][0]['as-path'], entry['bgp-path-entries']['bgp-path-entry'][0]['origin'], entry['bgp-path-entries'][
                                     'bgp-path-entry'][0]['metric'], entry['bgp-path-entries']['bgp-path-entry'][0]['local-pref'], entry['bgp-path-entries']['bgp-path-entry'][0]['weight'], entry['bgp-path-entries']['bgp-path-entry'][0]['as-path'].split(" ")[0], len(entry['bgp-path-entries']['bgp-path-entry'][0]['as-path'].split(" "))))
                            # f1.write("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n".format(timestamp, d["name"], entry['prefix'], entry['bgp-path-entries']['bgp-path-entry'][0]['nexthop'], entry['bgp-path-entries']['bgp-path-entry'][0]['as-path'], entry['bgp-path-entries']['bgp-path-entry'][0]['origin'], entry['bgp-path-entries']['bgp-path-entry'][0]['metric'], entry['bgp-path-entries']['bgp-path-entry'][0]['local-pref'], entry['bgp-path-entries']['bgp-path-entry'][0]['weight']), entry['bgp-path-entries']['bgp-path-entry'][0]['as-path'].split(" ")[0], len(entry['bgp-path-entries']['bgp-path-entry'][0]['as-path'].split(" ")))


def reformat(path):
    df = pd.read_csv("{:}/label.tsv".format(path), header=0, sep="\t")
    li_timestamps = df.timestamp.values

    metrics = ['network-incoming-packets-rate', 'network-outgoing-packets-rate',
               "network-incoming-packets-drop", "network-outgoing-packets-drop",
               "network-incoming-packets-error", "network-outgoing-packets-error"
               ]
    for metric in metrics:
        df = pd.read_csv("{:}/{:}.tsv".format(data_path, metric), sep="\t", index_col=0)
        df = df[['IntGW-01+link-intgw-exgw-a-1',
                 'IntGW-01+link-intgw-exgw-a-2', 'IntGW-01+link-intgw-exgw-a-3', 'IntGW-01+link-tr-intgw-a-1',
                 'IntGW-01+link-tr-intgw-b-1',
                 'IntGW-02+link-intgw-exgw-b-1', 'IntGW-02+link-intgw-exgw-b-2',
                 'IntGW-02+link-intgw-exgw-b-3',
                 'IntGW-02+link-tr-intgw-a-2', 'IntGW-02+link-tr-intgw-b-2',
                 'RR-01+link-tr-intgw-a-3', 'RR-01+link-tr-intgw-b-3',
                 'TR-01+link-tr-intgw-a-1', 'TR-01+link-tr-intgw-a-2',
                 'TR-01+link-tr-intgw-a-3',
                 'TR-01+link-tr-ssm-a-1', 'TR-01+link-tr-tr-a-1',
                 'TR-02+link-tr-intgw-b-1', 'TR-02+link-tr-intgw-b-2',
                 'TR-02+link-tr-intgw-b-3', 'TR-02+link-tr-ssm-b-1',
                 'TR-02+link-tr-tr-a-1', 'timestamp']]
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df[df.timestamp.isin(li_timestamps)]
        df = df.reset_index(drop=True)
        df.to_csv("{:}/{:}.tsv".format(path, metric), sep="\t")

    df = pd.read_csv("{:}/cpu_util.tsv".format(data_path), sep="\t", index_col=0)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df.timestamp.isin(li_timestamps)]
    df = df.reset_index(drop=True)
    df.to_csv("{:}/cpu_util.tsv".format(path), sep="\t")

    df = pd.read_csv("{:}/cpu_util.tsv".format(data_path), sep="\t", index_col=0)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df.timestamp.isin(li_timestamps)]
    df = df.reset_index(drop=True)
    ret = {
        "IntGW-01": [
            "IntGW-01_Gi2",
            "IntGW-01_Gi3",
            "IntGW-01_Gi5",
            "IntGW-01_Gi6",
            "IntGW-01_Gi7"
        ],
        "IntGW-02": [
            "IntGW-02_Gi2",
            "IntGW-02_Gi3",
            "IntGW-02_Gi5",
            "IntGW-02_Gi6",
            "IntGW-02_Gi7"
        ],
        "RR-01": [
            "RR-01_Gi2",
            "RR-01_Gi3"
        ],
        "TR-01": [
            "TR-01_Gi2",
            "TR-01_Gi3",
            "TR-01_Gi4",
            "TR-01_Gi5",
            "TR-01_Gi6"
        ],
        "TR-02": [
            "TR-02_Gi2",
            "TR-02_Gi3",
            "TR-02_Gi4",
            "TR-02_Gi5",
            "TR-02_Gi6"
        ]
    }
    devices = ret.keys()
    for device in devices:
        for r in ret[device]:
            df[r] = df.apply(lambda x: x[device], axis=1)
    df[sorted(set(df.columns) - set(ret.keys()))].to_csv("{:}/cpu-util.tsv".format(path), sep="\t")

    # bgp_metrics = [
    #     'bgp-neighbor-counters-received-updates',
    #     'bgp-neighbor-counters-sent-updates',
    #     'prefix-activity-received-current-prefixes',
    #     'prefix-activity-sent-current-prefixes',
    #     'installed-prefixes',
    #     'prefix-activity-received-bestpaths',
    #     'prefix-activity-sent-bestpaths',
    #     'bgp-neighbor-counters-sent-opens',
    #     'bgp-neighbor-counters-sent-updates',
    #     'bgp-neighbor-counters-sent-notifications',
    #     'bgp-neighbor-counters-sent-route-refreshes',
    #     'bgp-neighbor-counters-received-opens',
    #     'bgp-neighbor-counters-received-updates',
    #     'bgp-neighbor-counters-received-notifications',
    #     'bgp-neighbor-counters-received-route-refreshes',
    #     'bgp-neighbor-counters-inq-depth',
    #     'bgp-neighbor-counters-outq-depth',
    # ]
    # for metric in bgp_metrics:
    #     df = pd.read_csv("{:}/{:}.tsv".format(data_path, metric), sep="\t", index_col=0)
    #     df = df[set(df.columns) - {'IntGW-01-10.20.1.250-2516', 'IntGW-02-10.20.1.250-2516',
    #                                'RR-01-10.20.1.251-2516', 'RR-01-10.20.1.252-2516'}]
    #     df = df.sort_values("timestamp").reset_index(drop=True)
    #     df = df[df.timestamp.isin(li_timestamps)]
    #     df = df.reset_index(drop=True)
    #     ret = {
    #         "timestamp": "timestamp",
    #         "IntGW-01-10.30.2.2-10": "IntGW-01_Gi5",
    #         "IntGW-01-10.30.2.6-20": "IntGW-01_Gi6",
    #         "IntGW-01-10.30.2.10-30": "IntGW-01_Gi7",
    #         "IntGW-02-10.30.2.14-10": "IntGW-02_Gi5",
    #         "IntGW-02-10.30.2.18-20": "IntGW-02_Gi6",
    #         "IntGW-02-10.30.2.22-30": "IntGW-02_Gi7",
    #     }
    #     df.columns = df.columns.map(ret)
    #     df[sorted(df.columns)].to_csv("{:}/{:}.tsv".format(path, metric), sep="\t")

    df = pd.read_csv("{:}/admin-status.tsv".format(data_path), sep="\t", index_col=0)
    df = df[['IntGW-01+link-intgw-exgw-a-1', 'IntGW-01+link-intgw-exgw-a-2',
             'IntGW-01+link-intgw-exgw-a-3',
             'IntGW-01+link-tr-intgw-a-1', 'IntGW-01+link-tr-intgw-b-1',
             'IntGW-02+link-intgw-exgw-b-1', 'IntGW-02+link-intgw-exgw-b-2',
             'IntGW-02+link-intgw-exgw-b-3',
             'IntGW-02+link-tr-intgw-a-2', 'IntGW-02+link-tr-intgw-b-2',
             'RR-01+link-tr-intgw-a-3', 'RR-01+link-tr-intgw-b-3',
             'TR-01+link-tr-intgw-a-1', 'TR-01+link-tr-intgw-a-2',
             'TR-01+link-tr-intgw-a-3',
             'TR-01+link-tr-ssm-a-1', 'TR-01+link-tr-tr-a-1',
             'TR-02+link-tr-intgw-b-1', 'TR-02+link-tr-intgw-b-2',
             'TR-02+link-tr-intgw-b-3', 'TR-02+link-tr-ssm-b-1',
             'TR-02+link-tr-tr-a-1', 'timestamp']]
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df.timestamp.isin(li_timestamps)]
    df = df.reset_index(drop=True)
    df.to_csv("{:}/admin-status.tsv".format(path), sep="\t")

    metrics = ["tx-kbps", "rx-kbps", "tx-pps", "rx-pps", "in-octets", "in-unicast-pkts", "in-broadcast-pkts", "in-multicast-pkts",
               "in-discards", "out-octets", "out-unicast-pkts", "out-broadcast-pkts", "out-multicast-pkts", "out-discards"]
    for metric in metrics:
        df = pd.read_csv("{:}/{:}.tsv".format(data_path, metric), sep="\t", index_col=0)
        df = df[['IntGW-01+link-intgw-exgw-a-1', 'IntGW-01+link-intgw-exgw-a-2',
                 'IntGW-01+link-intgw-exgw-a-3',
                 'IntGW-01+link-tr-intgw-a-1', 'IntGW-01+link-tr-intgw-b-1',
                 'IntGW-02+link-intgw-exgw-b-1', 'IntGW-02+link-intgw-exgw-b-2',
                 'IntGW-02+link-intgw-exgw-b-3',
                 'IntGW-02+link-tr-intgw-a-2', 'IntGW-02+link-tr-intgw-b-2',
                 'RR-01+link-tr-intgw-a-3', 'RR-01+link-tr-intgw-b-3',
                 'TR-01+link-tr-intgw-a-1', 'TR-01+link-tr-intgw-a-2',
                 'TR-01+link-tr-intgw-a-3',
                 'TR-01+link-tr-ssm-a-1', 'TR-01+link-tr-tr-a-1',
                 'TR-02+link-tr-intgw-b-1', 'TR-02+link-tr-intgw-b-2',
                 'TR-02+link-tr-intgw-b-3', 'TR-02+link-tr-ssm-b-1',
                 'TR-02+link-tr-tr-a-1', 'timestamp']]
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df[df.timestamp.isin(li_timestamps)]
        df = df.reset_index(drop=True)
        df.to_csv("{:}/{:}.tsv".format(path, metric), sep="\t")

    metrics = ["output-pkts", "output-bytes", "queue-size-pkts", "queue-size-bytes", "drop-pkts", "drop-bytes"]
    for metric in metrics:
        df = pd.read_csv("{:}/{:}.tsv".format(data_path, metric), sep="\t", index_col=0)
        df = df.fillna(0)
        df = df[[
            'IntGW-01+link-intgw-exgw-a-1', 'IntGW-01+link-intgw-exgw-a-2',
            'IntGW-01+link-intgw-exgw-a-3', 'IntGW-02+link-intgw-exgw-b-1',
            'IntGW-02+link-intgw-exgw-b-2', 'IntGW-02+link-intgw-exgw-b-3',
            'timestamp'
        ]]
        df = df.sort_values("timestamp").reset_index(drop=True)
        df[list(set(df.columns) - {'timestamp'})] = df[list(set(df.columns) - {'timestamp'})].diff().fillna(0)
        df = df[df.timestamp.isin(li_timestamps)]
        df = df.reset_index(drop=True)
        df.to_csv("{:}/{:}.tsv".format(path, metric), sep="\t")

    df = pd.read_csv("{:}/failure.tsv".format(path), index_col=0, sep="\t")
    rets = []
    interfaces = {
        "IntGW-01": [
            "IntGW-01_Gi2",
            "IntGW-01_Gi3",
            "IntGW-01_Gi5",
            "IntGW-01_Gi6",
            "IntGW-01_Gi7"
        ],
        "IntGW-02": [
            "IntGW-02_Gi2",
            "IntGW-02_Gi3",
            "IntGW-02_Gi5",
            "IntGW-02_Gi6",
            "IntGW-02_Gi7"
        ],
        "RR-01": [
            "RR-01_Gi2",
            "RR-01_Gi3"
        ],
        "TR-01": [
            "TR-01_Gi2",
            "TR-01_Gi3",
            "TR-01_Gi4",
            "TR-01_Gi5",
            "TR-01_Gi6"
        ],
        "TR-02": [
            "TR-02_Gi2",
            "TR-02_Gi3",
            "TR-02_Gi4",
            "TR-02_Gi5",
            "TR-02_Gi6"
        ]
    }
    interface_map = {
        "GigabitEthernet2": "Gi2",
        "GigabitEthernet3": "Gi3",
        "GigabitEthernet4": "Gi4",
        "GigabitEthernet5": "Gi5",
        "GigabitEthernet6": "Gi6",
        "GigabitEthernet7": "Gi7"
    }
    for d in df.to_dict("record"):
        ret = {
            "timestamp": d["timestamp"],
            'status': d['status']
        }
        for interface in list(itertools.chain.from_iterable(interfaces.values())):
            ret[interface] = 0
        if d["status"] == 3.0:
            for interface in interfaces[d["node"]]:
                ret[interface] = 1
        if d["status"] == 4.0:
            ret[d["node"] + "_" + interface_map[d["interface"]]] = 1
        rets.append(ret)
    df = pd.DataFrame.from_dict(rets)
    df.to_csv("{:}/failure-interfaces.tsv".format(path), sep="\t")

    df = pd.read_csv("{:}/failure.tsv".format(path), index_col=0, sep="\t")
    rets = []
    for d in df.to_dict("record"):
        ret = {
            "timestamp": d["timestamp"],
            'status': d['status']
        }
        for node in interfaces.keys():
            ret[node] = 0
        if d["status"] == 3.0:
            ret[d["node"]] = 1
        rets.append(ret)
    df = pd.DataFrame.from_dict(rets)
    df.to_csv("{:}/failure-nodes.tsv".format(path), sep="\t")

    # df = pd.read_csv("{:}/as-path.txt".format(data_path), sep="\t", index_col=0)
    # df = df.reset_index()
    # df = df.groupby(["timestamp", "device", "as"]).mean().reset_index()
    # df["device-as"] = df["device"] + "_" + df["as"].astype(str)
    # df = df.pivot_table(index="timestamp", columns="device-as", values="as-path-size").reset_index()
    # df = df[["timestamp", "IntGW-01_10", "IntGW-01_20", "IntGW-01_30", "IntGW-02_10", "IntGW-02_20", "IntGW-02_30"]]
    # ret = {
    #         "timestamp": "timestamp",
    #         "IntGW-01_10": "IntGW-01_Gi5",
    #         "IntGW-01_20": "IntGW-01_Gi6" ,
    #         "IntGW-01_30":"IntGW-01_Gi7" ,
    #         "IntGW-02_10": "IntGW-02_Gi5",
    #         "IntGW-02_20": "IntGW-02_Gi6",
    #         "IntGW-02_30": "IntGW-02_Gi7",
    #         }
    # df.columns = df.columns.map(ret)
    # df = df[df.timestamp.isin(li_timestamps)]
    # tmp = set(li_timestamps) - set(df[df["timestamp"].isin(li_timestamps)].timestamp.values)
    # rets = []
    # for t in tmp:
    #     ret = {
    #             "timestamp": t,
    #             "IntGW-01_Gi5": 0,
    #             "IntGW-01_Gi6": 0,
    #             "IntGW-01_Gi7": 0,
    #             "IntGW-02_Gi5": 0,
    #             "IntGW-02_Gi6": 0,
    #             "IntGW-02_Gi7": 0,
    #             }
    #     rets.append(ret)
    # df = pd.concat([df, pd.DataFrame.from_dict(rets)])
    # df = df.sort_values("timestamp").reset_index(drop=True)
    # df = df.fillna(0)
    # df[sorted(df.columns)].to_csv("{:}/{:}.tsv".format(path, "as-path"), sep="\t")


def main():
    # print("as-path")
    # save_as_path()
    # print("bgp-state-data")
    # save_bgp_state_data()
    # save_bgp_state_data_details()
    print("queue")
    save_queue()
    print("failure")
    save_failure()
    print("cpu")
    save_cpu_util()
    print("interface")
    save_interface()
    print("packets")
    save_packets()
    print("pps")
    save_pps()
    print("reformat")
    path = '{:}/stable'.format(data_path)
    if not os.path.exists(path):
        os.makedirs(path)
    reformat(path)
    path = '{:}/unstable'.format(data_path)
    if not os.path.exists(path):
        os.makedirs(path)
    reformat(path)


if __name__ == '__main__':
    main()
