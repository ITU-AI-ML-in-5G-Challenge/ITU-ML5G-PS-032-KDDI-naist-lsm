import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
from PIL import Image
import datetime
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib as mpl
import matplotlib.dates as mdates
from tqdm import tqdm
import graphviz as gv
import glob
import os
import cv2
import warnings
warnings.filterwarnings('ignore')

events = {
    0: 'normal',
    1: 'route information failure',
    2: 'route information failure',
    3: 'network element failure',
    4: 'interface failure',
    5: 'interface failure',
    6: 'interface failure'
}

event_types = {
    0: 'normal',
    1: 'route information failure',
    2: 'route information failure',
    3: 'network element failure',
    4: 'interface failure',
    5: 'interface failure',
    6: 'interface failure'
}
event_name = {
    0: 'Normal',
    1: 'BGP Hijacking',
    2: 'BGP Injection',
    3: 'NE reboot',
    4: 'Interface Down',
    5: 'Packet Delay/Loss',
}


def draw_network(filename, timestamp, failure_points):
    status = int(failure_points['status'])
    g = gv.Graph('G', engine="neato", format='png')
    g.attr(  # label=events[event_id],
        dpi="300",
        bgcolor="#343434", style="filled", fontcolor="white",
        labelloc="t",
        labeljust="c",
        fontsize="18",
        margin="0",
        rankdir="TB",
        splines="spline",
        ranksep="1.0",
        nodesep="0")
    if event_types[status] == events[3] and failure_points['node'] == 'IntGW-01':
        g.node("IntGW-01", shape='cylinder', label="IntGW-01", style="filled", fillcolor="#C2036D",
               fontcolor="white", fixedsize='true', width="1.2", height="1.8", pos="8, 5!")
    else:
        g.node("IntGW-01", shape='cylinder', label="IntGW-01", style="filled", fillcolor="#3B80CC",
               fontcolor="white", fixedsize='true', width="1.2", height="1.8", pos="8, 5!")
    if event_types[status] == events[4] and failure_points['node'] == 'IntGW-01' and failure_points['interface'] == 'GigabitEthernet2':
        g.node("IntGW-01#Gi2", shape='rectangle', label="Gi2", style="filled",
               fillcolor="#C2036D", fixedsize='true', width="0.5", height="0.35", pos="7.2, 5.2!")
    else:
        g.node("IntGW-01#Gi2", shape='rectangle', label="Gi2", style="filled",
               fillcolor="white", fixedsize='true', width="0.5", height="0.35", pos="7.2, 5.2!")
    if event_types[status] == events[4] and failure_points['node'] == 'IntGW-01' and failure_points['interface'] == 'GigabitEthernet3':
        g.node("IntGW-01#Gi3", shape='rectangle', label="Gi3", style="filled",
               fillcolor="#C2036D", fixedsize='true', width="0.5", height="0.35", pos="7.2, 4.7!")
    else:
        g.node("IntGW-01#Gi3", shape='rectangle', label="Gi3", style="filled",
               fillcolor="white", fixedsize='true', width="0.5", height="0.35", pos="7.2, 4.7!")
    if event_types[status] == events[4] and failure_points['node'] == 'IntGW-01' and failure_points['interface'] == 'GigabitEthernet5':
        g.node("IntGW-01#Gi5", shape='rectangle', label="Gi5", style="filled",
               fillcolor="#C2036D", fixedsize='true', width="0.5", height="0.35", pos="8.7, 5.4!")
    else:
        g.node("IntGW-01#Gi5", shape='rectangle', label="Gi5", style="filled",
               fillcolor="white", fixedsize='true', width="0.5", height="0.35", pos="8.7, 5.4!")
    if event_types[status] == events[4] and failure_points['node'] == 'IntGW-01' and failure_points['interface'] == 'GigabitEthernet6':
        g.node("IntGW-01#Gi6", shape='rectangle', label="Gi6", style="filled",
               fillcolor="#C2036D", fixedsize='true', width="0.5", height="0.35", pos="8.7, 4.9!")
    else:
        g.node("IntGW-01#Gi6", shape='rectangle', label="Gi6", style="filled",
               fillcolor="white", fixedsize='true', width="0.5", height="0.35", pos="8.7, 4.9!")
    if event_types[status] == events[4] and failure_points['node'] == 'IntGW-01' and failure_points['interface'] == 'GigabitEthernet7':
        g.node("IntGW-01#Gi7", shape='rectangle', label="Gi7", style="filled",
               fillcolor="#C2036D", fixedsize='true', width="0.5", height="0.35", pos="8.7, 4.4!")
    else:
        g.node("IntGW-01#Gi7", shape='rectangle', label="Gi7", style="filled",
               fillcolor="white", fixedsize='true', width="0.5", height="0.35", pos="8.7, 4.4!")

    if event_types[status] == events[3] and failure_points['node'] == 'IntGW-02':
        g.node("IntGW-02", shape='cylinder', label="IntGW-02", style="filled", fillcolor="#C2036D",
               fontcolor="white", fixedsize='true', width="1.2", height="1.8", pos="8, 2!")
    else:
        g.node("IntGW-02", shape='cylinder', label="IntGW-02", style="filled", fillcolor="#3B80CC",
               fontcolor="white", fixedsize='true', width="1.2", height="1.8", pos="8, 2!")
    if event_types[status] == events[4] and failure_points['node'] == 'IntGW-02' and failure_points['interface'] == 'GigabitEthernet2':
        g.node("IntGW-02#Gi2", shape='rectangle', label="Gi2", style="filled",
               fillcolor="#C2036D", fixedsize='true', width="0.5", height="0.35", pos="7.2, 2.2!")
    else:
        g.node("IntGW-02#Gi2", shape='rectangle', label="Gi2", style="filled",
               fillcolor="white", fixedsize='true', width="0.5", height="0.35", pos="7.2, 2.2!")
    if event_types[status] == events[4] and failure_points['node'] == 'IntGW-02' and failure_points['interface'] == 'GigabitEthernet3':
        g.node("IntGW-02#Gi3", shape='rectangle', label="Gi3", style="filled",
               fillcolor="#C2036D", fixedsize='true', width="0.5", height="0.35", pos="7.2, 1.7!")
    else:
        g.node("IntGW-02#Gi3", shape='rectangle', label="Gi3", style="filled",
               fillcolor="white", fixedsize='true', width="0.5", height="0.35", pos="7.2, 1.7!")
    if event_types[status] == events[4] and failure_points['node'] == 'IntGW-02' and failure_points['interface'] == 'GigabitEthernet5':
        g.node("IntGW-02#Gi5", shape='rectangle', label="Gi5", style="filled",
               fillcolor="#C2036D", fixedsize='true', width="0.5", height="0.35", pos="8.7, 2.4!")
    else:
        g.node("IntGW-02#Gi5", shape='rectangle', label="Gi5", style="filled",
               fillcolor="white", fixedsize='true', width="0.5", height="0.35", pos="8.7, 2.4!")
    if event_types[status] == events[4] and failure_points['node'] == 'IntGW-02' and failure_points['interface'] == 'GigabitEthernet6':
        g.node("IntGW-02#Gi6", shape='rectangle', label="Gi6", style="filled",
               fillcolor="#C2036D", fixedsize='true', width="0.5", height="0.35", pos="8.7, 1.9!")
    else:
        g.node("IntGW-02#Gi6", shape='rectangle', label="Gi6", style="filled",
               fillcolor="white", fixedsize='true', width="0.5", height="0.35", pos="8.7, 1.9!")
    if event_types[status] == events[4] and failure_points['node'] == 'IntGW-02' and failure_points['interface'] == 'GigabitEthernet7':
        g.node("IntGW-02#Gi7", shape='rectangle', label="Gi7", style="filled",
               fillcolor="#C2036D", fixedsize='true', width="0.5", height="0.35", pos="8.7, 1.4!")
    else:
        g.node("IntGW-02#Gi7", shape='rectangle', label="Gi7", style="filled",
               fillcolor="white", fixedsize='true', width="0.5", height="0.35", pos="8.7, 1.4!")

    if event_types[status] == events[3] and failure_points['node'] == 'RR-01':
        g.node("RR#01", shape='cylinder', label="RR-01", style="filled", fillcolor="#C2036D",
               fontcolor="white", fixedsize='true', width="1.2", height="1.5", pos="8, 0!")
    else:
        g.node("RR#01", shape='cylinder', label="RR-01", style="filled", fillcolor="#3B80CC",
               fontcolor="white", fixedsize='true', width="1.2", height="1.5", pos="8, 0!")
    if event_types[status] == events[4] and failure_points['node'] == 'RR-01' and failure_points['interface'] == 'GigabitEthernet2':
        g.node("RR#01#Gi2", shape='rectangle', label="Gi2", style="filled", fillcolor="#C2036D",
               fixedsize='true', width="0.5", height="0.35", pos="7.2, 0.2!")
    else:
        g.node("RR#01#Gi2", shape='rectangle', label="Gi2", style="filled", fillcolor="white",
               fixedsize='true', width="0.5", height="0.35", pos="7.2, 0.2!")
    if event_types[status] == events[4] and failure_points['node'] == 'RR-01' and failure_points['interface'] == 'GigabitEthernet3':
        g.node("RR#01#Gi3", shape='rectangle', label="Gi3", style="filled", fillcolor="#C2036D",
               fixedsize='true', width="0.5", height="0.35", pos="7.2, -0.3!")
    else:
        g.node("RR#01#Gi3", shape='rectangle', label="Gi3", style="filled", fillcolor="white",
               fixedsize='true', width="0.5", height="0.35", pos="7.2, -0.3!")

    if event_types[status] == events[3] and failure_points['node'] == 'TR-01':
        g.node("TR-01", shape='cylinder', label="TR-01", style="filled", fillcolor="#C2036D",
               fontcolor="white", fixedsize='true', width="1.2", height="1.8", pos="5, 5!")
    else:
        g.node("TR-01", shape='cylinder', label="TR-01", style="filled", fillcolor="#3B80CC",
               fontcolor="white", fixedsize='true', width="1.2", height="1.8", pos="5, 5!")
    if event_types[status] == events[4] and failure_points['node'] == 'TR-01' and failure_points['interface'] == 'GigabitEthernet2':
        g.node("TR-01#Gi2", shape='rectangle', label="Gi2", style="filled", fillcolor="#C2036D",
               fixedsize='true', width="0.5", height="0.35", pos="5.7, 5.4!")
    else:
        g.node("TR-01#Gi2", shape='rectangle', label="Gi2", style="filled", fillcolor="white",
               fixedsize='true', width="0.5", height="0.35", pos="5.7, 5.4!")
    if event_types[status] == events[4] and failure_points['node'] == 'TR-01' and failure_points['interface'] == 'GigabitEthernet3':
        g.node("TR-01#Gi3", shape='rectangle', label="Gi3", style="filled", fillcolor="#C2036D",
               fixedsize='true', width="0.5", height="0.35", pos="5.7, 4.9!")
    else:
        g.node("TR-01#Gi3", shape='rectangle', label="Gi3", style="filled", fillcolor="white",
               fixedsize='true', width="0.5", height="0.35", pos="5.7, 4.9!")
    if event_types[status] == events[4] and failure_points['node'] == 'TR-01' and failure_points['interface'] == 'GigabitEthernet4':
        g.node("TR-01#Gi4", shape='rectangle', label="Gi4", style="filled", fillcolor="#C2036D",
               fixedsize='true', width="0.5", height="0.35", pos="5.7, 4.4!")
    else:
        g.node("TR-01#Gi4", shape='rectangle', label="Gi4", style="filled", fillcolor="white",
               fixedsize='true', width="0.5", height="0.35", pos="5.7, 4.4!")
    if event_types[status] == events[4] and failure_points['node'] == 'TR-01' and failure_points['interface'] == 'GigabitEthernet5':
        g.node("TR-01#Gi5", shape='rectangle', label="Gi5", style="filled",
               fillcolor="#C2036D", fixedsize='true', width="0.5", height="0.35", pos="5, 4!")
    else:
        g.node("TR-01#Gi5", shape='rectangle', label="Gi5", style="filled",
               fillcolor="white", fixedsize='true', width="0.5", height="0.35", pos="5, 4!")
    if event_types[status] == events[4] and failure_points['node'] == 'TR-01' and failure_points['interface'] == 'GigabitEthernet6':
        g.node("TR-01#Gi6", shape='rectangle', label="Gi6", style="filled",
               fillcolor="#C2036D", fixedsize='true', width="0.5", height="0.35", pos="4.2, 5!")
    else:
        g.node("TR-01#Gi6", shape='rectangle', label="Gi6", style="filled",
               fillcolor="white", fixedsize='true', width="0.5", height="0.35", pos="4.2, 5!")

    if event_types[status] == events[3] and failure_points['node'] == 'TR-02':
        g.node("TR-02", shape='cylinder', label="TR-02", style="filled", fillcolor="#C2036D",
               fontcolor="white", fixedsize='true', width="1.2", height="1.8", pos="5, 2!")
    else:
        g.node("TR-02", shape='cylinder', label="TR-02", style="filled", fillcolor="#3B80CC",
               fontcolor="white", fixedsize='true', width="1.2", height="1.8", pos="5, 2!")
    if event_types[status] == events[4] and failure_points['node'] == 'TR-02' and failure_points['interface'] == 'GigabitEthernet2':
        g.node("TR-02#Gi2", shape='rectangle', label="Gi2", style="filled", fillcolor="#C2036D",
               fixedsize='true', width="0.5", height="0.35", pos="5.7, 2.4!")
    else:
        g.node("TR-02#Gi2", shape='rectangle', label="Gi2", style="filled", fillcolor="white",
               fixedsize='true', width="0.5", height="0.35", pos="5.7, 2.4!")
    if event_types[status] == events[4] and failure_points['node'] == 'TR-02' and failure_points['interface'] == 'GigabitEthernet3':
        g.node("TR-02#Gi3", shape='rectangle', label="Gi3", style="filled", fillcolor="#C2036D",
               fixedsize='true', width="0.5", height="0.35", pos="5.7, 1.9!")
    else:
        g.node("TR-02#Gi3", shape='rectangle', label="Gi3", style="filled", fillcolor="white",
               fixedsize='true', width="0.5", height="0.35", pos="5.7, 1.9!")
    if event_types[status] == events[4] and failure_points['node'] == 'TR-02' and failure_points['interface'] == 'GigabitEthernet4':
        g.node("TR-02#Gi4", shape='rectangle', label="Gi4", style="filled", fillcolor="#C2036D",
               fixedsize='true', width="0.5", height="0.35", pos="5.7, 1.4!")
    else:
        g.node("TR-02#Gi4", shape='rectangle', label="Gi4", style="filled", fillcolor="white",
               fixedsize='true', width="0.5", height="0.35", pos="5.7, 1.4!")
    if event_types[status] == events[4] and failure_points['node'] == 'TR-02' and failure_points['interface'] == 'GigabitEthernet5':
        g.node("TR-02#Gi5", shape='rectangle', label="Gi5", style="filled",
               fillcolor="#C2036D", fixedsize='true', width="0.5", height="0.35", pos="5, 3!")
    else:
        g.node("TR-02#Gi5", shape='rectangle', label="Gi5", style="filled",
               fillcolor="white", fixedsize='true', width="0.5", height="0.35", pos="5, 3!")
    if event_types[status] == events[4] and failure_points['node'] == 'TR-02' and failure_points['interface'] == 'GigabitEthernet6':
        g.node("TR-02#Gi6", shape='rectangle', label="Gi6", style="filled",
               fillcolor="#C2036D", fixedsize='true', width="0.5", height="0.35", pos="4.2, 2!")
    else:
        g.node("TR-02#Gi6", shape='rectangle', label="Gi6", style="filled",
               fillcolor="white", fixedsize='true', width="0.5", height="0.35", pos="4.2, 2!")

    g.node("5GGW#01", shape='cylinder', label="5GGW#01", style="filled", fillcolor="#3B80CC", fontcolor="white", pos="3, 5!")
    g.node("5GGW#02", shape='cylinder', label="5GGW#02", style="filled", fillcolor="#3B80CC", fontcolor="white", pos="3, 2!")
    g.node("AGW#01", shape='cylinder', label="AGW#01", style="filled",
           fillcolor="#3B80CC", fontcolor="white", pos="10, 5.5!")
    g.node("BGW#01", shape='cylinder', label="BGW#01", style="filled",
           fillcolor="#3B80CC", fontcolor="white", pos="10, 4.5!")
    g.node("CGW#01", shape='cylinder', label="CGW#01", style="filled",
           fillcolor="#3B80CC", fontcolor="white", pos="10, 3.5!")
    g.node("AGW#02", shape='cylinder', label="AGW#02", style="filled",
           fillcolor="#3B80CC", fontcolor="white", pos="10, 2.5!")
    g.node("BGW#02", shape='cylinder', label="BGW#02", style="filled",
           fillcolor="#3B80CC", fontcolor="white", pos="10, 1.5!")
    g.node("CGW#02", shape='cylinder', label="CGW#02", style="filled",
           fillcolor="#3B80CC", fontcolor="white", pos="10, 0.5!")

    g.node("A#AS-10", shape='cylinder', label="A#AS-10", style="filled", fillcolor="white", pos="12, 5.5!")
    g.node("B#AS-20", shape='cylinder', label="B#AS-20", style="filled", fillcolor="white", pos="12, 3.5!")
    g.node("C#AS-30", shape='cylinder', label="C#AS-30", style="filled", fillcolor="white", pos="12, 1.5!")

    g.edge("TR-01#Gi2", "IntGW-01#Gi2")
    g.edge("TR-02#Gi2", "IntGW-01#Gi3")
    g.edge("AGW#01", "IntGW-01#Gi5")
    g.edge("BGW#01", "IntGW-01#Gi6")
    g.edge("CGW#01", "IntGW-01#Gi7")
    g.edge("TR-01#Gi3", "IntGW-02#Gi2")
    g.edge("TR-02#Gi3", "IntGW-02#Gi3")
    g.edge("AGW#02", "IntGW-02#Gi5")
    g.edge("BGW#02", "IntGW-02#Gi6")
    g.edge("CGW#02", "IntGW-02#Gi7")

    g.edge("TR-01#Gi4", "RR#01#Gi2")
    g.edge("TR-02#Gi4", "RR#01#Gi3")
    g.edge("TR-02#Gi5", "TR-01#Gi5")
    g.edge("5GGW#01", "TR-01#Gi6")
    g.edge("5GGW#02", "TR-02#Gi6")

    g.edge("AGW#01", "A#AS-10")
    g.edge("AGW#02", "A#AS-10")
    g.edge("BGW#01", "B#AS-20")
    g.edge("BGW#02", "B#AS-20")
    g.edge("CGW#01", "C#AS-30")
    g.edge("CGW#02", "C#AS-30")

    # if event_types[status] == events[1] and events[status] == events[2] and failure_points['original_sp'] == 'A' and failure_points['sp'] == 'B':
    #     g.edge("A#AS-10", "B#AS-20", arrowhead="forward", style="filled", fillcolor="red")
    # if event_types[status] == events[1] and events[status] == events[2] and failure_points['original_sp'] == 'A' and failure_points['sp'] == 'C':
    #     g.edge("A#AS-10", "C#AS-30", arrowhead="forward", style="filled", fillcolor="red")
    # if event_types[status] == events[1] and events[status] == events[2] and failure_points['original_sp'] == 'B' and failure_points['sp'] == 'A':
    #     g.edge("B#AS-20", "A#AS-10", arrowhead="forward", style="filled", fillcolor="red")
    # if event_types[status] == events[1] and events[status] == events[2] and failure_points['original_sp'] == 'B' and failure_points['sp'] == 'C':
    #     g.edge("B#AS-20", "C#AS-30", arrowhead="forward", style="filled", fillcolor="red")
    # if event_types[status] == events[1] and events[status] == events[2] and failure_points['original_sp'] == 'C' and failure_points['sp'] == 'A':
    #     g.edge("C#AS-30", "A#AS-10", arrowhead="forward", style="filled", fillcolor="red")
    # if event_types[status] == events[1] and events[status] == events[2] and failure_points['original_sp'] == 'C' and failure_points['sp'] == 'B':
    #     g.edge("C#AS-30", "B#AS-20", arrowhead="forward", style="filled", fillcolor="red")
    g.render(filename)


def draw_networks(result_dir):
    if not os.path.exists("/tmp/actual"):
        os.makedirs("/tmp/actual")
    df = pd.read_csv("{:}/classification.tsv".format(result_dir), sep="\t")
    actual_failure = pd.read_csv("dataset_eval/stable/failure.tsv", sep="\t", index_col=0)

    timestamps = df.timestamp.values
    event_ids = df.pred.values
    target_ids = df.status.values

    for i in tqdm(range(len(df))):
        timestamp = timestamps[i]
        # draw actual failures
        filename = "/tmp/actual/{:}".format(timestamp)
        failure_points = actual_failure[actual_failure.timestamp == timestamp].to_dict("records")[0]
        tdatetime = dt.strptime(str(timestamp), '%Y%m%d%H%M%S')
        draw_network(filename, tdatetime.strftime('%Y/%m/%d %H:%M'), failure_points)


def draw(result_dir="results"):
    if not os.path.exists("/tmp/figs"):
        os.makedirs("/tmp/figs")
    plt.rcParams['savefig.facecolor'] = '#343434'
    plt.rcParams['figure.facecolor'] = '#343434'
    mpl.rcParams['xtick.color'] = 'w'
    mpl.rcParams['ytick.color'] = 'w'
    mpl.rcParams['axes.labelcolor'] = 'w'

    label = pd.read_csv("{:}/classification.tsv".format(result_dir), sep="\t")
    timestamps = label.timestamp.values
    event_ids  = label.pred.values

    accuracy = pd.read_csv("{:}/classification.tsv".format(result_dir), sep="\t")
    accuracy["nrofevent_types"] = accuracy.index + 1
    accuracy["nrof_acc"] = accuracy.apply(lambda x: 1 if x["status"] == x["pred"] else 0, axis=1)
    accuracy["acc_cumsum"] = accuracy["nrof_acc"].cumsum()
    accuracy["acc"] = accuracy["acc_cumsum"] / accuracy["nrofevent_types"]
    accuracy["datetime"] = pd.to_datetime(accuracy.timestamp.astype(str), format='%Y%m%d%H%M%S')
    accuracy["datetime"] = mdates.date2num(accuracy["datetime"])
    accuracy.index = accuracy.datetime

    actual_failure = pd.read_csv("dataset_eval/stable/failure.tsv", sep="\t", index_col=0)
    target_ids = actual_failure.status.values

    device = pd.read_csv("dataset_eval/device_list.tsv", sep="\t", index_col=0)
    devices = sorted(list(set(device.device.values)))

    cpu_util = pd.read_csv("dataset_eval/stable/cpu_util.tsv", sep="\t", index_col=0)
    cpu_util["datetime"] = pd.to_datetime(cpu_util.timestamp.astype(str), format='%Y%m%d%H%M%S')
    cpu_util["datetime"] = mdates.date2num(cpu_util["datetime"])
    cpu_util.index = cpu_util.datetime

    interface = pd.read_csv("dataset_eval/stable/admin-status.tsv", sep="\t", index_col=0)
    interface = interface.fillna(0)
    interface["datetime"] = pd.to_datetime(interface.timestamp.astype(str), format='%Y%m%d%H%M%S')
    interface["datetime"] = mdates.date2num(interface["datetime"])
    interface.index = interface.datetime

    incoming_packets = pd.read_csv("dataset_eval/stable/network-incoming-packets-rate.tsv", sep="\t", index_col=0)
    incoming_packets["datetime"] = pd.to_datetime(incoming_packets.timestamp.astype(str), format='%Y%m%d%H%M%S')
    incoming_packets["datetime"] = mdates.date2num(incoming_packets["datetime"])
    incoming_packets.index = incoming_packets.datetime

    for i in tqdm(range(len(label))):
        timestamp = int(timestamps[i])
        target_id = int(target_ids[i])
        event_id  = int(event_ids[i])
        if target_id == 6:
            target_id = 5

        fig = plt.figure(figsize=(30, 24))
        fig.patch.set_facecolor('#343434')
        gs_master = GridSpec(nrows=5, ncols=12, height_ratios=[1.5, 0.4, 0.4, 0.4, 0.4])
        gs_master.update(wspace=0.5, hspace=0.40)

        gs = GridSpecFromSubplotSpec(nrows=1, ncols=12, subplot_spec=gs_master[0, 0:12])
        ax = fig.add_subplot(gs[:, :])

        filename = "/tmp/actual/{:}.png".format(timestamp)
        tdatetime = dt.strptime(str(timestamp), '%Y%m%d%H%M%S')
        im = Image.open(filename)
        im_list = np.asarray(im)
        ax.imshow(im_list)

        ax.text(100, 2150, "Actual failure type: {:}   Predicted failure type: {:}".format(
            event_name[target_id], event_name[event_id]), ha='left', color="white", fontsize=18)
        ax.text(100, 2300, "", ha='left',
                color="white", fontsize=18)
        ax.tick_params(bottom=False,
                       left=False,
                       right=False,
                       top=False)
        ax.tick_params(labelbottom=False,
                       labelleft=False,
                       labelright=False,
                       labeltop=False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        ax.set_title("Actual failure point", fontsize=18, color="white")

        # plot accuracy transition
        gs = GridSpecFromSubplotSpec(nrows=1, ncols=12, subplot_spec=gs_master[1, 0:11])
        ax = fig.add_subplot(gs[:, :])
        ax.patch.set_facecolor('#343434')

        df = accuracy[accuracy.datetime <= mdates.date2num(tdatetime)]
        df[set(df.columns) - {"timestamp"}].plot(x="datetime", y="acc", ax=ax, marker="s")
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylim(0.7, 1.05)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.set_xticks([mdates.date2num(tdatetime - datetime.timedelta(minutes=t)) for t in reversed(range(0, 8))
                       ] + [mdates.date2num(tdatetime + datetime.timedelta(minutes=t)) for t in range(1, 4)])
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=30, fontsize=10)
        ax.set_xlim(mdates.date2num(tdatetime - datetime.timedelta(minutes=7)),
                    mdates.date2num(tdatetime + datetime.timedelta(minutes=3)))
        ax.get_legend().remove()
        ax.set_title("Accuracy", fontsize=18, color="white")

        # CPU Utilization
        gs = GridSpecFromSubplotSpec(nrows=1, ncols=12, subplot_spec=gs_master[2, 0:11])
        ax = fig.add_subplot(gs[:, :])
        ax.patch.set_facecolor('#343434')

        df = cpu_util[cpu_util.datetime <= mdates.date2num(tdatetime)]
        df[set(df.columns) - {"timestamp"}].plot(x="datetime", ax=ax, marker="s")
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.set_ylabel("CPU Utilization", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.set_xticks([mdates.date2num(tdatetime - datetime.timedelta(minutes=t)) for t in reversed(range(0, 8))
                       ] + [mdates.date2num(tdatetime + datetime.timedelta(minutes=t)) for t in range(1, 4)])
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=30, fontsize=10)
        ax.set_xlim(mdates.date2num(tdatetime - datetime.timedelta(minutes=7)),
                    mdates.date2num(tdatetime + datetime.timedelta(minutes=3)))
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels, loc="lower left")
        ax.set_title("CPU Utilization", fontsize=18, color="white")

        # Interface condition
        for j, d in enumerate(devices):
            ncol = 2 * j
            gs = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[3, ncol:ncol + 2])
            ax = fig.add_subplot(gs[:, :])
            ax.patch.set_facecolor('#343434')
            df = interface[interface.datetime <= mdates.date2num(tdatetime)]
            df = df[[c for c in df.columns if d in c]]
            df.columns = [device[(device.network == n) & (device.device == d)].interface.values[0]
                          for n in [c.split('+')[1] for c in df.columns if d in c]]
            df.plot(ax=ax, marker='s')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            if j == 0:
                ax.set_ylabel("Interface condition", fontsize=14)
                ax.set_yticks((0, 1))
                ax.set_yticklabels(('DOWN', 'UP'))
            else:
                ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylim(-0.05, 1.05)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.set_xticks([mdates.date2num(tdatetime - datetime.timedelta(minutes=7)),
                           mdates.date2num(tdatetime)])
            # ax.set_xticks([mdates.date2num(tdatetime-datetime.timedelta(minutes=7)),
            #                mdates.date2num(tdatetime), mdates.date2num(tdatetime+datetime.timedelta(minutes=3))])
            labels = ax.get_xticklabels()
            plt.setp(labels, rotation=30, fontsize=10)
            ax.set_xlim(mdates.date2num(tdatetime - datetime.timedelta(minutes=7)),
                        mdates.date2num(tdatetime + datetime.timedelta(minutes=3)))
            ax.set_title(d, fontsize=18, color="white")
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(handles, labels, loc="lower left")

        for j, d in enumerate(devices):
            ncol = 2 * j
            gs = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[4, ncol:ncol + 2])
            ax = fig.add_subplot(gs[:, :])
            ax.patch.set_facecolor('#343434')
            df = incoming_packets[incoming_packets.datetime <= mdates.date2num(tdatetime)]
            df = df[[c for c in df.columns if d in c]]
            df.columns = [device[(device.network == n) & (device.device == d)].interface.values[0]
                          for n in [c.split('+')[1] for c in df.columns if d in c]]
            df.plot(ax=ax, marker='s')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            if j == 0:
                ax.set_ylabel("Incoming packets rate", fontsize=14)
            else:
                ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylim(0, 5000)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            # ax.set_xticks([mdates.date2num(tdatetime-datetime.timedelta(minutes=7)),
            #                mdates.date2num(tdatetime), mdates.date2num(tdatetime+datetime.timedelta(minutes=3))])
            ax.set_xticks([mdates.date2num(tdatetime - datetime.timedelta(minutes=7)),
                           mdates.date2num(tdatetime)])
            labels = ax.get_xticklabels()
            plt.setp(labels, rotation=30, fontsize=10)
            ax.set_xlim(mdates.date2num(tdatetime - datetime.timedelta(minutes=7)),
                        mdates.date2num(tdatetime + datetime.timedelta(minutes=3)))
            ax.set_title(d, fontsize=18, color="white")
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(handles, labels, loc="lower left")

        filename = "/tmp/figs/{:}.jpg".format(timestamp)
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.05)
        plt.close()


def conv_movie(result_dir="results"):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_array = []
    for filename in tqdm(sorted(glob.glob("/tmp/figs/*.jpg"))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    filename = '{:}/project.mp4'.format(result_dir)
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MP4V'), 5.0, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == '__main__':
    input_dir = "results/gcn"
    result_dir = "results/movie"
    # draw_networks(input_dir)
    draw(input_dir)
    conv_movie(result_dir)
