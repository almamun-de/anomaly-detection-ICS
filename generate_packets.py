#!/usr/bin/env python
# coding: utf-8


from scapy.all import *
import binascii
import cip
import argparse
import numpy as np
import pandas as pd
import enip_tcp
import struct
import argparse


argParser = argparse.ArgumentParser()
argParser.add_argument("-f", "--file", help="The csv files to be read.")
args = argParser.parse_args()

df = pd.read_csv(args.file)

df = df[df.columns[df.columns!='timestamp']]


rawpkt = b'\x00\x00\xbc\xd1`\xdax\xe7\xd1\xe0\x02^\x08\x00E\x00\x00zp&@\x00\x80\x06\x00\x00\x8dQ\x00\n\x8dQ\x00S\xc4c\xaf\x12\xdd\x88\x8d\x87\x94\x95CQP\x18\xf9t\x1bl\x00\x00p\x00:\x00\x00\x01\x02\x10\x00\x00\x00\x00\x1a9/\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x00\x02\x00\xa1\x00\x04\x00\t\x135\x00\xb1\x00&\x00\xe4j\n\x02 \x02$\x01\x02\x00\x06\x00\x12\x00L\x02 r$\x00\x00\xce\x04\x00\x01\x00L\x02 r$\x00,=\x04\x00\x01\x00'


for col in df.columns:
    pkt = Ether(rawpkt)
    pkt_lst=[]
    lst = df[col].to_list()
    name = args.file+'-'+str(col)+'.pcap'
    for item in lst:
        load = Raw(struct.pack("f", item))
        pkt[cip.CIP].payload = load
        pkt_lst.append(pkt)

    wrpcap(name, pkt_lst)
    

