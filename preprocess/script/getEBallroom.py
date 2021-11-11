# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:14:33 2016

@author: marchand
"""
from __future__ import print_function
import xml.etree.ElementTree as ET
import hashlib
import os
import sys
import time
try:
    from urllib import urlretrieve
except:
    from urllib.request import urlretrieve

import os


path_xml = './'
path_data = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), '_data', 'EBallroom')

# without threading, it may takes one/two hours to download the full dataset
# even with a good connexion, with threading and a full speed connexion
# it may take only 10/20 minutes.
threading = True
n_thread = 8

# do not download again already downloaded files
skip_already_downloaded_mp3 = True

# do not modify
version = '1.1'
mp3_link_template = 'http://media.ballroomdancers.com/mp3/{}.mp3'


# %% functions
def open_xml(path_xml, version):
    filename = os.path.join(path_xml, 'extendedballroom_v{}.xml'.format(version))
    tree = ET.parse(filename).getroot()
    return tree


def hashfile(afile, hasher, blocksize=65536):
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    return hasher.hexdigest()


def download(list_, threading=True, n_thread=8):

    print('Downloading {} files. This may take a while.'.format(len(list_)))
    if threading:
        try:
            import queue
        except:
            import Queue as queue
        import threading
        import time

        exit_flag = False
        class ThreadDLMp3(threading.Thread):
            """Thread to download multiple files at once"""
            def __init__(self, queue):
                threading.Thread.__init__(self)
                self.queue = queue

            def run(self):
                while not exit_flag:
                    mp3_link, mp3_filepath = self.queue.get()
                    if os.path.exists(mp3_filepath):
                        os.remove(mp3_filepath)
                    urlretrieve (mp3_link, mp3_filepath)
                    self.queue.task_done()

        q = queue.Queue()
        threads = []
        for i in range(n_thread):
            t = ThreadDLMp3(q)
            t.setDaemon(True)
            t.start()
            threads.append(t)

    for mp3_link, mp3_filepath in list_:
        if threading:
            q.put([mp3_link, mp3_filepath])
        else:
            urlretrieve (mp3_link, mp3_filepath)

    if threading:
        try:
            ##Wait for queue to be exhausted and then exit main program
            while not q.empty():
                pass
            # Stop the threads
            exit_flag = True
        except (KeyboardInterrupt, SystemExit) as e:
            sys.exit(e)

def check(folder, xml_root, sublist=[]):
    filelist = []
    hashlist = []
    download_again = []
    print('Checking dataset')
    for genre_node in xml_root:
        genre_folder = os.path.join(folder, genre_node.tag)
        for song_node in genre_node:
            song_id = song_node.get('id')
            mp3_filepath = os.path.join(genre_folder, genre_node.tag + '.' + song_id + '.mp3')
            if sublist and mp3_filepath not in sublist:
                continue
            h = hashfile(open(mp3_filepath, 'rb'), hashlib.md5())
            status = song_node.get('hash') == h
            filelist.append(mp3_filepath)
            hashlist.append(status)

            if not status:
                print('Error with file {}, expected hash {}, found hash {}'
                      .format(mp3_filepath, song_node.get('hash'), h))
                download_again.append((mp3_link_template.format(song_id), mp3_filepath))

    print('{} out of {} are valid'.format(sum(hashlist), len(filelist)))

    return download_again

# %% download
if not os.path.exists(path_data):
    os.mkdir(path_data)

xml_root = open_xml(path_xml, version)
download_list = []
for genre_node in xml_root:
    genre_folder = os.path.join(path_data, genre_node.tag)
    if not os.path.exists(genre_folder):
        os.mkdir(genre_folder)
    for song_node in genre_node:
        song_id = song_node.get('id')
        mp3_filepath = os.path.join(genre_folder, genre_node.tag + '.' + song_id + '.mp3')
        if skip_already_downloaded_mp3 and os.path.exists(mp3_filepath):
            continue
        download_list.append((mp3_link_template.format(song_id), mp3_filepath))

download(download_list, threading, n_thread)


# %% md5 sums check
download_again = check(path_data, xml_root)
while download_again:
    download(download_again, threading, n_thread)
    time.sleep(2)
    download_again = check(path_data, xml_root, sublist=[b for a, b in download_again])
