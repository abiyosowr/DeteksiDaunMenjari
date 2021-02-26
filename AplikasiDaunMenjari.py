import tkinter as tk
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

from tkinter import *
from tkinter import font as tkfont
from tkinter import messagebox

import tkinter.messagebox 
import cv2

tk = Tk()
tk.title("Aplikasi Deteksi Bunga")
tk.iconbitmap("flower.ico")
tk.resizable(width=False,height=False)
Kanvas = Canvas(tk,width=720,height=480, background="white")
background_image = PhotoImage ( file = "flower21.gif")
background_label = Label(tk, image=background_image)
background_label.place(x=0, y=35, relwidth=1, relheight=1)

lebar_home = 520
tinggi_home = 320

tk.configure(width=lebar_home,height=tinggi_home)
frame1 = Frame(tk,width=lebar_home,height=tinggi_home)

lebar_tampilan = tk.winfo_screenwidth()
tinggi_tampilan = tk.winfo_screenheight()
x = (lebar_tampilan/2)-(lebar_home/2)
y = (tinggi_tampilan/2)-(tinggi_home/2)
tk.geometry("%dx%d+%d+%d" % (lebar_home,tinggi_home,x,y))

def deteksi():
    cap = cv2.VideoCapture(0)

    sys.path.append("..")
     
    from utils import label_map_util
     
    from utils import visualization_utils as vis_util
     
    MODEL_NAME = 'Daun_Menjari'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
     
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
     
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data','Daun_label.pbtxt')
     
    NUM_CLASSES = 5

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.compat.v1.GraphDef()
      with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
     
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
     
    with detection_graph.as_default():
      with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.60)
     
            cv2.imshow('object detection', cv2.resize(image_np, (1440,720)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def tentang_aplikasi():
    jendela_about=Tk()
    jendela_about.title("Tentang Aplikasi")
    jendela_about.resizable(width=False,height=False)
    jendela_about.configure(background='grey')
    jendela_about.iconbitmap("flower.ico")

    lebar=720
    tinggi=480

    jendela_about.configure(width=lebar,height=tinggi)

    lebar_tampilan=jendela_about.winfo_screenwidth()
    tingg_tampilan=jendela_about.winfo_screenheight()
    x=(lebar_tampilan/2)-(lebar/2)
    y=(tinggi_tampilan/2)-(tinggi/2)
    jendela_about.geometry("%dx%d+%d+%d"%(lebar,tinggi,x,y))

    about=Label(jendela_about,
    text="Aplikasi ini merupakan aplikasi object detection untuk jenis-jenis bunga.\n\n"
    "Jenis bunga yang dapat dideteksi aplikasi ini adalah :\n\n"
    "1.  Bunga Anggrek\n\n"
    "2.  Bunga Aster\n\n"
    "3.  Bunga Dandelion\n\n"
    "4.  Bunga Kamboja\n\n"
    "5.  Bunga Lily\n\n"
    "6.  Bunga Matahari\n\n"
    "7.  Bunga Mawar\n\n"
    "8.  Bunga Melati\n\n"
    "9.  Bunga Tulip\n\n"
    "10. Kembang Sepatu\n",bg="white",fg="black",font="verdana 12 bold", 
    height=50,width=90)
    kembali = Button(jendela_about,
    text= "Keluar",
    font="Verdana 8 bold",
    bg='red',
    fg='white',
    height=2,
    width=20,
    command = jendela_about.destroy)

    kembali.pack(side=BOTTOM)
    about.pack()
    jendela_about.mainloop()

def bantuan():
    jendela_bantuan=Tk()
    jendela_bantuan.title("Bantuan")
    jendela_bantuan.resizable(width=False,height=False)
    jendela_bantuan.configure(background='grey')
    jendela_about.iconbitmap("flower.ico")

    lebar=720
    tinggi=360

    jendela_bantuan.configure(width=lebar,height=tinggi)

    lebar_tampilan=jendela_bantuan.winfo_screenwidth()
    tingg_tampilan=jendela_bantuan.winfo_screenheight()
    x=(lebar_tampilan/2)-(lebar/2)
    y=(tinggi_tampilan/2)-(tinggi/2)
    jendela_bantuan.geometry("%dx%d+%d+%d"%(lebar,tinggi,x,y))

    about=Label(jendela_bantuan,
    text="1. Jalankan 'Aplikasi Pendeteksi Bunga' \n\n"
    "2. Klik pada tombol 'Mulai Deteksi'\n\n"
    "3. Jika sudah selesai mendeteksi, "
    "silahkan tekan tombol 'q' pada Keyboard\n\n"
    "4. Jika sudah selesai menggunakan "
    "'Aplikasi Pendeteksi Bunga' "
    "silahkan klik tombol 'Keluar' \n",bg="white",fg="black",font="verdana 10 bold",
    height=50,width=90)
    kembali = Button(jendela_bantuan,
    text= "Keluar",
    font="Verdana 8 bold",
    bg='red',
    fg='white',
    height=2,
    width=20,
    command = jendela_bantuan.destroy)

    kembali.pack(side=BOTTOM)
    about.pack()
    jendela_bantuan.mainloop()

    # Membuat function yang isinya sebuah message konfirmasi
def Message_Keluar():
    answer = tkinter.messagebox.askquestion("Keluar", "Apakah anda ingin keluar dari aplikasi ?",icon = 'warning')
    if answer == 'yes':
        tk.destroy()
    else :
        tkinter.messagebox.showinfo('Return','Anda akan kembali ke menu utama')

        space = Label(tk,
        text="",
        bg='#fff')
        space1 = Label(tk,
        text="",
        bg='#fff')
        space2 = Label(tk,
        text="",
        bg='#fff')
        space3 = Label(tk,
        text="",
        bg='#fff')
        space4 = Label(tk,
        text="",
        bg='#fff')
        judul = Label(tk,
        text="Applikasi Deteksi Bunga",
        font="Arial 15 bold",
        bd=16,
        relief="groov",
        pady=10)
        mulai = Button(tk,
        text= "Mulai Deteksi",
        font="Verdana 12 bold",
        bg='deep sky blue',
        fg='white',
        width=20,
        command = deteksi)
        tentang = Button(tk,
        text= "Tentang Aplikasi",
        font="Verdana 12 bold",
        bg='deep sky blue',
        fg='white',
        width=20,
        command = tentang_aplikasi)
        bantuan = Button(tk,
        text= "Bantuan",
        font="Verdana 12 bold",
        bg='deep sky blue',
        fg='white',
        width=20,
        command = bantuan)

        keluar = Button(tk,
        text= "Keluar",
        font="Verdana 10 bold",
        bg='red',
        fg='white',
        height=2,
        command = Message_Keluar)

        judul.pack(side=TOP,fill=X)
        space.pack(side=TOP)
        mulai.pack(side=TOP)
        space2.pack(side=TOP)
        tentang.pack(side=TOP)
        space3.pack(side=TOP)
        bantuan.pack(side=TOP)
        space4.pack(side=TOP)
        keluar.pack(side=TOP)
        Kanvas.pack()
        tk.mainloop()
