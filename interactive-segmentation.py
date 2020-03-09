import os
import sys
import cv2
import math
import time
import random
import dataset
import network
import pygame
import pygame.camera
import numpy as np
from collections import deque

#store annotated data because it is expensive
datasetfolder = 'default'
if len(sys.argv) > 1:
    datasetfolder = sys.argv[1]

#Size of the image we analyze - small and fast
IMAGE_WIDTH = int(160)
IMAGE_HEIGHT = int(120)

#Displayscale is blown up a bit
SCALE = 8

#Minibatching is nicer than one sample at a time
MINI_BATCH_SIZE = 4

#Make sure window spawn same place always
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,0)

#Pygame is oldscool but gets the job done
pygame.init()

pygame.camera.init()

pygame.font.init()
pygame.mouse.set_visible(0)
myfont = pygame.font.SysFont('Arial', 24)
screen = pygame.display.set_mode((IMAGE_WIDTH*SCALE, IMAGE_HEIGHT*SCALE),  pygame.HWSURFACE|pygame.DOUBLEBUF)
pygame.display.set_caption('Interactive Segmentation')






ds = dataset.Dataset('data/'+datasetfolder)

data_fn = None
data = None

dp_fast_track = deque([])

amp = 1.0

cursor_radius = 1.5 
cursor_pos = (0,0)
drawmousepos = (0,0)

def screen2image(p):
    sw,sh = screen.get_size()
    x = int(IMAGE_WIDTH*p[1]*1.0/ sw)
    y = int(IMAGE_HEIGHT*p[0]*1.0/ sh)
    return (x,y)

def moveto(p):
    global drawmousepos
    drawmousepos = p

def drawto(i,p,c,r):
    x,y = p
    global drawmousepos
    mx,my = drawmousepos
    while mx != x or my != y:
        mx += max(-1.0, min(x-mx, 1.0))
        my += max(-1.0, min(y-my, 1.0))
        cv2.circle(i, (int(my),int(mx)), int(r), c, -1)
    cv2.circle(i, (int(my),int(mx)), int(r), c, -1)
    drawmousepos = (mx,my)


training = True


cam = pygame.camera.Camera("/dev/video2",(IMAGE_WIDTH,IMAGE_HEIGHT))
cam.start()

net = network.Network(IMAGE_HEIGHT, IMAGE_WIDTH, 'data/'+datasetfolder)

lr_index = 1
lr_values = [0.001,0.0005,0.0001, 0.00005,0.00001, 0.000005,0.000001, 0.0000005, 0.0000001]

reg_index = 0
reg_values = [0.0, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001]



mode = 'training'

global_step = 0

running = True

left_down_time = 0.0
right_down_time = 0.0

last_load_time = 0.0

while running:
    keys=pygame.key.get_pressed()
    
    if mode == 'training':
        cursor_color = (255,0,255)
        img = cam.get_image()
        w,h = img.get_size()
        if w != IMAGE_WIDTH or h != IMAGE_HEIGHT:
            img = pygame.transform.scale(img, (IMAGE_WIDTH,IMAGE_HEIGHT))
        liveimage = np.flip(pygame.surfarray.array3d(img).swapaxes(0,1), axis=1)
        target = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_a:
                    if amp == 1.0:
                        amp = 10.0
                    else:
                        amp = 1.0
                if event.key == pygame.K_e:
                    if len(ds.getKeys()):
                        print("switch to training mode")
                        mode = 'editing'
                        data_fn = ds.getKeys()[0]
                        data = ds.get(data_fn)
                    else:
                        print("Cant enter edit mode before we have data to edit.")
                if event.key == pygame.K_t:
                    training = not training
                if event.key == pygame.K_s:
                    print("save model")
                    net.save()
                
                if event.key == pygame.K_b:
                    target = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)*-1
                
                if event.key == pygame.K_r:
                    print("New network")
                    net = network.Network(IMAGE_HEIGHT, IMAGE_WIDTH, 'data/'+datasetfolder)
                
                if event.key == pygame.K_RIGHT:
                    reg_index = min(max(reg_index + 1, 0), len(reg_values)-1)
                if event.key == pygame.K_LEFT:
                    reg_index = min(max(reg_index - 1, 0), len(reg_values)-1)
                if event.key == pygame.K_DOWN:
                    lr_index = min(max(lr_index + 1, 0), len(lr_values)-1)
                if event.key == pygame.K_UP:
                    lr_index = min(max(lr_index - 1, 0), len(lr_values)-1)
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    cursor_radius = min(100.0, cursor_radius +0.5)
                if event.button == 5:
                    cursor_radius = max(0.5, cursor_radius -0.5)

            if event.type == pygame.MOUSEMOTION:
                cursor_pos = screen2image(event.pos)
           
        if pygame.mouse.get_pressed()[0]:
            target = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
            cv2.circle(target, (cursor_pos[1], cursor_pos[0]), int(cursor_radius), (1.0), -1)
            cursor_color = (255,0,0)
        elif pygame.mouse.get_pressed()[2]:
            target = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
            cv2.circle(target, (cursor_pos[1], cursor_pos[0]), int(cursor_radius), (-1.0), -1)
            cursor_color = (0,0,255)
        elif keys[pygame.K_m]:
            target = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
            cv2.circle(target, (cursor_pos[1], cursor_pos[0]), 6, (-1.0), -1)
            cv2.circle(target, (cursor_pos[1], cursor_pos[0]), 2, (1.0), -1)
            cursor_color = (255,255,255)


        if not target is None:
            #new samples are given priority in training before regular training
            dp_fast_track.appendleft({ 'image' : liveimage, 'annotation': target, 'active_pixels' : np.count_nonzero(target)})

        #always train with new data
        #optionally train with rest of dataset
        if training or len(dp_fast_track) > 0:
            datapoints = []
            fns = []
            images = []
            annotations = []
            while len(dp_fast_track) > 0 and len(images) < MINI_BATCH_SIZE:
                dp = dp_fast_track.pop() 
                fns.append(ds.put(dp))
                images.append(dp['image'])
                annotations.append(dp['annotation'])
                datapoints.append(dp)
                
            while len(images) < MINI_BATCH_SIZE and len(ds.getKeys()):
                #fetch random samples from dataset 
                dp, fn = ds.getRandom() 
                fns.append(fn)
                images.append(dp['image'])
                annotations.append(dp['annotation'])
                datapoints.append(dp)

            l = len(datapoints)
                
            if l > 0:
                [_, loss, responses, global_step] = net.train(np.stack(images), np.stack(annotations), lr_values[lr_index], reg_values[reg_index], 1.0)
                    
        #feed single image tensorflow
        [[framebuffer]] = net.evaluate([liveimage], amp)
        
        RFW = net.receptive_field_range*2+1
        margin = 3
        cv2.rectangle(framebuffer, (IMAGE_WIDTH - margin - RFW, margin), (IMAGE_WIDTH-margin, margin+RFW), (255,255,255))
        
        cv2.circle(framebuffer, (cursor_pos[1], cursor_pos[0]), int(cursor_radius), cursor_color, 1)
        
        screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(framebuffer, axes=[1,0,2])),screen.get_size()),(0,0))
        txt = myfont.render('samples:%d pixels: %d tr:%d tc:%0.6d reg:%0.8f lr:%0.8f'%(len(ds.getKeys()), ds.active_pixels, training, global_step, reg_values[reg_index], lr_values[lr_index]), False, (255, 255, 255))
        screen.blit(txt,(5,5))
    
    elif mode == 'editing':
        
        left = False
        right = False
        
        for event in pygame.event.get():    
            if event.type == pygame.QUIT:
                running = False        
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_e:
                    mode = 'training'
                    print("back to training mode")
                if event.key == pygame.K_d:
                    print("deleting datapoint", data_fn)
                    tmp = data_fn
                    data_keys = ds.getKeys()
                    ix = data_keys.index(data_fn)+1
                    if ix >= len(data_keys):
                        ix = 0
                    data_fn  =data_keys[ix]
                    data = ds.get(data_fn)
                    ds.delete(tmp)

                    if data_fn == tmp:
                        print("no more data, going back to training mode")
                        mode = 'training'
                
                if event.key == pygame.K_b:
                    data["annotation"] = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)*-1
               
                if event.key == pygame.K_s:
                    print("save datapoint", data_fn)
                    ds.put(data, data_fn)
                if event.key == pygame.K_LEFT:
                    left = True
                    left_down_time = time.time()
                if event.key == pygame.K_RIGHT:
                    right = True
                    right_down_time = time.time()
                
                        
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in [1,2,3]:
                    moveto(screen2image(event.pos))
                    if event.button == 1:
                        drawto(data["annotation"], screen2image(event.pos), 1.0, cursor_radius)
                    if event.button == 2:
                        drawto(data["annotation"], screen2image(event.pos), 0.0, cursor_radius)
                    if event.button == 3:
                        drawto(data["annotation"], screen2image(event.pos), -1.0, cursor_radius)
                    data["active_pixels"] = np.count_nonzero(data["annotation"])
                if event.button == 4:
                    cursor_radius = min(100.0, cursor_radius +0.5)
                if event.button == 5:
                    cursor_radius = max(0.5, cursor_radius -0.5)
            if event.type == pygame.MOUSEMOTION:
                cursor_pos = screen2image(event.pos)
                l,m,r = event.buttons
                if l:
                    drawto(data["annotation"], screen2image(event.pos), 1.0, cursor_radius)
                elif m:
                    drawto(data["annotation"], screen2image(event.pos), 0.0, cursor_radius)
                elif r:
                    drawto(data["annotation"], screen2image(event.pos), -1.0, cursor_radius)
        
        framebuffer = data["image"].copy()
        if not keys[pygame.K_SPACE]:
            gray = cv2.cvtColor(framebuffer, cv2.COLOR_BGR2GRAY)
            D = 255 * 0.25
            framebuffer[:,:,2] = np.clip(gray + (data["annotation"]*-D), 0, 255).astype(np.uint8)
            framebuffer[:,:,0] = np.clip(gray + (data["annotation"]*D), 0, 255).astype(np.uint8)
            framebuffer[:,:,1] = gray
            cv2.circle(framebuffer, (cursor_pos[1], cursor_pos[0]), int(cursor_radius), (255,0,255), 1)

        if keys[pygame.K_LEFT]:
            if time.time() > left_down_time + 0.5:
                left = True    
        
        if keys[pygame.K_RIGHT]:
            if time.time() > right_down_time + 0.5:
                right = True    
        
        #display last recorded response
        if keys[pygame.K_x] or time.time() < last_load_time + 0.25:
            [[framebuffer]] = net.evaluate([data['image']], amp)

        if left:
            print("edit previous datapoint", data_fn)
            data_keys = ds.getKeys()
            ix = data_keys.index(data_fn)-1
            if ix < 0:
                ix = len(data_keys) - 1
            data_fn  =data_keys[ix]
            ds.load(data_fn)
            data = ds.get(data_fn)
            last_load_time = time.time()
        
        
        if right:
            print("edit next datapoint", data_fn)
            data_keys = ds.getKeys()
            ix = data_keys.index(data_fn)+1
            if ix >= len(data_keys):
                ix = 0
            data_fn =data_keys[ix]
            ds.load(data_fn)
            data = ds.get(data_fn)
            last_load_time = time.time()
        
        screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(framebuffer, axes=[1,0,2])),screen.get_size()),(0,0))
        
        #display raw image
        if not keys[pygame.K_SPACE]:
            y = 5
            dy = 24
            keys = ds.getKeys()
            dirty = '!'
            if data_fn in keys:
                screen.blit(myfont.render('%d/%d %s %s'%(keys.index(data_fn)+1, len(keys), data_fn, dirty), False, (255, 255, 255)),(5,y))
            y += dy

    pygame.display.flip()