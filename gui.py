import PySimpleGUI as sg
import numpy as np
from connection import Pressures
import itertools
import cv2

NUMBER = np.full((128), 'E')
NUMBER[36] = 1
NUMBER[93] = 2
NUMBER[109] = 3
NUMBER[46] = 4
NUMBER[107] = 5
NUMBER[123] = 6
NUMBER[37] = 7
NUMBER[127] = 8
NUMBER[47] = 9
NUMBER[119] = 0

dsize = 4

fwidth = 640
fheight = 200

#pytesseract.pytesseract.tesseract_cmd = r'/sbin/tesseract'
pressure = Pressures()

class Number:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.ar = np.zeros((w, h))
        self.art = np.transpose(self.ar)
        self.sizes = np.zeros(7)
        self.coords = []
        self.ssls = []

    def set_digits(self, i, p0, p1, ssl):
        print ('adding digit')
        draw_line(self.ar, p0, p1, i+1)
        self.sizes[i] = np.size(self.ar[self.ar == i+1])
        self.art = np.transpose(self.ar)
        self.coords.append(np.array([p0, p1]))
        self.ssls.append(ssl)
        print(self.coords)

    def check_existence(self, temp, i):
        if self.sizes[i] == 0:
            return 0
        return np.sum(np.sum(temp[self.art == i+1])) / self.sizes[i]

    def get_num(self, temp, out):
        ll = np.zeros(7)
        b = 1
        num = 0
        for i in range(7):
            ll[i] = (b * self.check_existence(temp, i))
            if len(self.ssls) > i and ll[i] < self.ssls[i]:
                out[self.art == i+1] = (0, 255, 0)

        for i, ssl in zip(ll, self.ssls):
            num += (i < ssl) * b
            b *= 2

        return NUMBER[num]

    def calculate_distance(self, v1, v2):
        proj = np.dot(v1, v2)
        if proj < 0:
            return np.linalg.norm(v1)

        if proj > np.dot(v2, v2):
            return np.linalg.norm(v1 - v2)

        proj = proj * v2 / np.dot(v2, v2)
        return np.linalg.norm(v1 - proj)

    def get_closest(self, pos):
        min = 1000000
        index = 0
        for a, b in zip(self.coords, range(7)):
            v1 = pos - a[0]
            v2 = a[1] - a[0]
            dist = self.calculate_distance(v1, v2)
            if dist < min:
                min = dist
                index = b

        return (min, index)


    def draw(self, temp):
        temp[self.art != 0] = (0, 0, 255)

    def draw_index(self, index, temp):
        temp[self.art == index + 1] = (0, 0, 255)

    def get_index_ssl(self, index):
        return self.ssls[index]

    def set_index_ssl(self, index, ssl):
        self.ssls[index] = ssl


def draw_line(temp, p0, p1, color):
    global dsize
    xx1, yy1 = p1
    xx0, yy0 = p0
    if xx0 != xx1:
        if xx0 < xx1:
            x0 = xx0
            y0 = yy0
            x1 = xx1
            y1 = yy1
        else :
            x0 = xx1
            y0 = yy1
            x1 = xx0
            y1 = yy0

        m = (y1-y0)/(x1-x0)
        for x in range(x0, x1):
            y = int(y0 + m*(x-x0))
            temp[dsize + x, dsize + y] = color
    if yy0 != yy1:
        if yy0 < yy1:
            x0 = xx0
            y0 = yy0
            x1 = xx1
            y1 = yy1
        else :
            x0 = xx1
            y0 = yy1
            x1 = xx0
            y1 = yy0
        m2 = (x1-x0)/(y1-y0)
        for y in range(y0, y1):
            x = int(x0 + m2*(y-y0))
            temp[dsize + x, dsize + y] = color

def main():
    global dsize, fwidth, fheight
    draw = False

    output = np.array([[0, 0],
         [fwidth, 0],
         [fwidth, fheight],
         [0, fheight]])

    sg.theme('Black')
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ssl = 100

    layout = [[sg.Text('Demo', size=(40, 1), justification='center')],

              [sg.Graph(canvas_size=(width, height), enable_events=True, graph_bottom_left=(0,0), graph_top_right=(640, 480), key='grph')],
              [sg.Graph(canvas_size=(fwidth, fheight), enable_events=True, graph_bottom_left=(0, 0), graph_top_right=(fwidth, fheight), key='nngrp')],
              [[sg.Button("Add Number", key='numa'), 
                sg.Button("Remove Number", key='numr'),
                sg.Button("Use Tesseract", key='-TESS-USE-'),
                sg.Text('Detected Weight: '),
                sg.Text('', key='output'),
                sg.Text('Detected Pressure: '),
                sg.Text('', key='-pressure-')],
               sg.Slider((0, 255*3), ssl, tick_interval=1, enable_events=True, key='-SL-'),
               sg.Slider((0, 255*3), ssl, tick_interval=1, disabled=True, enable_events=True, key='IS'),
               sg.Button('Done', disabled=True, key='ISButton')]]

    window = sg.Window("My Window", layout)

    dots = [[0,0] for i in range(4)]
    colors = [[256, 0, 0],
              [0, 256, 0],
              [256, 256, 0],
              [0, 256, 256]]
    index = 0

    numbers = []
    adding_number = False
    num_add_info = {'index':0, 'p':0, 'x':None, 'y':None, 'temp': np.zeros((fheight, fwidth, 3))}
    tess_use = False
    final_num = ''

    change_num_info = {'active':False, 'index':0, 'number':None}

    while True:
        final_num = ''

        temp = np.ones([int(height)+2*dsize, int(width)+2*dsize, 3])
        event, values = window.read(timeout=20)
        if event == 'EXIT' or event == sg.WIN_CLOSED:
            return

        _, imen = cap.read()

        if event == 'grph':
            x, y = values[event]
            y = height - y
            dots[index] = [y, x]
            if index + 1 == 4:
                draw = True
            index = (index +1) %4

        if event == '-SL-':
            ssl = values[event]

        if event == 'numa' and adding_number == False:
            window['IS'].update(disabled=True)
            window['ISButton'].update(disabled=True)
            change_num_info['active'] = False
            num_add_info['index']= 0
            numbers.append(Number(fwidth, fheight))
            adding_number = True

        if event == 'numr' and adding_number == False and change_num_info['active'] == False:
            if len(numbers) > 1:
                numbers = numbers[:-1]

        if event == '-TESS-USE-':
            tess_use = not tess_use
            print (tess_use)
            if tess_use:
                window['-TESS-USE-'].update('Use 7 Segment')
            else:
                window['-TESS-USE-'].update('Use Tesseract')

        if event == 'ISButton':
            change_num_info['active'] = False
            window['IS'].update(disabled=True)
            window['ISButton'].update(disabled=True)

        if event == 'IS':
            change_num_info['number'].set_index_ssl(
                    change_num_info['index'],
                    values[event])
        
        for i in range(4):
            temp[dots[i][0]:dots[i][0]+2*dsize,dots[i][1]:dots[i][1]+2*dsize] = colors[i]

        for i in range(4):
            b = (i+1)%4
            draw_line(temp, dots[i], dots[b], [0, 0, 255])
        
        if draw:
            dds = [[b, a] for a, b in dots]
            T = cv2.getPerspectiveTransform(np.array(np.reshape(dds, (4, 2)), np.float32), np.array(output, np.float32))
            imentrans = cv2.warpPerspective(imen, T, (fwidth, fheight))

            if not tess_use:
                if change_num_info['active']:
                    change_num_info['number'].draw_index(
                            change_num_info['index'], num_add_info['temp'])

                for n in numbers:
                    final_num = str(n.get_num(imentrans, num_add_info['temp'])) + final_num

                window['output'].update(final_num)
                
                imentrans = num_add_info['temp'] + imentrans

                if adding_number:
                    imentrans[imentrans>255] = 255

            else:
                # imentrans[imentrans > ssl] = 255
                # imentrans[imentrans <= ssl] = 0
                #
                # imentrans = np.max(imentrans, 2)
                # #imentrans = cv2.cvtColor(imentrans, cv2.COLOR_BGR2GRAY)
                # # Use Pytesseract to extract text from the frame
                # detected_text = pytesseract.image_to_string(imentrans)
                pass
                # # Print the detected text
                # print("Detected Text:", detected_text)
                #


            mmb = cv2.imencode('.png', imentrans)[1].tobytes()

            window['nngrp'].erase()
            window['nngrp'].draw_image(data=mmb, location=(0, fheight))



        num_add_info['temp'] = np.zeros((fheight, fwidth, 3))
        if adding_number:
            numbers[len(numbers)-1].draw(num_add_info['temp'])

        if event == 'nngrp' and adding_number == True:
            (x, y) = values[event]
            y = fheight - y
            if num_add_info['p'] == 0:
                num_add_info['p'] = 1
                num_add_info['x'] = x
                num_add_info['y'] = y
            else:
                num_add_info['p'] = 0
                p0 = (num_add_info['x'], num_add_info['y'])
                p1 = (x, y)
                numbers[len(numbers)-1].set_digits(index, p0, p1, ssl)
                index += 1
                if index >= 7:
                    adding_number = False
                    index = 0
        elif event == 'nngrp' and len(numbers) >= 1:
            min = fwidth
            x, y = values[event]
            print(x, y)
            for num in numbers:
                print(num)
                dist, index = num.get_closest([x, fheight-y])
                print(dist)
                if dist < min:
                    min = dist
                    change_num_info['index'] = index
                    change_num_info['number'] = num

            change_num_info['active'] = True
            window['IS'].update(disabled=False)
            window['ISButton'].update(disabled=False)
            window['IS'].update(value = change_num_info['number'].get_index_ssl(
                change_num_info['index']))

            
        
        imen = temp[dsize:-dsize,dsize:-dsize] * imen
        imen[imen > 255] = 255
        background = cv2.imencode('.png', imen)[1].tobytes()

        window['grph'].erase()
        window['grph'].draw_image(data=background, location=(0, height))

        window['-pressure-'].update(pressure.get_pressure())

        background = []
        imen = []
        mmb = []
        imentrans = []
main()
