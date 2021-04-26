from .Layout import *


class MainWindow(MainWindowLayOut):

    def __init__(self, opt):
        super(MainWindow, self).__init__(opt)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(50)

    def update_image(self, pt='face.jpg'):
        try:
            _, frame = self.vs.read()
            if frame is None:
                return
            frame = imutils.resize(frame, height=800)
            isChecked = {
                'region': self.region_checkbtn.isChecked()
                #'wall': self.wall_checkbtn.isChecked()
            }
            packet = self.processor.getProcessedImage(frame, isChecked, region_bbox2draw=self.region_bbox2draw)
            faces_recorded = packet['faces']
            id_num = packet['num']
            use_time = packet['time']
            danger_num = packet['danger']
            long_num = packet['long']
            if len(faces_recorded) > 0:
                data = []
                for im, faceID, behavior in faces_recorded:
                    license_number = '{:04}'.format(faceID)
                    if faceID in self.processor.processor.illegals:
                        illegal = '危险行为-'
                        self.wav_player.play()
                    elif faceID in self.processor.processor.longstay:
                        illegal = 'douliu'
                    else:
                        illegal = '正常行为'
                    time_ = str(time.ctime())
                    location = '未知'

                    cv2.imwrite('./data/'+pt, im)
                    value = {'CARID': faceID,
                            'CARIMAGE': QPixmap('./data/'+pt),
                            'CARCOLOR': time_,
                            'LICENSEIMAGE': None,
                            'LICENSENUMBER': license_number,
                            'LOCATION': location,
                            'RULENAME': illegal}
                    data.append(value)
                self.updateLog(data)
            
            qimg0 = self.toQImage(packet['frame'], height=None)
            self.live_preview.setPixmap(QPixmap.fromImage(qimg0))
            self.label_4.setText(str(id_num))
            self.label_4.setStyleSheet("color:blue")
            self.label_6.setText(str(use_time))
            self.label_8.setText(str(danger_num))
            self.label_8.setStyleSheet("color:red")
            self.label_13.setText(str(long_num))
            self.label_13.setStyleSheet("color:red")
        except Exception as e:
            print('-[ERROR]', e)
