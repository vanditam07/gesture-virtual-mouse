import pyautogui


class Mouse:
    def __init__(self):
        self.tx_old = 0
        self.ty_old = 0
        self.trial = True
        self.flag = 0

    def move_mouse(self, frame, position, gesture):
        (sx, sy) = pyautogui.size()
        (camx, camy) = (frame.shape[:2][0], frame.shape[:2][1])
        (mx_old, my_old) = pyautogui.position()

        Damping = 2
        tx = position[0]
        ty = position[1]
        if self.trial:
            self.trial, self.tx_old, self.ty_old = False, tx, ty

        delta_tx = tx - self.tx_old
        delta_ty = ty - self.ty_old
        self.tx_old, self.ty_old = tx, ty

        if gesture == 3:
            self.flag = 0
            mx = mx_old + (delta_tx * sx) // (camx * Damping)
            my = my_old + (delta_ty * sy) // (camy * Damping)
            pyautogui.moveTo(mx, my, duration=0.1)
        elif gesture == 0:
            if self.flag == 0:
                pyautogui.doubleClick()
                self.flag = 1
        elif gesture == 1:
            print("1 Finger Open")

