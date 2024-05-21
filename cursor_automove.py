import pyautogui
import time

while True:
    pyautogui.moveTo(700, 300)
    pyautogui.click()
    pyautogui.write('Hello, world!')
    
    time.sleep(30*60)

    pyautogui.moveTo(650, 350)
    pyautogui.click()
    pyautogui.write('How are you?')
    
    time.sleep(30*60)