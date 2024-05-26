import pyautogui
import time
import random

simo = ["bello", "intelligente", "attraente", "buono"]
enf = ["bello", "intelligente", "attraente", "buono"]
ludo = ["bella", "intelligente", "attraente", "buona"]

arrays = {
    "simo": simo,
    "enf": enf,
    "ludo": ludo,
}


while True:
    time.sleep(30*60)

    pyautogui.moveTo(700, 300)
    pyautogui.click()

    selected_name, selected_array = random.choice(list(arrays.items()))
    pyautogui.write(f"{selected_name} e' {random.choice(selected_array)}")
    
    time.sleep(30*60)
    pyautogui.moveTo(650, 350)
    pyautogui.click()
    selected_name, selected_array = random.choice(list(arrays.items()))
    pyautogui.write(f"{selected_name} e' {random.choice(selected_array)}")