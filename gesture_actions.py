import time
import pygetwindow as gw
import pyautogui
import keyboard  

def get_active_app():
    win = gw.getActiveWindow()
    return win.title if win else ""

def trigger_action(label):
    app = get_active_app().lower()

    if "powerpoint" in app: # PowerPoint-specific actions
        if label == "ThumbsUp":
            keyboard.send("f5")       # Start presentation
        elif label == "ThumbsDown":
            keyboard.send("esc")      # Exit presentation
        elif label == "OpenPalm":
            keyboard.send("b")        # Black screen toggle
        elif label == "Peace":
            keyboard.send("right")    # Next slide
        elif label == "Fist":
            keyboard.send("left")     # Previous slide

    elif "media player" in app or "vlc" in app: # Media player actions
        if label == "ThumbsUp":
            keyboard.send("volume up") # Volume Up
        elif label == "ThumbsDown":
            keyboard.send("volume down") # Volume Down
        elif label == "OpenPalm":
            keyboard.send("space")    # Play/Pause
        elif label == "Peace":
            keyboard.send("ctrl+right")  # Forward 30s
        elif label == "Fist":
            keyboard.send("ctrl+left")   # Backward 30s
    else:
        print(f"No mapped action for {label} in app: {app}")

    time.sleep(0.3) # Delay between actions
