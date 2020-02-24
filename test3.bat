@echo off

path = %path% C:\Program Files (x86)\Google\Chrome\Application\chrome.exe
start chrome http://DESKTOP-8U6LF35:6006
echo 3 unfixed_with_averagePooling
call activate tensorflow
tensorboard --logdir "C:\Users\Qi\Desktop\final Project -version 5\Graphics3"

