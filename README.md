Here is my little try at computer vision. This uses MediaPipe and OpenCV to track hand 
landmarks in real time through your webcam, then counts how many fingers you're holding up. It 
supports both left and right hands, with different thumb logic for each since thumbs move sideways. 
The model downloads automatically on first run.


To run it:

git clone https://github.com/NurJohnston/HandTrackerCV.git

cd HandTrackerCV

pip install -r requirements.txt

python main.py
