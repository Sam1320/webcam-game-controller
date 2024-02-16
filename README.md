# Webcam Game Controller

This is a simple system that uses a webcam and a *very small* neural network to control a game in realtime. No pre-trained models or any other data is used. All the design, data collection, data processing, and training is done from scratch.




This project was developed as a part of a challenge. The final [project report](notebooks/project_report.ipynb) summarizes the full process of developing the system. The final result can be seen in the [demo video](webcam_game_controller/demo.mp4).

## Running The System

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## System Overview In A Nutshell

### main.py
Load the model's weights: 
```python
model = CNNv2()
model.load_state_dict(
    torch.load(os.path.join(env.models_path, 'cnnv2_3.pt'))
)
```
and run the game loop.
```python
run_system(model=model)
```

### Inside run_system

Define the mappings between the neural network's output and the game's controls. 
```python
action_mappings = {'left': 0, 'right': 2, 'wait': 1}
```
Setup the game environemnt to the initial state, create the VideoCapture object to capture the webcam's feed (device 0), and set the font for the text that will be displayed on the screen. 
```python
env = gym.make('MountainCar-v0', render_mode='human')
env.reset()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
```


Set the neural network to evaluation mode so that it doesn't update its weights or keep track of gradients. 
```python
model.eval()
```
While the user hasn't quit the game.
```python
while True and cap.isOpened():
```
Capture the webcam's feed, process it, and get the neural network's output.  
```python
ret, frame = cap.read()
im = np.array(frame)
im_processed, prediction = model_predict(model, im, raw=True)
```
Then, resize the processed image and render it to the screen, also render the original image with the model's prediction overlayed.
```python
im_processed = cv2.resize(im_processed, (im.shape[1], im.shape[0]))
cv2.putText(im, prediction, (250, 250), font, 1, (0, 255, 0), 3)
cv2.imshow('original', im)
cv2.imshow('processed', im_processed)
```



