# Atari Deep Q-Learning with PyTorch

This project implements a Deep Q-Network (DQN) to train an agent to play Atari's *Ms. Pac-Man* using PyTorch and the Gymnasium environment. The agent uses a convolutional neural network (CNN) to process image frames from the game and decide on actions using reinforcement learning principles.

## Features

- **Deep Q-Network (DQN)**: Implements a DQN with experience replay and a target network to stabilize learning.
- **Ms. Pac-Man Environment**: Uses the Atari *Ms. Pac-Man* environment from Gymnasium to train the agent.
- **Experience Replay**: A deque buffer stores previous experiences for training the network.
- **Frame Preprocessing**: Frames are converted to grayscale, resized, and normalized before being fed into the neural network.
- **CUDA Support**: Leverages GPU computation if available for faster training.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Make sure you have [PyTorch](https://pytorch.org/get-started/locally/) installed with CUDA support if you want to train on a GPU.

4. Install [FFmpeg](https://ffmpeg.org/) to render and save videos.

## Usage

### Training the Agent

To train the agent to play *Ms. Pac-Man*, run the `main.py` script:

```bash
python main.py
```

The agent will train for 2000 episodes, or until the environment is solved with an average score of over 2600. A checkpoint of the trained model will be saved as `checkpoint.pth`.


### Watching the Trained Agent

To watch the agent play using a pre-trained model, run the `show_video.py` script:

```bash
python show_video.py
```

If you have a trained model saved as `checkpoint.pth`, it will load the model and display a video of the agent playing the game.

### Displaying the Video

After running `show_video.py`, a video will be created and displayed using the IPython display module. If the video does not display automatically, you can find it saved as `video.mp4` in the project directory.

## Project Structure

```
├── main.py            # The main script to train the agent
├── show_video.py      # The script to load a trained model and display gameplay
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## Hyperparameters

- **Learning Rate**: `1e-4`
- **Batch Size**: `64`
- **Gamma**: `0.99`
- **Epsilon Decay**: `0.995`
- **Tau**: `1e-3`
- **Number of Episodes**: `2000`
- **Max Timesteps per Episode**: `10000`

## Dependencies

- Python 3.9+
- NumPy
- PyTorch
- Gymnasium
- Pillow
- ImageIO
- IPython

## Sample Output

Below is an example of the output image generated during the training process:



## License

This project is open-source and licensed under the MIT License.
