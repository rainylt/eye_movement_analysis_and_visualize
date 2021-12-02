# eye_movement_analysis_and_visualize
get eye movement event and visualize it
# Install
pull this repositories
```
git clone git@github.com:rainylt/eye_movement_analysis_and_visualize.git
```
install ffmpeg for video saving
```
conda config --add channels conda-forge
conda install ffmpeg
```
install moviepy for video editing
```
pip install moviepy
```
# Usage
change config in `ploter.py` and
```
python ploter.py
```
to get eye movement visualized

or modify the main function in `gaze_analysis.py` to get feature map from eye movement track. 
# Reference
- [eye-movements-predict-personality](https://github.com/sarikayamehmet/eye-movements-predict-personality)
