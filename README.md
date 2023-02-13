# Anime character extractor
Extract anime characters from anime video files, so they can later be used for texture inversion or other tasks
# Usage
`python main.py -i K:\anime\ReincarnatedPrincess -d characters\ReincarnatedPrincess.booru -o output\ReincarnatedPrincess\`
# TODO
1. Use TransNetV2 to segment video, for each shot we run multiple CascadeClassifier in parallel and yield a single frame with faces as far away from the edge as possible
2. Use character detector instead of face detector
3. Add tag filtering in object descriptor file to remove incorrect tags from generated txt file
4. Speed up, but how?
5. Merge multiple frames into a single frame if their optical flow all point to the same direction
# Credits
```
https://github.com/hysts/anime-face-detector
https://github.com/nagadomi/lbpcascade_animeface
```
