# Anime character extractor
Extract anime characters from anime video files, so they can later be used for training LoRA or other tasks
# Usage
`python main.py -i K:\anime\ReincarnatedPrincess -d characters\ReincarnatedPrincess.booru -o output\ReincarnatedPrincess\`
# Known issues
1. Very poor performance
2. <s>Character descriptor language does not handle "not" yet</s>
# Highlights
1. Custom language to define characters using tags and remove/append tags for better LoRA tagging
2. Use ensemble of danbooru taggers for better tagging
3. Use TensorRT for better performance
# TODO
1. <s>Use TransNetV2 to segment video</s>, for each shot we run multiple CascadeClassifier in parallel and yield a single frame with faces as far away from the edge as possible
2. Use character detector instead of face detector
3. <s>Add tag filtering in object descriptor file to remove incorrect tags from generated txt file</s>
4. Speed up, but how?
5. Merge multiple frames into a single frame if their optical flow all point to the same direction
6. Use NvDec instead of CPU ffmpeg
# Credits
```
https://github.com/hysts/anime-face-detector
https://github.com/nagadomi/lbpcascade_animeface
https://github.com/AUTOMATIC1111/TorchDeepDanbooru
https://github.com/soCzech/TransNetV2
https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags
```
