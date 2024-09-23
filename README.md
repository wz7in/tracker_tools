# A Simple Tools for Tracking any points in a video or Tracking any objects in a video by mask

## Installation
```bash
conda create -n tracker python=3.11
conda activate tracker
pip install -r requirements.txt
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118 # for cuda 11.1
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 # for cpu
mkdir checkpoints
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
cd ..
git clone https://github.com/facebookresearch/co-tracker
cd co-tracker
pip install -e .
wget -P checkpoints https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth
```

## Annotation Files
```bash
prepare a file named 'lang_config.json' in folder data, which contains the following information:
{
    'video_path':
    {
        'instruction': 'global instruction',
        'instructionC': 'global instruction in Chinese',
        'task_steps': 'clip description options',
        'task_stepsC': 'clip description options in Chinese',
        'action_steps': 'primtive action description options',
        'action_stepsC' : 'primtive action description options in Chinese'
   }
}
```

## Usage
```bash
python gui.py
```