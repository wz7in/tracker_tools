import io, zipfile, json, os, pickle
import numpy as np
from flask import Flask, request, send_file
import multiprocessing
import portalocker

app = Flask(__name__)
model_sam, model_cotracker = None, None
ROOT_DIR = '/mnt/hwfile/OpenRobotLab/Annotation4Manipulation'

def get_sam_history(user_name, time):
    user_config_path = os.path.join(ROOT_DIR, 'user_config', 'sam', f"{user_name}{time}.txt")
    if not os.path.exists(user_config_path):
        history = []
    else:
        with open(user_config_path, 'r') as f:
            history = f.readlines()
    return history

def get_diff(a, b):
    a_dict = {i.split('/')[-1].strip(): i for i in a}
    a_set = set(list(a_dict.keys()))
    b_set = set([i.split('/')[-1].strip() for i in b])
    diff = list(a_set.difference(b_set))
    return [a_dict[i] for i in diff]

def get_available(a, b):
    res = []
    a_dict = {i.split('/')[-1].strip(): i for i in a}
    b = set([i.split('/')[-1].strip() for i in b])
    res = [a_dict[i] for i in b if i in a_dict]
    return res

@app.route("/get_video_and_anno_lang", methods=["POST"])
def get_video_and_anno_lang():
    
    config = json.loads(request.data)
    user_name = config['username']
    mode = config['mode']
    last_video_path = config['last_video_path']
    
    with open(os.path.join(ROOT_DIR, 'no_annotation_lang.json'), 'r+') as f1:
        portalocker.lock(f1, portalocker.LOCK_EX)
        no_annotation = json.load(f1)
        with open(os.path.join(ROOT_DIR, 'has_annotation_lang.json'), 'r+') as f2:
            portalocker.lock(f2, portalocker.LOCK_EX)
            has_annotation = json.load(f2)
            if len(no_annotation) == 0:
                is_finished = True
            else:
                is_finished = False
                
            user_config_path = os.path.join(ROOT_DIR, 'user_config', 'lang', f"{user_name}.txt")
            if not os.path.exists(user_config_path):
                history = []
            else:
                with open(user_config_path, 'r') as f:
                    history = f.readlines()
            if mode == 'pre':
                video_path = history[-1].strip()
                no_annotation[last_video_path] = has_annotation[last_video_path].copy()
                del has_annotation[last_video_path]
                is_finished = False
            else:
                video_path = list(no_annotation.keys())[0]
                has_annotation[video_path] = no_annotation[video_path].copy()
                del no_annotation[video_path]
            f2.seek(0)
            f2.truncate()
            json.dump(has_annotation, f2)
            portalocker.unlock(f2)
        
        f1.seek(0)
        f1.truncate()
        json.dump(no_annotation, f1)
        portalocker.unlock(f1)
    
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, "w") as zf:
        if not is_finished:
            with zf.open("video.mp4", "w") as f:
                with open(video_path, "rb") as video_file:
                    f.write(video_file.read())
            npz_io = io.BytesIO()
            anno_file = np.load(has_annotation[video_path]['anno_path'], allow_pickle=True)
            np.savez_compressed(npz_io, anno_file=anno_file['data'])
            npz_io.seek(0)
            zf.writestr("anno.npz", npz_io.getvalue())
            save_path = has_annotation[video_path]['save_path'].rsplit('/', 1)[0]
            save_file_name = has_annotation[video_path]['save_path'].split('/')[-1].split('.')[0]
            save_path = os.path.join(save_path, save_file_name)
            zf.writestr("save_path", save_path)
            zf.writestr("video_path", video_path)
            zf.writestr("history_number", str(len(history)))
        zf.writestr("is_finished", str(is_finished))
            
    zip_io.seek(0)
    return send_file(
        zip_io,
        mimetype="application/zip",
        as_attachment=True,
        download_name="video_and_anno_lang.zip",
    )

@app.route("/get_video_and_anno_sam", methods=["POST"])
def get_video_and_anno_sam():
    config = json.loads(request.data)
    user_name = config['username']
    mode = config['mode']
    re_anno = int(config['re_anno'])
    last_video_path = config['last_video_path']
    history_finished = get_sam_history(user_name, '_finish')
    history = get_sam_history(user_name, '')
    history = get_diff(history, history_finished)
    
    history_1 = get_sam_history(user_name, '_1')
    history_1 = get_diff(history_1, history_finished)
    usable_1 = get_diff(history, history_1)
    all_one_anno_num = len(usable_1)
    
    history_2 = get_sam_history(user_name, '_2')
    history_2 = get_diff(history_2, history_finished)
    usable_2 = get_diff(history_1, history_2)
    all_two_anno_num = len(usable_2)
    
    history_3 = get_sam_history(user_name, '_3')
    usable_3 = get_diff(history_2, history_3)
    all_three_anno_num = len(usable_3)
    
    with open(os.path.join(ROOT_DIR, f'no_annotation_sam.json'), 'r+') as f1:
        portalocker.lock(f1, portalocker.LOCK_EX)
        no_annotation = json.load(f1)
        with open(os.path.join(ROOT_DIR, f'has_annotation_sam.json'), 'r+') as f2:
            portalocker.lock(f2, portalocker.LOCK_EX)
            has_annotation = json.load(f2)
            if len(no_annotation) == 0:
                is_finished = True
            else:
                is_finished = False
            
            available_0 = [i for i in no_annotation.keys() if 'ann_human' not in i]
            available_1 = get_available(list(no_annotation.keys()), usable_1)
            available_2 = get_available(list(no_annotation.keys()), usable_2)
            available_3 = get_available(list(no_annotation.keys()), usable_3)
            
            if re_anno == 1 and len(available_1) == 0:
                is_finished = True
            
            if re_anno == 2 and len(available_2) == 0:
                is_finished = True
            
            if re_anno == 3 and len(available_3) == 0:
                is_finished = True
            
            if re_anno > 0 and not is_finished:
                if re_anno == 1:
                    video_path = available_1[0]
                elif re_anno == 2:
                    video_path = available_2[0]
                elif re_anno == 3:
                    video_path = available_3[0]
                assert mode == 'next'
                has_annotation[video_path] = no_annotation[video_path].copy()
                del no_annotation[video_path]
            else:
                if mode == 'pre':
                    video_path = history[-1].strip()
                    no_annotation[last_video_path] = has_annotation[last_video_path].copy()
                    del has_annotation[last_video_path]
                    is_finished = False
                elif not is_finished: 
                    video_path = available_0[0]
                    has_annotation[video_path] = no_annotation[video_path].copy()
                    del no_annotation[video_path]
            
            f2.seek(0)
            f2.truncate()
            json.dump(has_annotation, f2)
            portalocker.unlock(f2)
        
        f1.seek(0)
        f1.truncate()
        json.dump(no_annotation, f1)
        portalocker.unlock(f1)
        
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, "w") as zf:
        if not is_finished:
            with zf.open("video.mp4", "w") as f:
                with open(video_path, "rb") as video_file:
                    f.write(video_file.read())
            # send save path
            save_path = has_annotation[video_path]['save_path'].rsplit('/', 1)[0]
            save_file_name = has_annotation[video_path]['save_path'].split('/')[-1].split('.')[0]
            save_path = os.path.join(save_path, save_file_name)
            zf.writestr("save_path", save_path)
            zf.writestr("video_path", video_path)
            zf.writestr("history_number", str(len(history)))
        
        zf.writestr("is_finished", str(is_finished))
        zf.writestr("all_one_anno_num", str(all_one_anno_num))
        zf.writestr("one_anno_num", str(len(available_1)))
        zf.writestr("all_two_anno_num", str(all_two_anno_num))
        zf.writestr("two_anno_num", str(len(available_2)))
        zf.writestr("all_three_anno_num", str(all_three_anno_num))
        zf.writestr("three_anno_num", str(len(available_3)))
    
    zip_io.seek(0)
    return send_file(
        zip_io,
        mimetype="application/zip",
        as_attachment=True,
        download_name="video_and_anno_sam.zip",
    )

@app.route("/save_anno", methods=["POST"])
def save_anno():
    file = request.files.get('file')
    file_content = file.read()
    with np.load(io.BytesIO(file_content), allow_pickle=True) as data:
        anno = data['anno_file'].item()
    save_path = request.form.get('save_path')
    user_name = anno['user']
    video_path = anno['video_path']
    
    if '/0/' in video_path:
        time = '_1'
    elif '/1/' in video_path:
        time = '_2'
    elif '/2/' in video_path:
        time = '_3'
    else:
        time = ''
    
    np.savez(save_path, pickle.dumps(anno))
    if 'sam' in save_path.split('/'):
        mode = 'sam'
    else:
        mode = 'lang'
    save_dir = os.path.join(ROOT_DIR, 'user_config', mode)
    
    if not os.path.exists(os.path.join(save_dir, f'{user_name}{time}.txt')):
        history = []
    else:
        with open(os.path.join(save_dir, f'{user_name}{time}.txt'), 'r') as f:
            history = f.readlines()
    
    if len(history) > 0 and video_path == history[-1].strip():
        history = history[:-1]
    
    history.append(video_path+'\n')
    with open(os.path.join(save_dir, f'{user_name}{time}.txt'), 'w') as f:
        f.writelines(history)
    
    if anno['is_finished']:
        if not os.path.exists(os.path.join(save_dir, f'{user_name}_finish.txt')):
            # create finish file
            with open(os.path.join(save_dir, f'{user_name}_finish.txt'), 'w') as f:
                f.writelines([video_path.strip()+'\n'])
        else:
            with open(os.path.join(ROOT_DIR, 'user_config', 'sam', f"{user_name}_finish.txt"), 'r+') as f:
                is_finished = f.readlines()
                is_finished.append(video_path.strip()+'\n')
                f.seek(0)
                f.truncate()
                f.writelines(is_finished)
        
        
    # with open(os.path.join(ROOT_DIR, f'no_annotation_{mode}.json'), 'w') as f1:
    #     portalocker.lock(f1, portalocker.LOCK_EX)
    #     no_annotation = json.load(f)
    #     with open(os.path.join(ROOT_DIR, f'has_annotation_{mode}.json'), 'w') as f2:
    #         portalocker.lock(f2, portalocker.LOCK_EX)
    #         has_annotation = json.load(f)
    #         if button_mode == 'pre':
    #             has_annotation[video_path] = no_annotation[video_path].copy()
    #             del no_annotation[video_path]
            
            
    #         portalocker.unlock(f2)
    #     portalocker.unlock(f1)
    # if video_path in no_annotation:
    #     has_annotation[video_path] = no_annotation[video_path].copy()
    #     del no_annotation[video_path]
    
    #     with open(os.path.join(ROOT_DIR, f'no_annotation_{mode}.json'), 'w') as f:
    #         portalocker.lock(f, portalocker.LOCK_EX)
    #         json.dump(no_annotation, f)
    #         portalocker.unlock(f)
        
    #     with open(os.path.join(ROOT_DIR, f'has_annotation_{mode}.json'), 'w') as f:
    #         portalocker.lock(f, portalocker.LOCK_EX)
    #         json.dump(has_annotation, f)
    #         portalocker.unlock(f)
    # assert video_path not in no_annotation
    
    return "success"
        
@app.route("/drawback_video_sam", methods=["POST"])
def drawback_video_sam():
    config = json.loads(request.data)
    video_path = config['video_path']
    with open(os.path.join(ROOT_DIR, 'no_annotation_sam.json'), 'r+') as f1:
        portalocker.lock(f1, portalocker.LOCK_EX)
        no_annotation = json.load(f1)
        with open(os.path.join(ROOT_DIR, 'has_annotation_sam.json'), 'r+') as f2:
            portalocker.lock(f2, portalocker.LOCK_EX)
            has_annotation = json.load(f2)
            no_annotation[video_path] = has_annotation[video_path].copy()
            del has_annotation[video_path]
            f2.seek(0)
            f2.truncate()
            json.dump(has_annotation, f2)
            portalocker.unlock(f2)
        
        f1.seek(0)
        f1.truncate()
        json.dump(no_annotation, f1)
        portalocker.unlock(f1)
    
    return "success"

@app.route("/drawback_video_lang", methods=["POST"])
def drawback_video_lang():
    config = json.loads(request.data)
    video_path = config['video_path']
    with open(os.path.join(ROOT_DIR, 'no_annotation_lang.json'), 'r+') as f1:
        portalocker.lock(f1, portalocker.LOCK_EX)
        no_annotation = json.load(f1)
        with open(os.path.join(ROOT_DIR, 'has_annotation_lang.json'), 'r+') as f2:
            portalocker.lock(f2, portalocker.LOCK_EX)
            has_annotation = json.load(f2)
            no_annotation[video_path] = has_annotation[video_path].copy()
            del has_annotation[video_path]
            f2.seek(0)
            f2.truncate()
            json.dump(has_annotation, f2)
            portalocker.unlock(f2)
        
        f1.seek(0)
        f1.truncate()
        json.dump(no_annotation, f1)
        portalocker.unlock(f1)
    
    return "success"
    
def run_on_port(port):
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":

    # with open("./config/config.yaml") as f:
    #     model_config = yaml.load(f, Loader=yaml.FullLoader)

    # sam_config = model_config["sam"]
    # co_tracker_config = model_config["cotracker"]
    
    # model_sam = Sam(
    #     sam_config["sam_ckpt_path"],
    #     sam_config["model_config"],
    #     sam_config["threshold"],
    #     False,
    #     sam_config["device"],
    # )

    # model_cotracker = CoTrackerPredictor(
    #     checkpoint=co_tracker_config["cotracker_ckpt_path"]
    # )
    process_list = []
    process_number = 10
    for i in range(process_number):
        process = multiprocessing.Process(target=run_on_port, args=(10050+i,))
        process_list.append(process)
        process.start()
    
    for process in process_list:
        process.join()
    # run_on_port(10050)