import pickle
from densepose.structures import DensePoseResult
from recon.recon import estimate_6Dof_pose_with_partiou, get_camera_mat, papare_sim_models, get_color, papare_for_reconstruction, convert_part_uv_to_global_uv, flann_match_keypoints,get_pc_from_depth, get_texture_part_color, convert_part_uv_to_global_uv_sparse, estimate_6Dof_pose, papare_template_vertexs, estimate_6Dof_pose_multiprocess, get_camera_mat,vis_part, append_uv_map
import cv2
import os
import time
from multiprocessing import Process
import numpy as np
from tqdm import tqdm
from recon.image_editing import get_car_name
import argparse
import yaml

def load_yaml_cfg(cfg):
	with open(cfg,'r',encoding='utf-8') as f:
		cfg = f.read()
		d = yaml.load(cfg)
		return d

def vis_part_uv(img_part, img_u, img_v, part_img, u_map, v_map, bbox_xyxy):
    part_mask = np.zeros((img_part.shape[0], img_part.shape[1]))
    part_mask[int(bbox_xyxy[1]):int(bbox_xyxy[1]) + part_img.shape[0], int(bbox_xyxy[0]):int(bbox_xyxy[0]) + part_img.shape[1]] = part_img
    vis_part(part_mask, img_part)

    u_mask = np.zeros((img_part.shape[0], img_part.shape[1]))
    u_mask[int(bbox_xyxy[1]):int(bbox_xyxy[1]) + part_img.shape[0], int(bbox_xyxy[0]):int(bbox_xyxy[0]) + part_img.shape[1]] = u_map
    append_uv_map(u_mask, img_u)

    v_mask = np.zeros((img_part.shape[0], img_part.shape[1]))
    v_mask[int(bbox_xyxy[1]):int(bbox_xyxy[1]) + part_img.shape[0], int(bbox_xyxy[0]):int(bbox_xyxy[0]) + part_img.shape[1]] = v_map
    append_uv_map(v_mask, img_v)

def save_models(best_shape_pc, others_str, instance_id, img_name, save_dir):
    '''
    best_shape_pc (numpy) shape = [3,n]
    others_str [str]
    '''
    with open(os.path.join(save_dir, img_name.split('.')[0] +'_' + str(instance_id) + '.obj'), 'w') as f:
        for i in range(best_shape_pc.shape[1]):
            f.write('v ' + str(best_shape_pc[0][i]) + ' ' + str(best_shape_pc[1][i]) + ' ' + str(best_shape_pc[2][i]) + '\n')
        for other in others_str:
            f.write(other)

def mk_res_dir(cfg):
    if not os.path.exists(cfg['part_seg_output_dir']):
        os.makedirs(cfg['part_seg_output_dir'])
    if not os.path.exists(cfg['uv_reg_output_dir']):
        os.makedirs(cfg['uv_reg_output_dir'])
    if not os.path.exists(cfg['re_render_ouput_dir']):
        os.makedirs(cfg['re_render_ouput_dir'])
    if not os.path.exists(cfg['pos_res_output_dir']):
        os.makedirs(cfg['pos_res_output_dir'])
    if not os.path.exists(cfg['rencon_output_dir']):
        os.makedirs(cfg['rencon_output_dir'])

def sovle_pose(data, test_dir, pcs, car_names, part_bboxes, spcs, face_indexs, t_us, t_vs, others_str, cfg):
    for img_id in tqdm(range(0, len(data))):
        target_num = len(data[img_id]['pred_boxes_XYXY'])
        start = time.time()
        img_name = os.path.basename(data[img_id]['file_name'])
    
        img = cv2.imread(test_dir + img_name)
        img_part = np.copy(img)
        img_u = np.copy(img)
        img_v = np.copy(img)
        # camera_mat = get_camera_mat(img_name)
        for instance_id in range(target_num):
            bbox_xyxy = data[img_id]['pred_boxes_XYXY'][instance_id]
            # pre_class = data[img_id]['pred_classes'][instance_id]
            result_encoded = data[img_id]['pred_densepose'].results[instance_id]
            iuv_arr = DensePoseResult.decode_png_data(*result_encoded)

            # part segmentation, shape = [bbox_hegiht, bbox_width]
            part_img = iuv_arr[0,:,:]
            part_mask = np.zeros((img.shape[0], img.shape[1]))
            part_mask[int(bbox_xyxy[1]):int(bbox_xyxy[1]) + part_img.shape[0], int(bbox_xyxy[0]):int(bbox_xyxy[0]) + part_img.shape[1]] = part_img
            # u_map rescale to [0, 1], shape = [bbox_hegiht, bbox_width]
            u_map = iuv_arr[1,:,:] / 255.0
            # v_map rescale to [0, 1], shape = [bbox_hegiht, bbox_width]
            v_map = iuv_arr[2,:,:] / 255.0

            vis_part_uv(img_part, img_u, img_v, part_img, u_map, v_map, bbox_xyxy)
            
            # for each pix in car instance, convert it to uv(coordinate in texute_map) and uv_in_raw(coordinate in input image)
            uv, uv_in_raw = convert_part_uv_to_global_uv(u_map, v_map, part_img, part_bboxes)
            uv_in_raw[:, 0] += int(bbox_xyxy[0])
            uv_in_raw[:, 1] += int(bbox_xyxy[1])

            # for each pix in car instance, find its corresponding 3D vertex index in template
            sample_count = min(len(uv), 4096)
            indexs = np.arange(len(uv))
            np.random.shuffle(indexs)
            new_uv = []
            new_uv_in_raw = []
            for s in indexs[0:sample_count]:
                new_uv_in_raw.append(uv_in_raw[s])
                new_uv.append(uv[s])
            vertexs_index = flann_match_keypoints(new_uv, target_uv, texture)


            # instance is too small to caculate, so drop it
            if len(new_uv_in_raw) < 10:
                continue
            camera_mat = get_camera_mat(img_name, cfg['calib_dir'])

            
            #　shapes.append(pc)
            img, detection_res, best_shape = estimate_6Dof_pose_with_partiou(vertexs_index, np.array(new_uv_in_raw), img, pcs, car_names, bbox_xyxy, img_name, spcs, face_indexs, t_us, t_vs, part_mask, part_bboxes, camera_mat)
            
            save_models(best_shape, others_str, instance_id, img_name, cfg['rencon_output_dir'])
            [bbox1, bbox2, bbox3, bbox4, height, width, length, x, y, z, b, s] = detection_res
            with open(os.path.join(cfg['pos_res_output_dir'], img_name.split('.')[0] + '.txt'), 'a') as f:
                    f.write('Car -1 -1 -10 ' + str(float(bbox1)) + ' ' + str(float(bbox2)) + ' ' + str(float(bbox3)) + ' ' + str(float(bbox4)) + ' ' + 
                    str(height) + ' ' + str(width) + ' ' + str(length) + ' ' + 
                    str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(b) + ' ' + str(float(s)) + '\n')
            
        cv2.imwrite(os.path.join(cfg['re_render_ouput_dir'], img_name), img)
        cv2.imwrite(os.path.join(cfg['part_seg_output_dir'], img_name.split('.')[0] + '_part.png'), img_part)
        cv2.imwrite(os.path.join(cfg['uv_reg_output_dir'], img_name.split('.')[0] + '_u.png'), img_u)
        cv2.imwrite(os.path.join(cfg['uv_reg_output_dir'], img_name.split('.')[0] + '_v.png'), img_v)

        
parser = argparse.ArgumentParser(description="pose solver")
parser.add_argument('--cfg', default='config.yaml', help='config_file')
args = parser.parse_args()
cfg = load_yaml_cfg(args.cfg)
# part_bboxes: pre-defined part bbox in texture map
# target_uv: croodinate of each template 3d vertex in texture map 
part_bboxes, target_uv, model_face_uv_str = papare_for_reconstruction(cfg['temaplate_models_dir'], cfg['texture_path'])
pcs, car_names = papare_template_vertexs(cfg['temaplate_models_dir'])
spcs, face_indexs, t_us, t_vs = papare_sim_models(cfg['simplification_temaplate_models_dir'])
test_dir = cfg['input_image_dir']
texture = cv2.imread(cfg['texture_path'])

mk_res_dir(cfg)
# load densepose network inferring result
f = open(cfg['stage1_network_res'], 'rb')
data = pickle.load(f)



#　sovle_pose(data, test_dir, pcs, car_names, part_bboxes)

## sovle pose and shape multiprocess
num_of_worker = 1
num_per_worker = len(data) // num_of_worker
if len(data) < num_of_worker:
    num_of_worker = len(data)
    num_per_worker = 1
processes = []
for i in range(num_of_worker):
    if i == num_of_worker - 1:
        p = Process(target=sovle_pose, args=(
        data[i * num_per_worker:], 
        test_dir, 
        pcs, 
        car_names,
        part_bboxes, spcs, face_indexs, t_us, t_vs, model_face_uv_str, cfg))
    else:
        p = Process(target=sovle_pose, args=(
        data[i * num_per_worker:(i + 1) * num_per_worker], 
        test_dir, 
        pcs, 
        car_names,
        part_bboxes, spcs, face_indexs, t_us, t_vs, model_face_uv_str, cfg))
    p.start()
    processes.append(p)
for p in processes:
    p.join()


