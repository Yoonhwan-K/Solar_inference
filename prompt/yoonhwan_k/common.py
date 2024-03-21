import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)

def get_available_memory() :
	
	gpu_count = torch.cuda.device_count()	
	gpu_mem_info = {}
	gpu_device_info = {}

	for gpu_id in range(gpu_count) :
		gpu_mem_info[gpu_id] = {'total' : 0,'available' : 0,'used' : 0}
		gpu_device_info[gpu_id] = torch.cuda.get_device_properties(gpu_id)
	
	for gpu_id in range(gpu_count) :
		torch.cuda.set_device(gpu_id)
		(available, max) = torch.cuda.mem_get_info()
		used = max - available
		gpu_mem_info[gpu_id]['total'] = max/1024/1024
		gpu_mem_info[gpu_id]['available'] = available/1024/1024
		gpu_mem_info[gpu_id]['used'] = used/1024/1024

	return (gpu_count, gpu_mem_info)

def get_max_available_mem_device() :
	
	gpu_cnt,mem_info = get_available_memory()
	return_gpu_id = 0
	return_mem_available = 0
	for gpu_id in range(gpu_cnt):
		if mem_info[gpu_id]['available'] > return_mem_available :
			return_gpu_id = gpu_id
			return_mem_available = mem_info[gpu_id]['available']

	return (return_gpu_id, return_mem_available)

def set_service_gpu() :
	
	gpus = []

	target_gpu, available_memory = get_max_available_mem_device()
	
	gpus.append(target_gpu)
	
	global target_device
	
	if len(gpus) > 0 :
		target_device = 'cuda:{}'.format(gpus[0])
	else:
		target_device = 'cpu'

	return target_device
