import os
import numpy as np

import torch
from torch.utils.data import Dataset
# max_points: tensor([9,273,742]) | min_points: tensor([85,855])

class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=None, test_area=5,
                 sample_rate=1.0, transform=None, shuffle_idx=False):

        super().__init__()
        self.split = split
        self.num_point = num_point
        self.transform = transform
        self.shuffle_idx = shuffle_idx
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        for room_name in rooms_split:
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]
        # print(f"room_idx: {room_idx}| points shape:{ points.shape} | labels shape:{labels.shape}")

        if self.num_point is not None:
            if N_points < self.num_point:
                # simply copy some points
                append_index = np.random.choice(range(N_points), self.num_point-N_points, replace=False)
                selected_points = np.concatenate((points,points[append_index]),axis=0)
                selected_lables = np.concatenate((labels,labels[append_index]),axis=0)
            else:
                selected_index = np.random.choice(range(N_points), self.num_point, replace=False)
                selected_points = points[selected_index,:]
                selected_lables = labels[selected_index]
        else:
            selected_points = points
            selected_lables = labels

        # normalized colors
        normalized_colors = selected_points[:, 3:6] / 255.0
        # normalized points
        normalized_points = selected_points[:, :3] / self.room_coord_max[room_idx]

        # transformation for centered points and normalized colors
        if self.transform is not None:
            normalized_points, normalized_colors = self.transform(normalized_points, normalized_colors)


        selected_points = np.concatenate((normalized_points, normalized_colors), axis=-1)

        if self.shuffle_idx:
            shuffle_idx = np.random.permutation(np.arange(selected_points.shape[0]))
            current_points, current_labels = selected_points[shuffle_idx], selected_lables[shuffle_idx]

        # to Tensor
        selected_points = torch.FloatTensor(selected_points)
        selected_lables = torch.LongTensor(selected_lables)

        return selected_points, selected_lables


    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    import transform
    data_root = 'dataset/s3dis/trainval_fullarea'
    num_point, test_area = 51200, 5

    train_transform = transform.Compose([transform.RandomRotate(along_z=True),
                                         transform.RandomScale(scale_low=0.8, 
                                                               scale_high=1.2),
                                         transform.RandomJitter(sigma=0.01,
                                                                clip=0.05),
                                         transform.RandomDropColor(p=0.8, color_augment=0.0)])
    point_data = S3DIS(split='train', data_root=data_root, num_point=num_point, test_area=test_area, transform=train_transform)
    print('point data size:', point_data.__len__())

    train_loader = torch.utils.data.DataLoader(point_data, batch_size=32, shuffle=True, num_workers=1,
                                               pin_memory=True)
    max_points = 0
    min_points = 9999999999
    tem = torch.rand([3,4])
    from tqdm import tqdm
    for i, (input, target) in tqdm(enumerate(train_loader)):
        tem = input

    print(f"tem shape: {tem.shape}")
    quit(0)


    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 10000 shape:', point_data.__getitem__(1000)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()
