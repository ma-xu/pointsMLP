import os
import numpy as np

import torch
from torch.utils.data import Dataset
# max_points: tensor([9,273,742]) | min_points: tensor([85,855])

class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5,
                 block_size=1.0, sample_rate=1.0, transform=None, fea_dim=6, shuffle_idx=False):

        super().__init__()
        self.split = split
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.fea_dim = fea_dim
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
        return N_points
        """
        selected_points = points[selected_point_idxs, :]  # num_point * 6

        # normalized colors
        normalized_colors = selected_points[:, 3:6] / 255.0
        # normalized points
        normalized_points = selected_points[:, :3] / self.room_coord_max[room_idx]

        # transformation for centered points and normalized colors
        if self.transform is not None:
            centered_points, normalized_colors = self.transform(centered_points, normalized_colors)

        # current points and current labels
        if self.fea_dim == 3:
            current_points = np.concatenate((centered_points, normalized_points), axis=-1)
        elif self.fea_dim == 6:
            current_points = np.concatenate((centered_points, normalized_colors, normalized_points), axis=-1)
        else:
            raise ValueError('Feature dim {} not supported.'.format(self.fea_dim))
        current_labels = labels[selected_point_idxs]

        if self.shuffle_idx:
            shuffle_idx = np.random.permutation(np.arange(current_points.shape[0]))
            current_points, current_labels = current_points[shuffle_idx], current_labels[shuffle_idx]
        
        # to Tensor
        current_points = torch.FloatTensor(current_points)
        current_labels = torch.LongTensor(current_labels)
 
        return current_points, current_labels
        """

    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    import transform
    data_root = 'dataset/s3dis/trainval_fullarea'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 1.0

    train_transform = transform.Compose([transform.RandomRotate(along_z=True),
                                         transform.RandomScale(scale_low=0.8, 
                                                               scale_high=1.2),
                                         transform.RandomJitter(sigma=0.01,
                                                                clip=0.05),
                                         transform.RandomDropColor(p=0.8, color_augment=0.0)])
    point_data = S3DIS(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=train_transform)
    print('point data size:', point_data.__len__())
    print(f"room points: {max(point_data.room_points)}-{min(point_data.room_points)}")
    print(f"room ranges: {point_data.room_coord_min}-{point_data.room_coord_max}")

    train_loader = torch.utils.data.DataLoader(point_data, batch_size=1, shuffle=True, num_workers=1,
                                               pin_memory=True)
    max_points = 0
    min_points = 9999999999
    from tqdm import tqdm
    for i, (n_points) in tqdm(enumerate(train_loader)):
        if n_points > max_points:
            max_points = n_points
        if n_points < min_points:
            min_points = n_points

    print(f"max_points: {max_points} | min_points: {min_points}")
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
