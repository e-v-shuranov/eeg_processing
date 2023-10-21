
# ----------------------------------------Data-----------------------------------------------------
class TEST_TUH(torch.utils.data.Dataset):
    def __init__(self, path): #, tuh_filtered_stat_vals):
        super(TEST_TUH, self).__init__()
        self.main_path = path
        self.paths = path
        # self.tuh_filtered_stat_vals = tuh_filtered_stat_vals
        # self.paths = ['{}/{}'.format(self.main_path, i) for i in os.listdir(self.main_path)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx, negative=False):
        path = self.paths[idx]
        # take 60s of recording with specified shift
        key = False
        while (key == False):
            try:
                # sample = np.load(path, allow_pickle=True).item()['value']
                sample = np.load(path, allow_pickle=True).item()
                key = True
            except Exception as e:
                print("Path: {} is broken ".format(path), e)
                path = np.random.choice(self.paths, 1)[0]
                # sample = np.load(path, allow_pickle=True).item()['value']
        real_len = min(3000, sample['value_pure'].shape[0])
        # if np.random.choice([0, 1], p=[0.9, 0.1]):
        #     real_len = np.random.randint(real_len // 2, real_len)

        # sample = torch.from_numpy(sample[:6000].astype(np.float32)).clone()
        # channels_ids = [i for i, val in enumerate(mitsar_chls) if val not in ['FCZ', 'PZ']]
        # channels_ids = [i for i, val in enumerate(sample['channels']) if i != 3 and val in mitsar_chls]
        channels_ids = [i for i, val in enumerate(sample['channels']) if val in mitsar_chls]

        sample = sample['value_pure'][:real_len]

        # choose 2 random channels
        channels_to_train = channels_ids  # np.random.choice(channels_ids, 2, replace=False)
        channels_vector = torch.tensor((channels_to_train))
        sample = sample[:, channels_to_train]

        sample_norm = sample
        # sample_norm = (sample - self.tuh_filtered_stat_vals['mean_vals_filtered'][channels_vector]) / (
        # self.tuh_filtered_stat_vals['std_vals_filtered'][channels_vector])

        # sample_norm = sample_norm * 2 - 1
        # _, mask = masking(sample_norm)
        if sample_norm.shape[0] < 3000:
            sample_norm = np.pad(sample_norm, ((0, 3000 - sample_norm.shape[0]), (0, 0)))



        if np.random.choice([0, 1], p=[0.7, 0.3]) and not negative:
            index = np.random.choice(self.__len__() - 1)
            negative_sample = self.__getitem__(index, True)
            negative_path = negative_sample['path']
            negative_sample_norm = negative_sample['current'].numpy()

            negative_person = negative_sample['path'].split('/')[-1]  # .split('_')
            current_person = path.split('/')[-1]  # .split('_')
            if negative_person.split('_')[0] == current_person.split('_')[0] and \
                    abs(int(negative_person.split('_')[1][:-4]) - int(current_person.split('_')[1][:-4])) < 20000:
                negative_label = torch.tensor(0)               # возможно стоит запретить позитивы отличающиеся < 20000 , если состояние реально изменилось то сеть будет учиться странному.
            else:
                negative_label = torch.tensor(1)
        else:
            negative_sample_norm = sample_norm.copy()
            negative_label = torch.tensor(0)
            negative_path = ''

        attention_mask = torch.ones(3000)
        attention_mask[real_len:] = 0
        return {'current': torch.from_numpy(sample_norm).float(),
                'negative': torch.from_numpy(negative_sample_norm).float(),
                'path': path,
                'label': negative_label,
                'channels': channels_vector,
                'attention_mask': attention_mask}



if __name__ == '__main__':
    splitted_paths = ['/media/hdd/data/TUH_pretrain.filtered_1_40.v2.splited/{}'.format(i) for i in
                      os.listdir('/media/hdd/data/TUH_pretrain.filtered_1_40.v2.splited/')]
    train_dataset = TEST_TUH(splitted_paths[:-15000]) #,tuh_filtered_stat_vals)
    train_loader = torch.utils.data.DataLoader(train_dataset, .......
            .....
    for batch in train_loader:
        decoded_predict, embedding_before_masking, negative_predict = model_test(
            batch['current'],
            batch['negative'],
            batch['channels'],
            batch['attention_mask'],
            !!!   вот тут хотелось бы отправить еще 2 аргумента
                     1 фичи расчитанные для каждого отрезка  (длинна отрезка и оверлап надо определить, могу сегодня попозже посчитать какие дожны быть и пришлю)
                     2 номер ер кластера к которому принадлежит каждый отрезок
    для однозначности можно  batch['current'] тоже переписать как набор отрезков
            True
        )


