import torch
import os, cv2, random
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import train_test_split

seed_num = 42
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)


def seed_worker(worker_id):
    numpy.random.seed(seed_num)
    random.seed(seed_num)

g = torch.Generator()
g.manual_seed(seed_num)


class Wafer_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def __getitem__(self, index):
        return self.data[index], self.label[index]
        
    def __len__(self):
        return len(self.data)
        
        

class Dataset:
    def __init__(self, dataset, _batch_size, augmentation, percentSeenCls, splitRatio):
        super(Dataset, self).__init__()
                
        if dataset == "wafer":
            print("Loading WM-811k Wafer Data ...", "\n")
            
            # Loading WM-811k data
            df = pd.read_pickle("LSWMD.pkl")

            trte = []
            for j in df["trianTestLabel"]:
              try:
                trte.append(j[0][0])
              except:
                trte.append(np.nan)
            df["trianTestLabel"] = trte
            
            
            ft = []
            for j in df["failureType"]:
              try:
                ft.append(j[0][0])
              except:
                ft.append(np.nan)
            df["failureType"] = ft
            
            
            """
            Mapping :
            
            'Center':0, 'Donut':1, 'Edge-Loc':2, 'Edge-Ring':3, 'Loc':4, 'Random':5, 'Scratch':6, 'Near-full':7, 'none':8
            
            'Training':0,'Test':1
            
            """
            
            df['failureNum'] = df.failureType
            df['trainTestNum'] = df.trianTestLabel
            mapping_type = {'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
            mapping_traintest = {'Training':0,'Test':1}
            df = df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})
            
            df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)].reset_index(drop=True)
            df_withlabel['failureNum'] = df_withlabel['failureNum'].astype("int")
            
            # Wafer data
            wafer_data = df_withlabel[df_withlabel["trainTestNum"]==0].reset_index(drop=True)
            
            # Random sampling without replacement
            wafer_data = wafer_data.sample(n=len(wafer_data), replace=False, random_state=1)
            
            print("SPLITRATIO:", splitRatio)
            X_train, X_test, y_train, y_test = [], [], [], []

            if (splitRatio == '8:2') or (splitRatio == '8:1:1'):
                cls_map_num = {0:2767, 1:329, 2:1958, 3:6802, 4:1311, 5:498, 6:413, 7:49, 8:29357}
            elif splitRatio == '7:3':
                cls_map_num = {0:2489, 1:296, 2:1762, 3:6122, 4:1180, 5:448, 6:372, 7:46, 8:25400}
            
            for cls in range(len(cls_map_num)):
              X_cls = wafer_data[wafer_data['failureNum'] == cls]
            
              for x,y in zip(X_cls["waferMap"].values[:cls_map_num[cls]], X_cls["failureNum"].values[:cls_map_num[cls]]):
                X_train.append(x)
                y_train.append(y)
            
              for x,y in zip(X_cls["waferMap"].values[cls_map_num[cls]:], X_cls["failureNum"].values[cls_map_num[cls]:]):
                X_test.append(x)
                y_test.append(y)
                
            
                
            numOfSeenCls = int(np.round((percentSeenCls/100) * 9))
            seen_cls = range(numOfSeenCls)
            unseen_cls = range(numOfSeenCls, 9)
                
            """
            Augmenting same number of samples as mentioned in below paper.
            
            M. B. Alawieh, D. Boning, and D. Z. Pan, “Wafer map defect patterns
            classification using deep selective learning,” in Proc. ACM/IEEE Design
            Automation Conf. (DAC), 2020, pp. 1–6.
            """
            print("Augmentation:", bool(augmentation))
            print("Generating augmented samples ...", "\n")
            if augmentation:
                
                aug_map_cls_fold =  {'0':2, '1':23, '2':4, '4':8, '5':16, '6':20, '7':166}
                aug_map_cls_oprnd = {'0':7, '1':284, '2':-122, '4':-135, '5':-184, '6':-273, '7':-19}
                            
                if (splitRatio == '8:2') or (splitRatio == '8:1:1'):
                    aug_map_cls_added_samples = {'0':5541, '1':7851, '2':7710, '3':3884, '4':10353, '5':7784, '6':7987, '7':8115}
                elif splitRatio == '7:3':
                    aug_map_cls_added_samples = {'0':4986, '1':7065, '2':6939, '3':3495, '4':9317, '5':7005, '6':7188, '7':7303}
                
                
                X_train_aug = []
                y_train_aug = []
                
                indices_cls_3 = np.where(np.array(y_train) == 3)[0]
                X_aug, y_aug = [], []
                np.random.shuffle(indices_cls_3)
                
                for ix in indices_cls_3:
                  X_train_aug.append(X_train[ix])
                  y_train_aug.append(3)
                
                for idx in indices_cls_3[:aug_map_cls_added_samples['3']]:
                  img = X_train[idx]
                  aug_images = self.AugmentedImages(1, img)
                  for aug_img in aug_images:
                    X_aug.append(aug_img.squeeze())
                    y_aug.append(3)
                
                
                X_train_aug += X_aug
                y_train_aug += y_aug
                
                
                for cls in aug_map_cls_fold:
                
                  indices_cls = np.where(np.array(y_train) == int(cls))[0]
                  fold = aug_map_cls_fold[cls]
                  operand = aug_map_cls_oprnd[cls]
                  X_aug, y_aug = [], []
                  for idx in indices_cls:
                    img = X_train[idx]
                    aug_images = self.AugmentedImages(fold, img)
                    
                    for aug_img in aug_images:
                      X_aug.append(aug_img.squeeze())
                      y_aug.append(int(cls))
                
                    X_aug.append(img)
                    y_aug.append(int(cls))
                
                  if operand > 0:
                    np.random.shuffle(indices_cls)
                    for idx in indices_cls[:operand]:
                      img = X_train[idx]
                      aug_images = self.AugmentedImages(1, img)
                
                      for aug_img in aug_images:
                        X_aug.append(aug_img.squeeze())
                        y_aug.append(int(cls))
                  else:
                    X_aug = X_aug[:operand]
                    y_aug = y_aug[:operand]
                
                
                  X_train_aug += X_aug
                  y_train_aug += y_aug
                
                
                indices_cls_8 = np.where(np.array(y_train) == 8)[0]
                for ix in indices_cls_8:
                  X_train_aug.append(X_train[ix])
                  y_train_aug.append(8)
                
                y_train_aug = np.array(y_train_aug)
                
            
                y_train_aug_index = np.random.permutation(len(y_train_aug))
                wafer_tr_data = self.preprocess_images(X_train_aug)
                wafer_tr_data = wafer_tr_data[y_train_aug_index]
                
                
                wafer_tr_label = y_train_aug[y_train_aug_index]
                if list(unseen_cls) != []:
                    wafer_tr_data, wafer_tr_label, _, _ = self.getSeenUnseen(wafer_tr_data, wafer_tr_label, seen_cls, unseen_cls)
                    print("wafer_tr_label", np.unique(wafer_tr_label, return_counts=True))
                wafer_tr_label = torch.tensor(self.OneHotEncoding(wafer_tr_label)).float()
        
            
            else:
                y_train_ran_index = np.random.permutation(len(y_train))
                wafer_tr_data = self.preprocess_images(X_train)
                wafer_tr_data = wafer_tr_data[y_train_ran_index]
                

                wafer_tr_label = np.array(y_train)[y_train_ran_index]
                if list(unseen_cls) != []:
                    wafer_tr_data, wafer_tr_label, _, _ = self.getSeenUnseen(wafer_tr_data, wafer_tr_label, seen_cls, unseen_cls)
                    print("wafer_tr_label", np.unique(wafer_tr_label, return_counts=True))
                wafer_tr_label = torch.tensor(self.OneHotEncoding(wafer_tr_label)).float()
                
                
            wafer_test_data = self.preprocess_images(X_test)
            wafer_test_label = np.array(y_test)
            
            dataset_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.30001,), (0.1922,)) ]) 
            
            
            if list(unseen_cls) != []:
                wafer_seen_test_data, wafer_seen_test_label, wafer_unseen_test_data, wafer_unseen_test_label = self.getSeenUnseen(wafer_test_data, wafer_test_label, seen_cls, unseen_cls)
            
                wafer_test_data = np.concatenate([wafer_seen_test_data, wafer_unseen_test_data], axis = 0)
                wafer_test_label = np.concatenate([wafer_seen_test_label, wafer_unseen_test_label], axis = 0)
                numSeen_te_Samples = len(wafer_seen_test_data)
                wafer_val_label = torch.tensor(self.OneHotEncoding(wafer_seen_test_label)).float()
                
            if (splitRatio == '8:1:1'):
                wafer_test_data, wafer_val_data, wafer_test_label, wafer_val_label = train_test_split(wafer_test_data, wafer_test_label, test_size=0.50, random_state=42)
                wafer_val_label = torch.tensor(self.OneHotEncoding(wafer_val_label)).float()
                wafer_val_data = dataset_transform(wafer_val_data).permute(1,0,2)
                wafer_val_data = wafer_val_data[:, None, :, :]
                print("Validation data :", wafer_val_data.shape)
 
            
            wafer_test_label = torch.tensor(self.OneHotEncoding(wafer_test_label)).float()
            wafer_tr_data = dataset_transform(wafer_tr_data).permute(1,0,2)
            wafer_test_data = dataset_transform(wafer_test_data).permute(1,0,2)
            wafer_tr_data = wafer_tr_data[:, None, :, :]
            wafer_test_data = wafer_test_data[:, None, :, :]
            
            print("Training data :", wafer_tr_data.shape, "Test data :", wafer_test_data.shape)
            print()
            
            self.train_loader = DataLoader(dataset = Wafer_dataset(wafer_tr_data, wafer_tr_label), batch_size=_batch_size, shuffle=True,
            worker_init_fn=seed_worker, generator=g)
            if list(unseen_cls) != []:
                self.val_loader = DataLoader(dataset = Wafer_dataset(wafer_test_data[:numSeen_te_Samples], wafer_val_label), batch_size=_batch_size, shuffle=False)
                self.test_loader = DataLoader(dataset = Wafer_dataset(wafer_test_data, wafer_test_label), batch_size=_batch_size, shuffle=False)
                
            elif (splitRatio == '8:1:1'):
                self.val_loader = DataLoader(dataset = Wafer_dataset(wafer_val_data, wafer_val_label), batch_size=_batch_size, shuffle=False)
                self.test_loader = DataLoader(dataset = Wafer_dataset(wafer_test_data, wafer_test_label), batch_size=_batch_size, shuffle=False)
                
            else:
                self.test_loader = DataLoader(dataset = Wafer_dataset(wafer_test_data, wafer_test_label), batch_size=_batch_size, shuffle=False)
                
            
        elif dataset == 'MixedWM38':
            pass
        
        
    def preprocess_images(self, images):
        TARGET_SIZE = (36, 36)
        images_ = []
        for img in images:
            image = cv2.resize(img/img.max(), dsize=(TARGET_SIZE[0], TARGET_SIZE[1]), interpolation=cv2.INTER_CUBIC)
            images_.append(image)
            
        return np.asarray(images_).astype("float32")
        
        
    def AugmentedImages(self, fold, img):
        aug_images = []

        while fold != len(np.unique(aug_images, axis=0)):
            angle = np.random.randint(0, 361)
            aug_images.append(self.Transformations(angle, img))
        
        len_aug_img = len(np.unique(aug_images, axis=0))
        
        assert fold == len_aug_img
        
        return np.unique(aug_images, axis=0)


    def OneHotEncoding(self,y):
        y1 = np.zeros((y.size, y.max() + 1))
        y1[np.arange(y.size), y] = 1
        
        return y1.astype('int')
        
    
    def getSeenUnseen(self, X, Y, seen_cls, unseen_cls):
        seen_cls_mask = np.in1d(Y, seen_cls)
        unseen_cls_mask = np.in1d(Y, unseen_cls)
        to_seen_cls = {j:i for i, j in enumerate(seen_cls)}
        unseen_cls_label = len(seen_cls)
        to_unseen_cls = {j:unseen_cls_label for j in unseen_cls}
        return X[seen_cls_mask], np.vectorize(to_seen_cls.get)(Y[seen_cls_mask]), X[unseen_cls_mask], np.vectorize(to_unseen_cls.get)(Y[unseen_cls_mask])
        

    def Transformations(self, angle, img):
        
        t1 = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(angle), transforms.RandomHorizontalFlip(),])
        t2 = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(angle),])
        t3 = transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip(0.25), ])
        t4 = transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip(0.65), transforms.RandomAffine(angle)])
        t5 = transforms.Compose([transforms.ToTensor(),transforms.RandomRotation(angle), transforms.RandomAffine(angle)])
        t6 = transforms.Compose([transforms.ToTensor(),transforms.RandomRotation(angle), transforms.RandomHorizontalFlip(0.45), transforms.RandomAffine(angle)])
        t7 = transforms.Compose([transforms.ToTensor(),transforms.RandomRotation(angle), transforms.RandomVerticalFlip(0.2), transforms.RandomAffine(angle)])
        t8 = transforms.Compose([transforms.ToTensor(), transforms.RandomVerticalFlip(0.35), transforms.RandomAffine(angle)])
        t9 = transforms.Compose([transforms.ToTensor(), transforms.RandomVerticalFlip(0.05),])
        t10 = transforms.Compose([transforms.ToTensor(), transforms.RandomVerticalFlip(0.25),transforms.RandomRotation(angle), ])
        t11 = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.25), transforms.RandomRotation(angle), transforms.RandomVerticalFlip(0.75),])
        t12 = transforms.Compose([transforms.ToTensor(), transforms.RandomVerticalFlip(0.15), transforms.RandomHorizontalFlip(0.45)])
        t13 = transforms.Compose([transforms.ToTensor(), transforms.RandomAffine(angle),])
        
        transformations = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13]
        
        return transformations[np.random.randint(0, 13)](img).numpy()