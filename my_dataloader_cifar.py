from operator import imod
from pkgutil import get_data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from data import cifar


class cifar_dataset(Dataset): 
    def __init__(self, dataset, transform, mode, pred=[], probability=[]): 
        
        self.transform = transform
        self.mode = mode  
     
        if self.mode=='test':
            if dataset=='cifar10':            
                cifar10 = cifar.CIFAR10(root='~/data/',
                                        download=True,
                                        train=False,
                                        noise_type = None,
                                        noise_path = None
                                        )
                self.test_data, self.test_label =  cifar10.get_data()
            elif dataset=='cifar100':
                cifar100 = cifar.CIFAR100(root='~/data/',
                                        download=True,
                                        train=False,
                                        noise_type = 'noisy_label',
                                        noise_path = 'C:\\Users\\wang\\Desktop\\DivideMix-master\\data\\CIFAR-100_human.pt'
                                        )
                self.test_data, self.test_label = cifar100.get_data()
        else: 
            if dataset=='cifar10': 
                cifar10 = cifar.CIFAR10(root='~/data/',
                                        download=True,
                                        train=True,
                                        noise_type = None,
                                        noise_path = None
                                        )
                train_data, train_label = cifar10.get_data()
            elif dataset=='cifar100':    
                cifar100 = cifar.CIFAR100(root='~/data/',
                                        download=True,
                                        train=True,
                                        noise_type = 'noisy_label',
                                        noise_path = 'C:\\Users\\wang\\Desktop\\DivideMix-master\\data\\CIFAR-100_human.pt'
                                        )
                train_data, train_label = cifar100.get_data()
            
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = train_label
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]         
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = [train_label[i] for i in pred_idx]         
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob        
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)       
        
        
class cifar_dataloader():  
    def __init__(self, dataset, batch_size, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
    def run(self,mode,pred=[], prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, transform=self.transform_train, mode="all")                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, transform=self.transform_train, mode="labeled", pred=pred, probability=prob)             
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, transform=self.transform_train, mode="unlabeled", pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, transform=self.transform_test, mode='all')
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)  
            return eval_loader    

if __name__ == '__main__':
    transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
    d = cifar_dataset('cifar100', transform_train, mode='all')
    d = iter(d)
    for i in range(100):
        img, label, _ = next(d)
        print(label)