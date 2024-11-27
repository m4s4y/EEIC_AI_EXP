from dataset import MyDataset 

if(__name__ == '__main__'):
    opt = __import__('options')
    dataset = MyDataset(opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train')
    
    print(dataset.sil_index)