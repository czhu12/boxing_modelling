from model import RNNClassifier
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import cv
import pandas as pd
import pdb
import plac


class VideoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, video_size:Tuple[int]=(60, 3, 224, 224)):
        self.df = df
        self.video_size = video_size

    def __getitem__(self, index):
        path = self.df.iloc[index]['path']
        video = self._load_video_from_path(path)

        label = self.df.iloc[index]['label']
        return video, label

    def __len__(self):
        return len(self.df)

    def _load_video_from_path(self, path):
        capture = cv.CaptureFromFile(path)
        frames = []
        for i in range(self.video_size[0]):
            img = cv.QueryFrame(capture)
            tmp = cv.CreateImage(cv.GetSize(img),8,3)
            cv.CvtColor(img,tmp,cv.CV_BGR2RGB)
            frames.append(asarray(cv.GetMat(tmp)))
        frames = array(frames)

        assert frames[1:].shape == self.video_size
        return frames

def main(dataset_csv: ("Dataset CSV", 'option', 'd')):
    classifier = RNNClassifier(1000, 256, 3, 0.2, 2)
    df = pd.read_csv(dataset_csv)
    dataset = VideoDataset(df)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=10,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i, (images, labels) in enumerate(mn_dataset_loader):
        images = Variable(images)
        labels = Variable(labels)
        # Clear gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Calculate loss
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        break


    videos = Variable(torch.randn(2, 3, 3, 224, 224))
    outputs = classifier(videos)


