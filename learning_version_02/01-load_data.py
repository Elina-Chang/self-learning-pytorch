import torch
import torchaudio

yesno_data=torchaudio.datasets.YESNO(root="./",url="",folder_in_archive="",download=True)
dataloader=torch.utils.data.DataLoader(yesno_data,batch_size=batch_size,shuffle=True)
for data in dataloader:
    print("Data:",data)
    print("Waveform:{}\nSample rate:{}\nLabel:{}".format(data[0],data[1],data[2]))
    break
import matplotlib.pyplot as plt
print(data[0][0].numpy())
plt.figure()
plt.plot(Waveform.t().numpy())