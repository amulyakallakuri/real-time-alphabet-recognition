import torch
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self,model,data_loader,device,loss_func,optimizer, trained_parameters,num_epochs,lr, weight_decay = 0):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.optimizer = optimizer(trained_parameters, lr = lr, weight_decay = weight_decay)
        self.num_epochs = num_epochs
        print(self.model)
    
    def train(self):
        train_acc_list = []
        val_acc_list = []
        for epoch in range(self.num_epochs):
            print("Epoch {}/{}".format(epoch+1,self.num_epochs))
            print("-"*10)

            if (epoch+1)%10==0:
                self.save_network('./temp'+str(int((epoch+1)/10)))

            for phase in ["train","val"]:
                if phase == "train":
                    print("Training...")
                    self.model.train(True)
                else:
                    print("Validing...")
                    self.model.train(False)
                
                running_loss = 0.0
                running_corrects = 0
                batch_size = self.data_loader[phase].batch_size

                for batch, data in enumerate(self.data_loader[phase],1):
                    images,labels = data
                    images,labels = images.to(self.device),labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.loss_func(outputs,labels)
                    predictions = torch.max(outputs,1)[1]
                    self.optimizer.zero_grad()

                    if phase == "train":
                        loss.backward()
                        self.optimizer.step()
                    
                    running_loss += loss.item()
                    running_corrects += torch.sum(predictions == labels)

                    if batch%500 == 0 and phase == "train":
                        print("Batch {}, Train Loss:{:.4f},Train ACC:{:.4f}%".format(
                                batch, running_loss/batch, 100.0*running_corrects/(batch_size*batch)
                                )
                        )

                epoch_loss = running_loss*batch_size/len(self.data_loader[phase].dataset)
                epoch_acc = 100.0 * running_corrects/len(self.data_loader[phase].dataset)
                if phase == 'train':
                    train_acc_list.append(epoch_acc.item())
                else:
                    val_acc_list.append(epoch_acc.item())

                print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase,epoch_loss,epoch_acc))

        plt.plot(train_acc_list,'-rx')
        plt.plot(val_acc_list,'-bo')
        plt.title('training and validation accuracy over epochs')
        plt.legend(['training accuracy','validation accuracy'])
        plt.savefig('acc plot')



        # count = 0
        
        # for phase in ('Train','Valid'):
        #     print(phase+"ing...")
        #     for epoch in range(self.num_epochs):
        #         print("Epoch {}/{}".format(epoch+1,self.num_epochs))
        #         print("-"*10)
        #         running_loss = 0.0
        #         correct = 0
        #         i_batch =0
        #         for images,labels in self.train_data:
        #             i_batch+=1
        #             images,labels = images.to(self.device),labels.to(self.device)
        #             count +=1
        #             outputs = self.model(images)
        #             loss = self.loss_func(outputs,labels)

        #             self.optimizer.zero_grad()
        #             loss.backward()

        #             self.optimizer.step()
                    
        #             running_loss += loss.item()

        #             predictions = torch.max(outputs,1)[1]
        #             correct += (predictions==labels).sum()
        #             if i_batch%500 == 0:
        #                 print("Batch {i_batch}, Train Loss:{:.4f},Train ACC:{:.4f}%".format(
        #                     i_batch, running_loss/i_batch, 100.0*correct/(self.train_data.batch_size*i_batch))
        #             )

        #         epoch_loss = running_loss/self.num_epochs
        #         epoch_acc = 100.0 * correct/len(self.train_data.dataset)
        #         print("{}} Loss:{:.4f} Acc:{:.4f}%".format(phase,epoch_loss,epoch_acc))
    
    def save_network(self, network_path:str):
        torch.save(self.model.state_dict(), network_path)
        ## save params
        # model = TheModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load(PATH))
        # model.eval()