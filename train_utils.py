in(bcnn):
    print('Training Fine Grained and Coarse Grained Classifiers')

    for i in range(bcnn.options['epochs']):
#         coarse_loss = []
#         num_correct_coarse = 0
#         num_total_coarse = 0
#         coarse_loss = 0
#         for img_dict in bcnn.data_loaders_coarse["train"]:

#             img = img_dict["feature"].to(bcnn.device)
#             coarse_label = (img_dict["coarse_label"]).to(bcnn.device)
#             #a = (img_dict["coarse_label"]-1).squeeze().tolist()
#             #fine_label_names = np.array(bcnn.coarse_classes)[a]

#             bcnn.coarse_solver.zero_grad()


#             coarse_scores = bcnn.coarse_clf(img)
#             batch_size = coarse_scores.size()[0]
#             coarse_scores = coarse_scores.view(batch_size,5)
# #             print(coarse_scores,coarse_label.size())
#             loss = bcnn.criterion(coarse_scores, coarse_label)
#             #coarse_loss+=loss.item()
#             coarse_preds = torch.argmax(coarse_scores,1)
#             #idx = [coarse_preds == coarse_label]

#             #fine_img = img[idx]
#             #fine_label = fine_label[idx]
#             #fine_label_names = fine_label_names[idx]

#             num_total_coarse += coarse_preds.shape[0]
#             num_correct_coarse += torch.sum(coarse_preds == coarse_label)
#             loss.backward()
#             bcnn.coarse_solver.step()
# #             bcnn.coarse_scheduler.step(loss.item())

#         train_acc_coarse = 100 * num_correct_coarse / num_total_coarse    
# #         bcnn.coarse_scheduler.step(train_acc_coarse)

#         if i >=15:
        fine_loss_list = []
        num_total_fine = {}
        num_correct_fine = {}
        for coarse_class,loader in bcnn.data_loaders_fine["train"].items():
            num_total_fine[coarse_class]=0.0
            num_correct_fine[coarse_class] = 0.0
            for img,label in loader:
                img = img.to(bcnn.device)
                fine_label = label.to(bcnn.device)

                bcnn.fine_solver.zero_grad()

                fine_features = bcnn.fine_clf_features(img)
                fine_features_pad = F.pad(fine_features,(1,1,1,1))
                attention_weights_list = [[torch.zeros(fine_features_pad.shape[0]).to(bcnn.device) 
                                     for j in range(fine_features_pad.shape[1])] for i in range(fine_features_pad.shape[2])]
                hidden_states_list = [[torch.zeros(fine_features_pad.shape[0]).to(bcnn.device) 
                                     for j in range(fine_features_pad.shape[1])] for i in range(fine_features_pad.shape[2])]
#                 print(hidden_states_list[0][0])
#                 print(fine_features.shape,attention_weights.shape,hidden_states.shape)
                for i in range(1,fine_features_pad.shape[1]-1):
                    for j in range(1,fine_features_pad.shape[2]-1):
                        hidden = [attention_weights_list[i-1][j],attention_weights_list[i][j-1],
                                  hidden_states_list[i-1][j],hidden_states_list[i][j-1]]
                        feat = torch.cat((fine_features_pad[:,i-1,j-1:j+2],fine_features_pad[:,i,j-1].unsqueeze(1)),1)
                        x,y =  bcnn.attention(feat,hidden)
                        attention_weights_list[i][j] = x
                        hidden_states_list[i][j] = y
#                     x,y = bcnn.attention(feat,hidden)
                attention_weights = torch.stack([torch.stack(attention_weights_list[i]) 
                                                 for i in range(fine_features_pad.shape[1])]).permute(2,0,1)[:,1:-1,1:-1]
                normalized_attn = nn.Softmax(2)(attention_weights.view(*attention_weights.size()[:1], -1)).view_as(attention_weights)
                print(normalized_attn.shape)
                fine_scores = bcnn.fine_clf[coarse_class](fine_features*normalized_attn)
                temp_shape = fine_scores.size()
                fine_scores = fine_scores.view(temp_shape[0],temp_shape[1])
                fine_loss = bcnn.criterion(fine_scores,fine_label)
                fine_loss.backward()  
                fine_loss_list.append(fine_loss.item())
                fine_preds = torch.argmax(fine_scores.data,1)
#                     print(fine_preds,fine_scores)
                bcnn.fine_solver.step()
                num_total_fine[coarse_class] += fine_preds.size(0)
                num_correct_fine[coarse_class] += torch.sum(fine_preds == fine_label).cpu().numpy()   

            val_acc_coarse,val_acc_fine = accuracy(bcnn,"val")
#                 bcnn.fine_scheduler.step(val_acc_fine[coarse_class])

        train_acc_fine = {x : 100.0 * num_correct_fine[x] / num_total_fine[x] for x in num_total_fine.keys()}
        if i % 1 == 0:
#             print('Epoch %d   Train acc coarse %f , Val acc coarse %f' % (i,train_acc_coarse,val_acc_coarse))
            for coarse_class,train_acc in train_acc_fine.items():
                print("Epoch %d, Coarse Class %s, Train acc fine %f, Val Acc fine %f "% (i,coarse_class,train_acc_fine[coarse_class],val_acc_fine[coarse_class]))

    print("End of training")
    print("Final Accuracies")
    for i in ["train","val","test"]:
        acc_coarse,acc_fine = accuracy(bcnn,i)
        print("Final %s accuracy coarse : %f"%(i,acc_coarse))
        for key, item in acc_fine.items():
            print("Final fine-grained %s  accuracy for class %s : %f"%(i,key,item))

def predict(bcnn,x):
    if len(x.size()) == 3:
        print("Single image supplied in this batch")
        x = x.unsqueeze(0)
        coarse_label = torch.argmax(bcnn.coarse_clf(x).data)
        fine_label = None
        #if coarse_classes[i]!= "birds_":
        fine_label  = torch.argmax(bcnn.fine_clf(x,bcnn.coarse_classes[coarse_label]).data)
        return coarse_label, fine_label
    else:
        print("Batch size greater than 1")
        coarse_label = torch.argmax(bcnn.coarse_clf(x).data)
        fine_label = torch.zeros(coarse_label.size())
        for i in range(len(bcnn.coarse_classes)):
            c_size = x.size()
            c_size[0] = torch.sum(coarse_label==i)
            x_c = x[coarse_label==i]
            x_c = x_c.view(c_size)
            fine_label[coarse_label==i] = torch.argmax(bcnn.fine_clf(x_c,bcnn.coarse_classes[i]).data)
        return coarse_label,fine_label

def accuracy(bcnn,mode):
        num_total = 0.0
        num_correct = 0.0
        num_total_fine = {}
        num_correct_fine = {}
        acc_fine = {}
        for coarse_class,loader in bcnn.data_loaders_fine[mode].items():
            num_total_fine[coarse_class]=0.0
            num_correct_fine[coarse_class] = 0.0  

            for img,label in loader:
                img = img.to(bcnn.device)
                fine_label = label.to(bcnn.device)
                coarse_preds = torch.argmax(bcnn.coarse_clf(img)[:,:,0,0],1)
                curr_ind = bcnn.coarse_classes.index(coarse_class)
                num_correct+= torch.sum(coarse_preds==curr_ind).cpu().numpy()
                num_total += fine_label.size()[0]
                fine_label = fine_label[coarse_preds==curr_ind]
                old_img_size = img.size()
                old_label_size = fine_label.size()
#                 old_img_size[0] = num_correct
#                 old_label_size[0] = num_correct
                img = img[coarse_preds == curr_ind]
#                 img = img.view(old_img_size)
                fine_features = bcnn.fine_clf_features(img)
                fine_preds = torch.argmax(bcnn.fine_clf[coarse_class](fine_features),1)
                num_total_fine[coarse_class] += coarse_preds.size()[0]
                num_correct_fine[coarse_class] += torch.sum(fine_preds == fine_label).cpu().numpy()  
            acc_fine[coarse_class] = 100 * num_correct_fine[coarse_class] / num_total_fine[coarse_class]
        acc_coarse = 100*num_correct/num_total
        return acc_coarse,acc_fine

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
def load_checkpoint(file,model,optimizer,best_prec1=None):
    if os.path.isfile(file):
        print("=> loading checkpoint '{}'".format(file))
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch']
#         best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(file, checkpoint['epoch']))
        return start_epoch
    else:
        print("=> no checkpoint found at '{}'".format(file))
        return 0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
