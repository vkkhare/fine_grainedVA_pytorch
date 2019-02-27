
class BCNNWrapper(object):
    def __init__(self,d_loaders_coarse,d_loaders_fine,coarse_class,fine_class,options):
        super(BCNNWrapper,self).__init__()
        self.device = torch.device('cuda:3')
        #self.dsets_coarse = dsets_coarse
        #self.dsets_fine = dsets_fine
        self.options = options
        self.coarse_classes = coarse_class
        self.fine_classes = fine_class
        #{x: len(items) for x, items in dsets_coarse["train"].description_dict.items()}
        self.data_loaders_coarse = d_loaders_coarse
        #{x:torch.utils.data.DataLoader(dsets_coarse[x],batch_size=self.options["batch_size"], shuffle=True, num_workers=4,pin_memory=True) for x in dsets_coarse.keys()}
        self.data_loaders_fine = d_loaders_fine
        #for x,item  in dsets_fine.items():
            #self.data_loaders_fine[x] = {y : torch.utils.data.DataLoader(fdset,batch_size=self.options["batch_size"]/2, shuffle=True, num_workers=4,pin_memory=True)}for y,fdset in item.items()}
        self.coarse_clf = CoarseClf(len(self.coarse_classes)).to(self.device)
        self.fine_clf_features = FineClf().to(self.device)
        self.attention = LSTM_Attention(4,1).to(self.device)
        self.fine_clf = {}
        for coarse_class,num_classes in self.fine_classes.items():
            self.fine_clf[coarse_class] = FineClf_FC_layer(512,num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        coarse_params_features = list(filter(lambda p :p.requires_grad, self.coarse_clf.features.parameters()))
        coarse_params_clf = list(filter(lambda p :p.requires_grad, self.coarse_clf.classifier.parameters()))
        coarse_params = coarse_params_clf + coarse_params_features
        
        self.coarse_solver = torch.optim.Adam(
                coarse_params, lr=self.options['base_lr']*1e-2, weight_decay=self.options['weight_decay'])
#         self.coarse_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         self.coarse_solver, mode='max', factor=0.5, patience=40, verbose=True,threshold=1e-8)
        fine_params_model1 = list(filter(lambda p :p.requires_grad, self.fine_clf_features.model1.parameters()))
        fine_params_model2 = list(filter(lambda p :p.requires_grad, self.fine_clf_features.model2.parameters()))
        fine_params_attention = list(filter(lambda p :p.requires_grad, self.attention.parameters()))
        fine_params_clf = []
        for i,clf in self.fine_clf.items():
            fine_params_clf = fine_params_clf + list(filter(lambda p :p.requires_grad, clf.parameters()))
        fine_params = fine_params_model1 + fine_params_model2 + fine_params_clf + fine_params_attention
        self.fine_solver = torch.optim.Adam(fine_params, lr=self.options['base_lr'],
            weight_decay=self.options['weight_decay'])
#         self.fine_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         self.fine_solver, mode='max', factor=0.8, patience=40, verbose=True,threshold=1e-8)
        
