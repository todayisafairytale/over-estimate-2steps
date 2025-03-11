import torch
import torch.nn as nn

from .._base import _Extractor


class Over_estimate(_Extractor, nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, device, dtype,step='step1',first_ever=False,latent_dim=None):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.device = device
        self.dtype = dtype
        self.first_ever=first_ever
        self.latent_dim = knowledge_num
        self.theta=nn.Parameter(torch.zeros(student_num,1)).to(device)
        
        self.__diff_emb = nn.Embedding(self.exercise_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__disc_emb = nn.Embedding(self.exercise_num, 1, dtype=self.dtype).to(self.device)
        self.__map = {
            "over_estimate":self.theta,
            "diff": self.__diff_emb.weight,
            "disc": self.__disc_emb.weight,
        }
        self.apply(self.initialize_weights)

        if step=='step2':
            self.__diff_emb.weight.requires_grad=False
            self.__disc_emb.weight.requires_grad=False
        elif step=='step1':
            self.theta.requires_grad=False

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def extract(self,student_id,exercise_id,S,theta_tuda):
        Ones=torch.ones(S.shape)
        if self.first_ever:
            self.theta[student_id]=theta_tuda
        student_ts = S+self.theta[student_id]*(Ones-S)
        diff_ts = self.__diff_emb(exercise_id)
        disc_ts = self.__disc_emb(exercise_id)
        return student_ts, diff_ts, disc_ts

    def __getitem__(self, item):
        if item not in self.__map.keys():
            raise ValueError("We can only detach {} from embeddings.".format(self.__map.keys()))
        self.__map["over_estimate"] = self.theta
        self.__map["diff"] = self.__diff_emb.weight
        self.__map["disc"] = self.__disc_emb.weight
        return self.__map[item]
    
    def clamp_theta(self):
        with torch.no_grad():
            self.theta.data = torch.clamp(self.theta.data, min=-1, max=1)

