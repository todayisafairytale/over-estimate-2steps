import torch
import torch.nn as nn
import torch.optim as optim

from ...._base import _CognitiveDiagnosisModel
from ....datahub import DataHub
from ....interfunc import NCD_IF
from ....extractor import Default
from ....extractor import Over_estimate


class NCDM(_CognitiveDiagnosisModel):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, save_flag=False,first_ever=False,step='step1'):
        """
        Description:
        NCDM ...

        Parameters:
        student_num: int type
            The number of students in the response logs
        exercise_num: int type
            The number of exercises in the response logs
        knowledge_num: int type
            The number of knowledge concepts in the response logs
        method: Ignored
            Not used, present here for API consistency by convention.
        """
        super().__init__(student_num, exercise_num, knowledge_num, save_flag)
        self.first_ever=first_ever
        self.step=step

    def build(self, hidden_dims: list = None, dropout=0.5, device="cpu", dtype=torch.float32, **kwargs):
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.extractor = Over_estimate(
            student_num=self.student_num,
            exercise_num=self.exercise_num,
            knowledge_num=self.knowledge_num,
            device=device,
            dtype=dtype,
            first_ever=self.first_ever,
            step=self.step
        )
        if not self.first_ever:
            self.extractor.load_state_dict(torch.load("D:/Git/Over-estimate/extractor_after_step1.pth"))

        self.inter_func = NCD_IF(knowledge_num=self.knowledge_num,
                                 hidden_dims=hidden_dims,
                                 dropout=dropout,
                                 device=device,
                                 dtype=dtype,step=self.step)
        
        if not self.first_ever:
            self.inter_func.load_state_dict(torch.load("D:/Git/Over-estimate/interfunc_after_step1.pth"))
    
    def train(self, datahub: DataHub, set_type="train", valid_set_type="valid",
              valid_metrics=None, epoch=10, lr=2e-3,  batch_size=256):
        if valid_metrics is None:
            valid_metrics = ["acc", "auc"]
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam([{'params': self.extractor.parameters(),
                                 'lr': lr},
                                {'params': self.inter_func.parameters(),
                                 'lr': lr}])
        for epoch_i in range(0, epoch):
            print("[Epoch {}]".format(epoch_i + 1))
            self._train(datahub=datahub, set_type=set_type,
                        valid_set_type=valid_set_type, valid_metrics=valid_metrics,
                        batch_size=batch_size, loss_func=loss_func, optimizer=optimizer)
        if self.save_flag:
            self.save("D:/Git/Over-estimate/extractor_after_step1.pth","D:/Git/Over-estimate/interfunc_after_step1.pth")
    
    def predict(self, datahub: DataHub, set_type, batch_size=256, **kwargs):
        return self._predict(datahub=datahub, set_type=set_type, batch_size=batch_size)

    def score(self, datahub: DataHub, set_type, metrics: list, batch_size=256, **kwargs) -> dict:
        if metrics is None:
            metrics = ["acc","auc"]
        return self._score(datahub=datahub, set_type=set_type, metrics=metrics, batch_size=batch_size)

    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])

    def load(self, ex_path: str, if_path: str):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        self.extractor.load_state_dict(torch.load(ex_path))
        self.inter_func.load_state_dict(torch.load(if_path))

    def save(self, ex_path: str, if_path: str):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        torch.save(self.extractor.state_dict(), ex_path)
        torch.save(self.inter_func.state_dict(), if_path)

    def get_attribute(self, attribute_name):
        if attribute_name == 'mastery':
            return self.diagnose().detach().cpu().numpy()
        elif attribute_name == 'diff':
            return self.inter_func.transform(self.extractor["diff"]).detach().cpu().numpy()
        elif attribute_name == 'knowledge':
            return self.extractor["knowledge"].detach().cpu().numpy()
        elif attribute_name == 'over_estimate':
            return self.extractor["over_estimate"].detach().cpu().numpy()
        else:
            return None
