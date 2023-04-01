import numpy as np

class Edgefeature():
    def __init__(self,el: list, length: int):
        self.edataDic = dict(el)
        self.length = length

    def get_edata(self):
        return list(self.edataDic.values())
    
    def get_edataM(self):
        edataM = np.zeros((self.length, self.length))
        for i in self.edataDic:
            edataM[i[0]][i[1]] = self.edataDic[i]
        return edataM
